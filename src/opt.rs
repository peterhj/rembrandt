use data::{SampleDatum, SampleLabel, DataSource};
use layer::{Layer};
use net::{NetArch};

use async_cuda::context::{DeviceContext};

use time::{Duration, get_time};

#[derive(Clone, Copy)]
pub enum AnnealingPolicy {
  None,
  Inv{time_scale: f32},
  Step{step_iters: usize, decay: f32},
  StepOnce{step_iters: usize, new_rate: f32},
  StepTwice{first_step: usize, second_step: usize, first_rate: f32, second_rate: f32},
}

impl AnnealingPolicy {
  pub fn scale_step_size(&self, bare_step_size: f32, t: usize) -> f32 {
    match *self {
      AnnealingPolicy::None => {
        bare_step_size
      }
      AnnealingPolicy::Inv{time_scale} => {
        bare_step_size / (1.0 + (t as f32 / time_scale))
      }
      AnnealingPolicy::Step{step_iters, decay} => {
        bare_step_size * decay.powi((t / step_iters) as i32)
      }
      AnnealingPolicy::StepOnce{step_iters, new_rate} => {
        if t < step_iters {
          bare_step_size
        } else {
          new_rate
        }
      }
      AnnealingPolicy::StepTwice{first_step, second_step, first_rate, second_rate} => {
        if t < first_step {
          bare_step_size
        } else if t < second_step {
          first_rate
        } else {
          second_rate
        }
      }
    }
  }
}

#[derive(Clone, Copy)]
pub struct OptConfig {
  pub minibatch_size: usize,
  pub max_iters:      usize,
  pub init_step_size: f32,
  pub momentum:       f32,
  pub l2_reg_coef:    f32,
  pub anneal:         AnnealingPolicy,
  pub interval_size:  usize,
  pub save_iters:     Option<usize>,
}

impl OptConfig {
  pub fn step_size(&self, t: usize) -> f32 {
    self.anneal.scale_step_size(self.init_step_size, t)
  }
}

pub type DescentConfig = OptConfig;

#[derive(Clone, Copy)]
pub struct DescentSchedule {
  config: OptConfig,
}

impl DescentSchedule {
  pub fn new(config: OptConfig) -> DescentSchedule {
    DescentSchedule{config: config}
  }

  pub fn minibatch_size(&self/*, t: usize*/) -> usize {
    self.config.minibatch_size
  }

  pub fn step_size(&self, t: usize) -> f32 {
    self.config.anneal.scale_step_size(self.config.init_step_size, t)
  }

  pub fn momentum(&self/*, t: usize*/) -> f32 {
    self.config.momentum
  }

  pub fn l2_reg_coef(&self/*, t: usize*/) -> f32 {
    self.config.l2_reg_coef
  }
}

#[derive(Clone, Copy)]
pub enum OptPhase {
  Training,
  Evaluation,
}

pub struct OptState {
  pub epoch:  usize,
  pub t:      usize,
}

pub trait Optimizer {
  fn train(&self, opt_cfg: &OptConfig, state: &mut OptState, arch: &mut NetArch, train_data: &mut DataSource, test_data: &mut DataSource, ctx: &DeviceContext);
  fn validate(&self, opt_cfg: &OptConfig, arch: &mut NetArch, eval_data: &mut DataSource, ctx: &DeviceContext);
}

pub struct SgdOptimizer;

impl Optimizer for SgdOptimizer {
  fn train(&self, opt_cfg: &OptConfig, state: &mut OptState, arch: &mut NetArch, train_data: &mut DataSource, test_data: &mut DataSource, ctx: &DeviceContext) {
    let descent = DescentSchedule::new(*opt_cfg);
    let epoch_size = (train_data.len() / opt_cfg.minibatch_size) * opt_cfg.minibatch_size;
    let batch_size = arch.batch_size();
    assert!(opt_cfg.minibatch_size >= batch_size);
    assert_eq!(0, opt_cfg.minibatch_size % batch_size);
    let mut start_time = get_time();
    // FIXME(20151022): gradients are initialized to zero; should do it explicitly.
    //arch.initialize_gradients();
    let mut interval_correct = 0;
    let mut interval_total = 0;
    let mut idx = 0;
    loop {
      interval_correct = 0;
      interval_total = 0;
      train_data.each_sample(&mut |epoch_idx, datum, maybe_label| {
        if epoch_idx >= epoch_size {
          return;
        }
        let batch_idx = idx % batch_size;
        match datum {
          &SampleDatum::RgbPerChannelBytes(ref frame) => {
            arch.data_layer().preload_frame(batch_idx, frame, ctx);
          }
          _ => unimplemented!(),
        }
        arch.loss_layer().preload_label(batch_idx, maybe_label.unwrap().0, ctx);
        idx += 1;
        if idx % batch_size == 0 {
          arch.data_layer().load_frames(batch_size, ctx);
          arch.loss_layer().load_labels(batch_size, ctx);
          for layer in arch.hidden_layers_forward() {
            layer.forward(OptPhase::Training, batch_size, ctx);
          }
          arch.loss_layer().forward(OptPhase::Training, batch_size, ctx);
          arch.loss_layer().store_labels(batch_size, &ctx);
          interval_correct += arch.loss_layer().count_accuracy(batch_size, &ctx);
          interval_total += batch_size;
          arch.loss_layer().backward(&descent, batch_size, ctx);
          for layer in arch.hidden_layers_backward() {
            layer.backward(&descent, batch_size, ctx);
          }
        }
        //if idx % opt_cfg.interval_size == 0 {
        if idx % (5 * 1024) == 0 { // FIXME(20151016)
          let lap_time = get_time();
          let elapsed_ms = (lap_time - start_time).num_milliseconds();
          start_time = lap_time;
          println!("DEBUG: epoch: {} iter: {} interval: {}/{} train accuracy: {:.3} elapsed: {:.3} s",
              state.epoch, state.t + 1, epoch_idx + 1, epoch_size,
              interval_correct as f32 / interval_total as f32,
              elapsed_ms as f32 * 0.001,
          );
          interval_correct = 0;
          interval_total = 0;
        }
        if idx % opt_cfg.minibatch_size == 0 {
          for layer in arch.hidden_layers_forward() {
            layer.descend(&descent, state.t, ctx);
            layer.reset_gradients(&descent, ctx);
          }
          state.t += 1;
          if let Some(save_iters) = opt_cfg.save_iters {
            if state.t % save_iters == 0 {
              arch.save_layer_params(state.t, ctx);
            }
          }
          // FIXME(20151016): Doing this at the end of a (mini)batch b/c the
          // predicted labels are stored in a buffer in the net itself, and
          // indices get clobbered if we call .validate() in the middle of a
          // batch.
          if (idx / opt_cfg.minibatch_size) % (opt_cfg.interval_size) == 0 {
            self.validate(opt_cfg, arch, test_data, ctx);
          }
        }
      });
      state.epoch += 1;
    }
  }

  fn validate(&self, opt_cfg: &OptConfig, arch: &mut NetArch, eval_data: &mut DataSource, ctx: &DeviceContext) {
    //let epoch_size = eval_data.len();
    let epoch_size = (eval_data.len() / opt_cfg.minibatch_size) * opt_cfg.minibatch_size;
    let batch_size = arch.batch_size();
    let mut epoch_correct = 0;
    let mut epoch_total = 0;
    eval_data.each_sample(&mut |epoch_idx, datum, maybe_label| {
      if epoch_idx >= epoch_size {
        return;
      }
      let batch_idx = epoch_idx % batch_size;
      match datum {
        &SampleDatum::RgbPerChannelBytes(ref frame) => {
          arch.data_layer().preload_frame(batch_idx, frame, ctx);
        }
        _ => unimplemented!(),
      }
      arch.loss_layer().preload_label(batch_idx, maybe_label.unwrap().0, ctx);
      if (epoch_idx + 1) % batch_size == 0 {
        arch.data_layer().load_frames(batch_size, ctx);
        arch.loss_layer().load_labels(batch_size, ctx);
        for layer in arch.hidden_layers_forward() {
          layer.forward(OptPhase::Evaluation, batch_size, ctx);
        }
        arch.loss_layer().forward(OptPhase::Evaluation, batch_size, ctx);
        arch.loss_layer().store_labels(batch_size, &ctx);
        epoch_correct += arch.loss_layer().count_accuracy(batch_size, &ctx);
        epoch_total += batch_size;
      }
    });
    println!("DEBUG: validation accuracy: {:.3}", epoch_correct as f32 / epoch_total as f32);
  }
}
