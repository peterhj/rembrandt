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
  fn eval(&self, arch: &mut NetArch, eval_data: &mut DataSource, ctx: &DeviceContext);
}

pub struct SgdOptimizer;

impl Optimizer for SgdOptimizer {
  fn train(&self, opt_cfg: &OptConfig, state: &mut OptState, arch: &mut NetArch, train_data: &mut DataSource, test_data: &mut DataSource, ctx: &DeviceContext) {
    let epoch_size = (train_data.len() / opt_cfg.minibatch_size) * opt_cfg.minibatch_size;
    let descent = DescentSchedule::new(*opt_cfg);
    let mut start_time = get_time();
    let mut interval_correct = 0;
    let mut idx = 0;
    loop {
      // XXX(20151002): gradients are initialized to zero.
      train_data.each_sample(&mut |epoch_idx, datum, maybe_label| {
        if epoch_idx >= epoch_size {
          return;
        }
        arch.data_layer().load_sample(datum, ctx);
        arch.loss_layer().load_sample(maybe_label, ctx);
        arch.data_layer().forward(OptPhase::Training, ctx);
        for layer in arch.hidden_layers() {
          layer.forward(OptPhase::Training, ctx);
        }
        arch.loss_layer().forward(OptPhase::Training, ctx);
        if arch.loss_layer().correct_guess(&ctx) {
          interval_correct += 1;
        }
        arch.loss_layer().backward(&descent, ctx);
        for layer in arch.hidden_layers().iter_mut().rev() {
          layer.backward(&descent, ctx);
        }
        idx += 1;
        if idx % opt_cfg.interval_size == 0 {
          let elapsed_time = get_time();
          let elapsed_ms = (elapsed_time - start_time).num_milliseconds();
          start_time = elapsed_time;
          println!("DEBUG: interval: {}/{} train accuracy: {:.3} elapsed: {:.3} s",
              epoch_idx + 1, epoch_size,
              interval_correct as f32 / opt_cfg.interval_size as f32,
              elapsed_ms as f32 * 0.001,
          );
          interval_correct = 0;
        }
        if idx % (100 * opt_cfg.interval_size) == 0 {
          self.eval(arch, test_data, ctx);
        }
        if idx % opt_cfg.minibatch_size == 0 {
          for layer in arch.hidden_layers() {
            layer.descend(&descent, state.t, ctx);
            layer.reset_gradients(&descent, ctx);
          }
          state.t += 1;
        }
      });
      state.epoch += 1;
    }
  }

  fn eval(&self, arch: &mut NetArch, eval_data: &mut DataSource, ctx: &DeviceContext) {
    let epoch_size = eval_data.len();
    let mut epoch_correct = 0;
    eval_data.each_sample(&mut |_, datum, maybe_label| {
      arch.data_layer().load_sample(datum, ctx);
      arch.loss_layer().load_sample(maybe_label, ctx);
      arch.data_layer().forward(OptPhase::Evaluation, ctx);
      for layer in arch.hidden_layers() {
        layer.forward(OptPhase::Evaluation, ctx);
      }
      arch.loss_layer().forward(OptPhase::Evaluation, ctx);
      if arch.loss_layer().correct_guess(&ctx) {
        epoch_correct += 1;
      }
    });
    println!("DEBUG: test accuracy: {:.3}", epoch_correct as f32 / epoch_size as f32);
  }
}
