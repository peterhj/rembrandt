use data::{SampleDatum, SampleLabel, DataSource};
use layer::{Layer};
use net::{NetArch};

use async_cuda::context::{DeviceContext};

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
  pub init_step_size: f32,
  pub momentum:       f32,
  pub anneal:         AnnealingPolicy
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

  pub fn momentum(&self, t: usize) -> f32 {
    self.config.momentum
  }
}

pub struct OptState {
  pub epoch:  usize,
  pub t:      usize,
}

pub trait Optimizer {
  fn train(&self, opt_cfg: &OptConfig, state: &mut OptState, arch: &mut NetArch, train_data: &mut DataSource, test_data: &mut DataSource, ctx: &DeviceContext);
  fn eval(&self, opt_cfg: &OptConfig, arch: &mut NetArch, eval_data: &mut DataSource, ctx: &DeviceContext);
}

pub struct SgdOptimizer;

impl Optimizer for SgdOptimizer {
  fn train(&self, opt_cfg: &OptConfig, state: &mut OptState, arch: &mut NetArch, train_data: &mut DataSource, test_data: &mut DataSource, ctx: &DeviceContext) {
    let num_epoch_samples = train_data.len();
    let interval_size = 1000;
    let schedule = DescentSchedule::new(*opt_cfg);
    loop {
      let mut epoch_correct = 0;
      let mut interval_correct = 0;
      for layer in arch.hidden_layers() {
        layer.reset_gradients(ctx);
      }
      let mut idx: usize = 0;
      loop {
        let (datum, maybe_label) = match train_data.request_sample() {
          Some(x) => x,
          None => break,
        };
        arch.data_layer().load_sample(&datum, ctx);
        arch.loss_layer().load_sample(maybe_label, ctx);
        arch.data_layer().forward(ctx);
        for layer in arch.hidden_layers() {
          layer.forward(ctx);
        }
        arch.loss_layer().forward(ctx);
        if arch.loss_layer().correct_guess(&ctx) {
          epoch_correct += 1;
          interval_correct += 1;
        }
        arch.loss_layer().backward(&schedule, ctx);
        for layer in arch.hidden_layers().iter_mut().rev() {
          layer.backward(&schedule, ctx);
        }
        let next_idx = idx + 1;
        if next_idx % interval_size == 0 {
          println!("DEBUG: interval: {}/{} train accuracy: {:.3}",
              next_idx, num_epoch_samples,
              interval_correct as f32 / interval_size as f32);
          interval_correct = 0;
          self.eval(opt_cfg, arch, test_data, ctx);
        }
        if next_idx % opt_cfg.minibatch_size == 0 {
          for layer in arch.hidden_layers() {
            layer.descend(&schedule, state.t, ctx);
            layer.reset_gradients(ctx);
          }
          state.t += 1;
        }
        idx += 1;
      }
      println!("DEBUG: epoch: {} train accuracy: {:.3}", state.epoch, epoch_correct as f32 / num_epoch_samples as f32);
      state.epoch += 1;
      train_data.reset();
    }
  }

  fn eval(&self, opt_cfg: &OptConfig, arch: &mut NetArch, eval_data: &mut DataSource, ctx: &DeviceContext) {
    let num_epoch_samples = eval_data.len();
    let mut epoch_correct = 0;
    let mut idx: usize = 0;
    loop {
      let (datum, maybe_label) = match eval_data.request_sample() {
        Some(x) => x,
        None => break,
      };
      arch.data_layer().load_sample(&datum, ctx);
      arch.loss_layer().load_sample(maybe_label, ctx);
      arch.data_layer().forward(ctx);
      for layer in arch.hidden_layers() {
        layer.forward(ctx);
      }
      arch.loss_layer().forward(ctx);
      if arch.loss_layer().correct_guess(&ctx) {
        epoch_correct += 1;
      }
      idx += 1;
    }
    println!("DEBUG: test accuracy: {:.3}", epoch_correct as f32 / num_epoch_samples as f32);
    eval_data.reset();
  }
}
