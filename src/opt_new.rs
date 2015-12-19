use arch_new::{ArchWorker};

#[derive(Clone, Copy)]
pub struct SgdOptConfig {
  pub minibatch_size: usize,
  pub learning_rate:  LearningRateSchedule,
  pub momentum:       f32,
  pub l2_reg_coef:    f32,

  pub display_iters:  usize,
  pub valid_iters:    usize,
  pub save_iters:     usize,
}

#[derive(Clone, Copy)]
pub enum LearningRateSchedule {
  Fixed{lr: f32},
  StepOnce{
    lr0: f32, lr0_iters: usize,
    lr1: f32, lr1_iters: usize,
  },
  StepTwice{
    lr0: f32, lr0_iters: usize,
    lr1: f32, lr1_iters: usize,
    lr2: f32, lr2_iters: usize,
  },
}

pub struct SgdOptimization;

impl SgdOptimization {
  pub fn train(&self, config: SgdOptConfig, arch: &mut ArchWorker) {
    // TODO(20151218)
    unimplemented!();
  }
}
