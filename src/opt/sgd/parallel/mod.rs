use operator::{FullOperator};

use std::ops::{Deref, DerefMut};

pub mod dev_allreduce;
//pub mod mpi_dist_allreduce;
//pub mod mpi_dist_dev_allreduce;

pub trait ParallelSgdOptWorker: Deref<Target=FullOperator> + DerefMut {
  fn signal_checkpoint(&mut self);
  fn wait_checkpoint(&mut self) -> bool;

  fn save_params(&mut self);
  fn restore_params(&mut self);

  fn stage_params(&mut self);
  fn merge_params(&mut self);
  fn sync_params(&mut self);

  fn stage_grads(&mut self);
  fn merge_grads(&mut self);
  fn sync_grads(&mut self);
}

#[derive(Clone)]
pub struct ParallelSgdOptConfig {
  pub init:           InitBehavior,
  pub minibatch_size: usize,
  pub step_size:      StepSizeSchedule,
  pub momentum:       Momentum,
  pub l2_reg_coef:    f32,

  pub display_iters:  usize,
  pub checkpoint_iters:   usize,
  pub checkpoint_dir:     PathBuf,
  pub save_iters:     usize,
  pub valid_iters:    usize,
}

pub struct ParallelSgdOpt {
}
