use operator::{FullOperator};

pub mod cg;
//pub mod lanczos;

pub trait IterativeSolveStep {
  fn step(&self, batch_size: usize, operator: &mut FullOperator);
}

#[derive(Clone, Copy)]
pub struct FisherVectorProduct;

impl IterativeSolveStep for FisherVectorProduct {
  fn step(&self, batch_size: usize, operator: &mut FullOperator) {
    operator.r_forward(batch_size);
    operator.set_target_factors_with_rfwd_targets(batch_size);
    operator.backward(batch_size);
    operator.reset_target_factors(batch_size);
  }
}

#[derive(Clone, Copy)]
pub struct IterativeSolveConfig<Iteration> where Iteration: IterativeSolveStep {
  pub params_len:   usize,
  pub max_iters:    usize,
  pub iteration:    Iteration,
}
