use operator::{CompleteOperator, OpPhase};

pub mod cg;
//pub mod lanczos;

pub trait IterativeSolveStep: Copy {
  fn init(&self, batch_size: usize, phase: OpPhase, operator: &mut CompleteOperator);
  fn step(&self, batch_size: usize, operator: &mut CompleteOperator);
}

#[derive(Clone, Copy)]
pub struct FisherVectorProduct;

impl IterativeSolveStep for FisherVectorProduct {
  fn init(&self, batch_size: usize, phase: OpPhase, operator: &mut CompleteOperator) {
    operator.forward(batch_size, phase);
  }

  fn step(&self, batch_size: usize, operator: &mut CompleteOperator) {
    //operator.read_direction(...);
    operator.r_forward(batch_size);
    operator.set_targets_with_r_loss(batch_size);
    operator.backward(batch_size);
    operator.reset_targets(batch_size);
    //operator.write_grad(...);
  }
}

#[derive(Clone, Copy)]
pub struct HessianVectorProduct;

impl IterativeSolveStep for HessianVectorProduct {
  fn init(&self, batch_size: usize, phase: OpPhase, operator: &mut CompleteOperator) {
    operator.forward(batch_size, phase);
    operator.backward(batch_size);
  }

  fn step(&self, batch_size: usize, operator: &mut CompleteOperator) {
    operator.r_forward(batch_size);
    operator.r_backward(batch_size);
  }
}

#[derive(Clone, Copy)]
pub struct IterativeSolveConfig<Iteration> where Iteration: IterativeSolveStep {
  pub params_len:   usize,
  pub max_iters:    usize,
  pub iteration:    Iteration,
}
