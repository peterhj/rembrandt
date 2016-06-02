use operator::{OpRead, OpWrite, OpPhase};
use opt::sgd::parallel::{ParallelSgdOptWorker};

pub mod cg;
pub mod lanczos;

pub trait ParallelIterativeSolveStep {
  fn init(&self, phase: OpPhase, worker: &mut ParallelSgdOptWorker);
  fn step(&self, worker: &mut ParallelSgdOptWorker, reader: &mut OpRead, writer: &mut OpWrite);
}

pub struct ParallelFisherVectorProduct;

impl ParallelIterativeSolveStep for ParallelFisherVectorProduct {
}
