use solve::{IterativeSolveStep, IterativeSolveConfig};

pub struct ConjGradSolve<Iteration> where Iteration: IterativeSolveStep {
  config:   IterativeSolveConfig<Iteration>,
}
