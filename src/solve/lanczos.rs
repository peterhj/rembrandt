use operator::{OpRead, OpWrite, OpCursor, FullOperator};
use solve::{IterativeSolveStep, IterativeSolveConfig};

use array_cuda::device::context::{DeviceContext};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut};
use array_cuda::device::random::{RandomSampleExt, GaussianDist};

use std::iter::{repeat};
use std::rc::{Rc};

pub struct LanczosSolve<Iteration> where Iteration: IterativeSolveStep {
  config:   IterativeSolveConfig<Iteration>,
  context:  Rc<DeviceContext>,
  r:        OpCursor<DeviceBuffer<f32>>,
  v_curr:   OpCursor<DeviceBuffer<f32>>,
  v_prev:   DeviceBuffer<f32>,
  alphas:   DeviceBuffer<f32>,
  alphas_h: Vec<f32>,
  betas:    DeviceBuffer<f32>,
  betas_h:  Vec<f32>,
}

impl<Iteration> LanczosSolve<Iteration> where Iteration: IterativeSolveStep {
  pub fn new(config: IterativeSolveConfig<Iteration>, context: Rc<DeviceContext>) -> LanczosSolve<Iteration> {
    LanczosSolve{
      config:   config,
      context:  context,
      r:        OpCursor::new(DeviceBuffer::zeros(config.params_len)),
      v_curr:   OpCursor::new(DeviceBuffer::zeros(config.params_len)),
      v_prev:   DeviceBuffer::zeros(config.params_len),
      alphas:   DeviceBuffer::zeros(config.max_iters),
      alphas_h: repeat(0.0).take(config.max_iters).collect(),
      betas:    DeviceBuffer::zeros(config.max_iters),
      betas_h:  repeat(0.0).take(config.max_iters).collect(),
    }
  }

  pub fn solve_condition_number(&mut self, batch_size: usize, operator: &mut FullOperator) -> f32 {
    self.v_prev.as_ref_mut(ctx).set_constant(0.0);
    self.v_curr.as_ref_mut(ctx).sample(&GaussianDist{mean: 0.0, std: 1.0});
    self.betas.as_ref_mut_range(0, 1, ctx).vector_l2_norm(&self.v_curr.as_ref(ctx))
      .sync_store(&mut self.betas_h[0 .. 1]);
    self.v_curr.as_ref_mut(ctx).vector_scale(1.0 / self.betas_h[0]);
    for k in 0 .. self.max_iters {
      operator.read_direction(0, &mut self.v_curr);
      self.iteration.step(batch_size, operator);
      operator.write_direction(0, &mut self.r);
      if k > 0 {
        self.r.as_ref_mut(ctx).vector_sum(-self.betas_h[k], &self.v_prev.as_ref(ctx));
      }
      self.alphas.as_ref_mut_range(k, k+1, ctx).vector_prod(&self.v_curr.as_ref(ctx), &self.r.as_ref(ctx))
        .sync_store(&mut self.alphas_h[k .. k+1]);
      self.r.as_ref_mut(ctx).vector_sum(-self.alphas_h[k], &self.v_curr.as_ref(ctx));
      if k < self.max_iters-1 {
        self.betas.as_ref_mut_range(k+1, k+2, ctx).vector_l2_norm(&self.r.as_ref(ctx));
          .sync_store(&mut self.betas_h[k+1 .. k+2]);
        self.v_prev.as_ref_mut(ctx).copy(&self.v_curr.as_ref(ctx));
        self.v_curr.as_ref_mut(ctx).copy(&self.r.as_ref(ctx));
          .vector_scale(1.0 / self.betas_h[k+1]);
      }
    }

    // FIXME(20160521): now that we have alphas and betas, call out to an
    // eigensolve routine to get the max and min eigenvalues.
    unimplemented!();
  }
}
