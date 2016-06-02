use solve::{IterativeSolveStep, IterativeSolveConfig};

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceContext};
use array_cuda::device::linalg::{};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut};

use std::rc::{Rc};

pub struct CgSolve<Iteration> where Iteration: IterativeSolveStep {
  config:   IterativeSolveConfig<Iteration>,
  context:  Rc<DeviceContext>,
  b:        OpCursor<DeviceBuffer<f32>>,
  x:        OpCursor<DeviceBuffer<f32>>,
  w:        OpCursor<DeviceBuffer<f32>>,
  p:        OpCursor<DeviceBuffer<f32>>,
  r:        DeviceBuffer<f32>,
  rhos:     DeviceBuffer<f32>,
  rhos_h:   Vec<f32>,
  alphas:   DeviceBuffer<f32>,
  alphas_h: Vec<f32>,
}

impl<Iteration> CgSolve<Iteration> where Iteration: IterativeSolveStep {
  pub fn new(config: IterativeSolveConfig<Iteration>, context: Rc<DeviceContext>) -> CgSolve<Iteration> {
    CgSolve{
      config:   config,
    }
  }

  pub fn solve(&mut self, batch_size: usize, phase: OpPhase, operator: &mut CompleteOperator) {
    let ctx = &(*self.context).as_ref();
    self.x.as_ref_mut(ctx).set_constant(0.0);
    operator.reset();
    self.config.iteration.init(batch_size, phase, operator);
    operator.read_direction(0, &mut self.x);
    self.config.iteration.step(batch_size, operator);
    operator.write_grad(0, &mut self.w);
    self.r.as_ref_mut(ctx).copy(&self.b.as_ref(ctx));
    self.r.as_ref_mut(ctx).vector_add(-1.0, &self.w.as_ref(ctx));
    self.rhos.as_ref_mut_range(0, 1, ctx).vector_l2_norm(&self.r.as_ref(ctx));
    self.rhos.as_ref_range(0, 1, ctx).sync_store(&mut self.rhos_h[0 .. 1]);
    for k in 1 .. self.config.max_iters+1 {
      if k == 1 {
        self.p.as_ref_mut(ctx).copy(&self.r.as_ref(ctx));
      } else {
        self.p.as_ref_mut(ctx)
          .vector_scale(
              self.rhos_h[k-1] * self.rhos_h[k-1] / (self.rhos_h[k-2] * self.rhos_h[k-2])
          )
          .vector_add(&self.r.as_ref(ctx));
      }
      operator.read_direction(0, &mut self.p);
      self.config.iteration.step(batch_size, operator);
      operator.write_grad(0, &mut self.w);
      self.alphas.as_ref_mut(ctx).vector_prod(1.0, &self.p.as_ref(ctx), &self.w.as_ref(ctx));
      self.alphas.as_ref_mut(ctx).vector_inv(self.rhos_h[k-1] * self.rhos_h[k-1]);
      self.alphas.as_ref_range(k, k+1, ctx).sync_store(&self.alphas_h[k .. k+1]);
      self.x.as_ref_mut(ctx).vector_add(self.alphas_h[k], &self.p.as_ref(ctx));
      self.r.as_ref_mut(ctx).vector_add(-self.alphas_h[k], &self.w.as_ref(ctx));
      self.rhos.as_ref_mut_range(k, k+1, ctx).vector_l2_norm(&self.r.as_ref(ctx));
      self.rhos.as_ref_range(k, k+1, ctx).sync_store(&mut self.rhos_h[k .. k+1]);
    }
  }
}
