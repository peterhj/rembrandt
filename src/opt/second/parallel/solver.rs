use operator::{OpRead, OpWrite, OpCursor};
use opt::second::parallel::{ParallelSecondOptWorker};

use array_cuda::device::{
  DeviceContext, DeviceBufferInitExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut,
};
use array_cuda::device::linalg::{VectorExt};
use array_cuda::device::num::{NumExt};
use array_cuda::device::random::{RandomSampleExt, GaussianDist};
use comm_cuda::{RingDeviceBufCommBuilder, RingDeviceBufComm};

use std::rc::{Rc};

pub trait ParallelSolver {
}

pub trait SolverIteration {
  fn step(&self, batch_size: usize, worker: &mut ParallelSecondOptWorker);
}

pub struct FisherIteration {
}

impl SolverIteration for FisherIteration {
  fn step(&self, batch_size: usize, worker: &mut ParallelSecondOptWorker) {
    worker.operator().reset_grad();
    worker.operator().r_forward(batch_size);
    // FIXME(20160617): set loss targets.
    worker.operator().backward(batch_size);
    // FIXME(20160617): reset loss targets.
    worker.sync_grad();
  }
}

pub struct HessianIteration {
}

impl SolverIteration for HessianIteration {
  fn step(&self, batch_size: usize, worker: &mut ParallelSecondOptWorker) {
    worker.operator().reset_r_grad();
    worker.operator().r_forward(batch_size);
    // FIXME(20160617): set loss targets (?)
    worker.operator().r_backward(batch_size);
    // FIXME(20160617): reset loss targets (?).
    worker.sync_grad();
  }
}

/*pub struct LanczosDeviceParallelSolverBuilder {
}

pub struct LanczosDeviceParallelSolver {
  config:   ParallelSolveConfig<Iteration>,
  context:  Rc<DeviceContext>,
  comm:     RingDeviceBufComm<f32>,
  r:        OpCursor<DeviceBuffer<f32>>,
  v_curr:   OpCursor<DeviceBuffer<f32>>,
  v_prev:   DeviceBuffer<f32>,
  alphas:   DeviceBuffer<f32>,
  alphas_h: Vec<f32>,
  betas:    DeviceBuffer<f32>,
  betas_h:  Vec<f32>,
}

impl LanczosDeviceParallelSolver {
  pub fn solve(&mut self, batch_size: usize, worker: &mut ParallelSecondOptWorker) {
    let ctx = &(*self.context).as_ref();
    self.v_prev.as_ref_mut(ctx).set_constant(0.0);
    self.v_curr.as_ref_mut(ctx).sample(&GaussianDist{mean: 0.0, std: 1.0});
    self.betas.as_ref_mut_range(0, 1, ctx).vector_l2_norm(&self.v_curr.as_ref(ctx));
    self.betas.as_ref_mut_range(0, 1, ctx).sync_store(&mut self.betas_h[0 .. 1]);
    self.v_curr.as_ref_mut(ctx).vector_scale(1.0 / self.betas_h[0]);
    operator.reset();
    //self.config.iteration.init(batch_size, phase, operator);
    for k in 0 .. self.config.max_iters {
      worker.operator().read_direction(0, &mut self.v_curr);
      self.config.iteration.step(batch_size, worker);
      worker.sync_grad();
      worker.operator().write_grad(0, &mut self.r);
      if k > 0 {
        self.r.as_ref_mut(ctx).vector_add(-self.betas_h[k], &self.v_prev.as_ref(ctx));
      }
      self.alphas.as_ref_mut_range(k, k+1, ctx).vector_prod(&self.v_curr.as_ref(ctx), &self.r.as_ref(ctx));
      self.alphas.as_ref_mut_range(k, k+1, ctx).sync_store(&mut self.alphas_h[k .. k+1]);
      self.r.as_ref_mut(ctx).vector_add(-self.alphas_h[k], &self.v_curr.as_ref(ctx));
      if k < self.max_iters-1 {
        self.betas.as_ref_mut_range(k+1, k+2, ctx).vector_l2_norm(&self.r.as_ref(ctx));
        self.betas.as_ref_range(k+1, k+2, ctx).sync_store(&mut self.betas_h[k+1 .. k+2]);
        self.v_prev.as_ref_mut(ctx).copy(&self.v_curr.as_ref(ctx));
        self.v_curr.as_ref_mut(ctx).vector_set(1.0 / self.betas_h[k+1], &self.r.as_ref(ctx));
      }
    }
  }
}*/

pub struct CgDeviceParallelSolver<Iter> {
  max_iters:    usize,
  iteration:    Iter,
  ctx:      Rc<DeviceContext>,
  //comm:     RingDeviceBufComm<f32>,
  lambda:   f32,
  ng:       OpCursor<DeviceBuffer<f32>>,
  p:        OpCursor<DeviceBuffer<f32>>,
  w:        OpCursor<DeviceBuffer<f32>>,
  x:        DeviceBuffer<f32>,
  r:        DeviceBuffer<f32>,
  pw:       DeviceBuffer<f32>,
  pw_h:     Vec<f32>,
  rhos:     DeviceBuffer<f32>,
  rhos_h:   Vec<f32>,
  alphas_h: Vec<f32>,
}

impl<Iter> CgDeviceParallelSolver<Iter> where Iter: SolverIteration {
  pub fn solve(&mut self, batch_size: usize, worker: &mut ParallelSecondOptWorker) {
    worker.operator().write_grad(0, &mut self.ng);
    {
      let ctx = &self.ctx.set();
      self.x.as_ref_mut(ctx).set_constant(0.0);
      self.rhos.as_ref_mut_range(0, 1, ctx).vector_l2_norm(&self.r.as_ref(ctx));
      self.rhos.as_ref_range(0, 1, ctx).sync_store(&mut self.rhos_h[0 .. 1]);
    }
    for k in 1 .. self.max_iters + 1 {
      if k == 1 {
        let ctx = &(*self.ctx).as_ref();
        self.p.as_ref_mut(ctx).copy(&self.r.as_ref(ctx));
      } else {
        let ctx = &(*self.ctx).as_ref();
        self.p.as_ref_mut(ctx).vector_add(1.0, &self.r.as_ref(ctx), (self.rhos_h[k-1] * self.rhos_h[k-1]) / (self.rhos_h[k-2] * self.rhos_h[k-2]));
      }
      worker.operator().read_direction(0, &mut self.p);
      self.iteration.step(batch_size, worker);
      worker.operator().write_grad(0, &mut self.w);
      {
        let ctx = &self.ctx.set();
        worker.stage(&self.w.as_ref(ctx));
      }
      worker.sync();
      {
        let ctx = &self.ctx.set();
        worker.merge(&mut self.w.as_ref_mut(ctx));
      }
      let ctx = &(*self.ctx).as_ref();
      self.w.as_ref_mut(ctx).vector_add(self.lambda, &self.p.as_ref(ctx), 1.0 - self.lambda);
      self.pw.as_ref_mut_range(k, k+1, ctx).vector_inner_prod(&self.p.as_ref(ctx), &self.w.as_ref(ctx));
      self.pw.as_ref_range(k, k+1, ctx).sync_store(&mut self.pw_h[k .. k+1]);
      self.alphas_h[k] = self.rhos_h[k-1] * self.rhos_h[k-1] / self.pw_h[k];
      self.x.as_ref_mut(ctx).vector_add(self.alphas_h[k], &self.p.as_ref(ctx), 1.0);
      self.r.as_ref_mut(ctx).vector_add(-self.alphas_h[k], &self.w.as_ref(ctx), 1.0);
      self.rhos.as_ref_mut_range(k, k+1, ctx).vector_l2_norm(&self.r.as_ref(ctx));
      self.rhos.as_ref_range(k, k+1, ctx).sync_store(&mut self.rhos_h[k .. k+1]);
    }
  }
}
