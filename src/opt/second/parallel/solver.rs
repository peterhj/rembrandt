use operator::{OpRead, OpWrite, OpCursor};
use opt::second::parallel::{ParallelSecondOptWorker};

use array_cuda::device::{
  DeviceContext, DeviceBufferInitExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut,
};
use array_cuda::device::linalg::{VectorExt};
use array_cuda::device::num::{NumExt};
use array_cuda::device::random::{RandomSampleExt, GaussianDist};
use comm_cuda::{RingDeviceBufCommBuilder, RingDeviceBufComm};

use std::iter::{repeat};
use std::rc::{Rc};

pub trait ParallelSolver {
  fn solve(&mut self, batch_size: usize, worker: &mut ParallelSecondOptWorker);
}

pub trait SolverIteration {
  fn step(&self, batch_size: usize, worker: &mut ParallelSecondOptWorker);
}

pub struct FisherIteration {
}

impl FisherIteration {
  pub fn new() -> FisherIteration {
    FisherIteration{}
  }
}

impl SolverIteration for FisherIteration {
  fn step(&self, batch_size: usize, worker: &mut ParallelSecondOptWorker) {
    worker.operator().reset_grad();
    worker.operator().r_forward(batch_size);
    worker.operator().set_targets_with_r_loss(batch_size);
    worker.operator().backward(batch_size);
    worker.operator().reset_targets(batch_size);
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

pub struct CgSolverConfig {
  pub max_iters:    usize,
  pub epsilon:      f32,
  pub lambda:       f32,
}

pub struct CgDeviceParallelSolver<Iter> {
  config:   CgSolverConfig,
  //max_iters:    usize,
  iteration:    Iter,
  ctx:      Rc<DeviceContext>,
  //comm:     RingDeviceBufComm<f32>,
  //epsilon:  f32,
  //lambda:   f32,
  g:        OpCursor<DeviceBuffer<f32>>,
  gnorm:    DeviceBuffer<f32>,
  gnorm_h:  Vec<f32>,
  p:        OpCursor<DeviceBuffer<f32>>,
  w:        OpCursor<DeviceBuffer<f32>>,
  r:        OpCursor<DeviceBuffer<f32>>,
  x:        OpCursor<DeviceBuffer<f32>>,
  xnorm:    DeviceBuffer<f32>,
  xnorm_h:  Vec<f32>,
  pw:       DeviceBuffer<f32>,
  pw_h:     Vec<f32>,
  rhos:     DeviceBuffer<f32>,
  rhos_h:   Vec<f32>,
  alphas_h: Vec<f32>,
}

impl<Iter> CgDeviceParallelSolver<Iter> where Iter: SolverIteration {
  pub fn new(param_len: usize, config: CgSolverConfig, /*max_iters: usize,*/ iteration: Iter, context: Rc<DeviceContext>) -> CgDeviceParallelSolver<Iter> {
    let ctx = &context.set();
    let max_iters = config.max_iters;
    CgDeviceParallelSolver{
      //max_iters:    max_iters,
      //epsilon:  1.0e-2,
      //lambda:   0.5,
      config:       config,
      iteration:    iteration,
      ctx:      context.clone(),
      g:        OpCursor::new(DeviceBuffer::zeros(param_len, ctx)),
      gnorm:    DeviceBuffer::zeros(1, ctx),
      gnorm_h:  vec![0.0],
      p:        OpCursor::new(DeviceBuffer::zeros(param_len, ctx)),
      w:        OpCursor::new(DeviceBuffer::zeros(param_len, ctx)),
      r:        OpCursor::new(DeviceBuffer::zeros(param_len, ctx)),
      x:        OpCursor::new(DeviceBuffer::zeros(param_len, ctx)),
      xnorm:    DeviceBuffer::zeros(1, ctx),
      xnorm_h:  vec![0.0],
      pw:       DeviceBuffer::zeros(max_iters+1, ctx),
      pw_h:     repeat(0.0).take(max_iters+1).collect(),
      rhos:     DeviceBuffer::zeros(max_iters+1, ctx),
      rhos_h:   repeat(0.0).take(max_iters+1).collect(),
      alphas_h: repeat(0.0).take(max_iters+1).collect(),
    }
  }
}

impl<Iter> ParallelSolver for CgDeviceParallelSolver<Iter> where Iter: SolverIteration {
  fn solve(&mut self, batch_size: usize, worker: &mut ParallelSecondOptWorker) {
    {
      let ctx = &self.ctx.set();
      self.g.as_ref_mut(ctx).set_constant(0.0);
      self.p.as_ref_mut(ctx).set_constant(0.0);
      self.w.as_ref_mut(ctx).set_constant(0.0);
      self.r.as_ref_mut(ctx).set_constant(0.0);
      self.x.as_ref_mut(ctx).set_constant(0.0);
      worker.read_step(&mut self.x.as_ref_mut(ctx));
      self.xnorm.as_ref_mut_range(0, 1, ctx).vector_l2_norm(&self.x.as_ref(ctx));
      self.xnorm.as_ref_range(0, 1, ctx).sync_store(&mut self.xnorm_h[0 .. 1]);
      self.pw.as_ref_mut(ctx).set_constant(0.0);
      self.rhos.as_ref_mut(ctx).set_constant(0.0);
    }
    let x0_norm = self.xnorm_h[0];
    worker.operator().write_grad(0, &mut self.g);
    {
      let ctx = &self.ctx.set();
      self.gnorm.as_ref_mut_range(0, 1, ctx).vector_l2_norm(&self.g.as_ref(ctx));
      self.gnorm.as_ref_range(0, 1, ctx).sync_store(&mut self.gnorm_h[0 .. 1]);
    }
    worker.operator().read_direction(0, &mut self.x);
    self.iteration.step(batch_size, worker);
    worker.operator().write_grad(0, &mut self.r);
    {
      let ctx = &self.ctx.set();
      //self.r.as_ref_mut(ctx).vector_add(-1.0, &self.g.as_ref(ctx), 0.0);
      if self.config.lambda > 0.0 {
        self.r.as_ref_mut(ctx).vector_add(self.config.lambda, &self.x.as_ref(ctx), 1.0);
        //self.r.as_ref_mut(ctx).vector_add(self.lambda, &self.x.as_ref(ctx), 1.0 - self.lambda);
      }
      self.r.as_ref_mut(ctx).vector_add(-1.0, &self.g.as_ref(ctx), -1.0);
      self.rhos.as_ref_mut_range(0, 1, ctx).vector_l2_norm(&self.r.as_ref(ctx));
      self.rhos.as_ref_range(0, 1, ctx).sync_store(&mut self.rhos_h[0 .. 1]);
    }
    let mut last_k = 0;
    for k in 1 .. self.config.max_iters + 1 {
      if self.rhos_h[k-1] <= self.config.epsilon * self.gnorm_h[0] {
        break;
      }
      last_k = k;
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
      let ctx = &(*self.ctx).as_ref();
      if self.config.lambda > 0.0 {
        self.w.as_ref_mut(ctx).vector_add(self.config.lambda, &self.p.as_ref(ctx), 1.0);
        //self.w.as_ref_mut(ctx).vector_add(self.lambda, &self.p.as_ref(ctx), 1.0 - self.lambda);
      }
      self.pw.as_ref_mut_range(k, k+1, ctx).vector_inner_prod(&self.p.as_ref(ctx), &self.w.as_ref(ctx));
      self.pw.as_ref_range(k, k+1, ctx).sync_store(&mut self.pw_h[k .. k+1]);
      self.alphas_h[k] = self.rhos_h[k-1] * self.rhos_h[k-1] / self.pw_h[k];
      self.x.as_ref_mut(ctx).vector_add(self.alphas_h[k], &self.p.as_ref(ctx), 1.0);
      self.r.as_ref_mut(ctx).vector_add(-self.alphas_h[k], &self.w.as_ref(ctx), 1.0);
      self.rhos.as_ref_mut_range(k, k+1, ctx).vector_l2_norm(&self.r.as_ref(ctx));
      self.rhos.as_ref_range(k, k+1, ctx).sync_store(&mut self.rhos_h[k .. k+1]);
      //println!("DEBUG: cg solver: iter: {} alpha[k]: {} rho[k]: {}", k, self.alphas_h[k], self.rhos_h[k]);
    }
    {
      let ctx = &self.ctx.set();
      self.xnorm.as_ref_mut_range(0, 1, ctx).vector_l2_norm(&self.x.as_ref(ctx));
      self.xnorm.as_ref_range(0, 1, ctx).sync_store(&mut self.xnorm_h[0 .. 1]);
    }
    println!("DEBUG: cg solver: K: {} |g|: {} |r[0]|: {} |r[K]|: {} |x[0]|: {} |x|: {}", last_k, self.gnorm_h[0], self.rhos_h[0], self.rhos_h[last_k], x0_norm, self.xnorm_h[0]);
    worker.operator().read_grad(0, &mut self.x);
  }
}
