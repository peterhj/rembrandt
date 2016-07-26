use operator::{CompleteOperator, OpRead, OpWrite, OpCursor, OpCursorInner, ExtentMap};
use opt::sgd::parallel::{ParallelSgdOptWorker};
use opt::second::parallel::{ParallelSecondOptWorker};

use array_cuda::device::context::{DeviceContext};
use array_cuda::device::linalg::{VectorExt, AsyncVectorExt};
use array_cuda::device::memory::{DeviceBufferInitExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut, RawDeviceBuffer};
use comm_cuda::{RingDeviceBufCommBuilder, RingDeviceBufComm};
use fixarith::{Fix24x64};
use mpi::{MpiCtx, MpiThreadLevel, MpiComm, MpiSumOp};
use multicore::{SpinBarrier};

use rand::{Rng, thread_rng};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering, fence};

/*#[derive(Clone)]
pub struct MpiAllreduceOptWorkerBuilder {
  num_workers:  usize,
  shared:       DevWorkerSharedData,
  comm_builder: RingDeviceBufCommBuilder<f32>,
}

impl MpiAllreduceOptWorkerBuilder {
  pub fn new(num_workers: usize) -> MpiAllreduceOptWorkerBuilder {
    let shared = DevWorkerSharedData{
      shared_seed:  [thread_rng().next_u64(), thread_rng().next_u64()],
      reduce_count: Arc::new(AtomicUsize::new(0)),
    };
    MpiAllreduceOptWorkerBuilder{
      num_workers:  num_workers,
      shared:       shared,
      comm_builder: RingDeviceBufCommBuilder::new(num_workers),
    }
  }

  pub fn into_worker(self, worker_rank: usize, context: Rc<DeviceContext>, operator: Box<CompleteOperator>) -> MpiAllreduceOptWorker {
    let params_len = operator.params_len();
    match self.comm_builder.buf_len() {
      Some(buf_len) => assert_eq!(params_len, buf_len),
      None => self.comm_builder.set_buf_len(params_len),
    }
    let comm = OpCursor::new(self.comm_builder.into_comm(worker_rank, context.clone()));
    let pad = self.num_workers * 32;
    let padded_len = (params_len + pad - 1) / pad * pad;
    let ctx = &(*context).as_ref();
    MpiAllreduceOptWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      context:      context.clone(),
      shared:       self.shared,
      operator:     operator,
      comm:         comm,
      grad_acc:     OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      saved_param:  OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      w_norm:       DeviceBuffer::zeros(1, ctx),
      w_norm_h:     vec![0.0],
      g_norm:       DeviceBuffer::zeros(1, ctx),
      g_norm_h:     vec![0.0],
      sig_chkpt:    false,
    }
  }
}*/

pub struct MpiAllreduceOptWorker {
  mpi_ctx:      MpiCtx,
  mpi_comm:     MpiComm,
  worker_rank:  usize,
  num_workers:  usize,
  context:      Rc<DeviceContext>,
  //shared:       DevWorkerSharedData,
  operator:     Box<CompleteOperator>,
  //comm:         OpCursor<RingDeviceBufComm<f32>>,
  grad_acc:     OpCursor<DeviceBuffer<f32>>,
  grad_buf:     OpCursor<DeviceBuffer<f32>>,
  grad_buf_src: Vec<f32>,
  grad_buf_dst: Vec<f32>,
  saved_param:  OpCursor<DeviceBuffer<f32>>,
  w_norm:       DeviceBuffer<f32>,
  w_norm_h:     Vec<f32>,
  g_norm:       DeviceBuffer<f32>,
  g_norm_h:     Vec<f32>,
  sig_chkpt:    bool,
}

impl MpiAllreduceOptWorker {
  pub fn new(context: Rc<DeviceContext>, operator: Box<CompleteOperator>) -> MpiAllreduceOptWorker {
    let params_len = operator.params_len();
    /*match self.comm_builder.buf_len() {
      Some(buf_len) => assert_eq!(params_len, buf_len),
      None => self.comm_builder.set_buf_len(params_len),
    }*/
    //let comm = OpCursor::new(self.comm_builder.into_comm(worker_rank, context.clone()));
    let mpi_ctx = MpiCtx::new(MpiThreadLevel::Serialized);
    let mpi_comm = MpiComm::world(&mpi_ctx);
    let worker_rank = mpi_comm.rank().unwrap();
    let num_workers = mpi_comm.size().unwrap();
    let pad = num_workers * 32;
    let padded_len = (params_len + pad - 1) / pad * pad;
    let ctx = &(*context).as_ref();
    let mut grad_buf_src = Vec::with_capacity(padded_len);
    unsafe { grad_buf_src.set_len(padded_len) };
    let mut grad_buf_dst = Vec::with_capacity(padded_len);
    unsafe { grad_buf_dst.set_len(padded_len) };
    MpiAllreduceOptWorker{
      mpi_ctx:      mpi_ctx,
      mpi_comm:     mpi_comm,
      worker_rank:  worker_rank,
      num_workers:  num_workers,
      context:      context.clone(),
      //shared:       self.shared,
      operator:     operator,
      //comm:         comm,
      grad_acc:     OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      grad_buf:     OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      grad_buf_src: grad_buf_src,
      grad_buf_dst: grad_buf_dst,
      saved_param:  OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      w_norm:       DeviceBuffer::zeros(1, ctx),
      w_norm_h:     vec![0.0],
      g_norm:       DeviceBuffer::zeros(1, ctx),
      g_norm_h:     vec![0.0],
      sig_chkpt:    false,
    }
  }
}

impl ParallelSgdOptWorker for MpiAllreduceOptWorker {
  fn worker_rank(&self) -> usize {
    self.worker_rank
  }

  fn num_workers(&self) -> usize {
    self.num_workers
  }

  fn operator(&mut self) -> &mut CompleteOperator {
    &mut *self.operator
  }

  fn shared_seed(&mut self) -> [u64; 2] {
    //self.shared.shared_seed
    //unimplemented!();
    [12345678, 12345678]
  }

  fn reduce_scalar(&self, count: usize) -> usize {
    /*if self.worker_rank == 0 {
      self.shared.reduce_count.store(0, Ordering::Release);
    }
    self.comm.inner.barrier();
    fence(Ordering::AcqRel);
    self.shared.reduce_count.fetch_add(count, Ordering::AcqRel);
    self.comm.inner.barrier();
    fence(Ordering::AcqRel);
    let x = self.shared.reduce_count.load(Ordering::Acquire);
    self.comm.inner.barrier();
    fence(Ordering::AcqRel);
    x*/
    //unimplemented!();
    0
  }

  fn reduce_scalar_f32(&self, count: f32) -> f32 {
    /*if self.worker_rank == 0 {
      self.shared.reduce_count.store(0, Ordering::Release);
    }
    self.comm.inner.barrier();
    fence(Ordering::AcqRel);
    let count_fix = Fix24x64::round_nearest_f64(count as f64);
    let r = count_fix.into_usize_repr();
    self.shared.reduce_count.fetch_add(r, Ordering::AcqRel);
    self.comm.inner.barrier();
    fence(Ordering::AcqRel);
    let x = self.shared.reduce_count.load(Ordering::Acquire);
    self.comm.inner.barrier();
    fence(Ordering::AcqRel);
    let sum = Fix24x64::from_usize_repr(x);
    sum.into_f64() as f32*/
    //unimplemented!();
    0.0
  }

  fn signal_checkpoint(&mut self) {
    self.sig_chkpt = true;
  }

  fn wait_checkpoint(&mut self) -> bool {
    if self.sig_chkpt {
      //self.comm.inner.barrier();
      self.mpi_comm.barrier().unwrap();
      self.sig_chkpt = false;
      true
    } else {
      false
    }
  }

  fn sync_loss(&mut self, batch_size: usize) -> f32 {
    /*let local_loss = self.operator().store_loss(batch_size);
    let total_loss = self.reduce_scalar_f32(local_loss) / self.num_workers() as f32;
    total_loss*/
    //unimplemented!();
    0.0
  }

  fn sync_param(&mut self) {
    unimplemented!();
  }

  fn save_param(&mut self) {
    self.operator.write_param(0, &mut self.saved_param);
    /*let ctx = &self.context.set();
    self.w_norm.as_ref_mut(ctx).vector_l2_norm(&self.saved_param.as_ref(ctx));
    self.w_norm.as_ref(ctx).sync_store(&mut self.w_norm_h);
    println!("DEBUG: worker: rank: {} |param|: {:.6}", self.worker_rank(), self.w_norm_h[0]);*/
  }

  fn restore_param(&mut self) {
    self.operator.read_param(0, &mut self.saved_param);
  }

  fn sync_grad(&mut self) {
    self.operator.write_grad(0, &mut self.grad_buf);
    {
      let ctx = &self.context.set();
      self.grad_buf.as_ref_mut(ctx).vector_scale(1.0 / (self.num_workers as f32));
      self.grad_buf.as_ref(ctx).sync_store(&mut self.grad_buf_src);
    }
    self.mpi_comm.nonblocking_allreduce(&self.grad_buf_src, &mut self.grad_buf_dst, MpiSumOp);
    {
      let ctx = &self.context.set();
      self.grad_buf.as_ref_mut(ctx).sync_load(&self.grad_buf_dst);
    }
    self.operator.read_grad(0, &mut self.grad_buf);
  }

  fn accumulate_grad(&mut self, alpha: f32, mu: f32) {
    self.operator.accumulate_grad_(0, alpha, mu, &mut self.grad_acc);
    /*let ctx = &self.context.set();
    self.g_norm.as_ref_mut(ctx).vector_l2_norm(&self.grad_acc.as_ref(ctx));
    self.g_norm.as_ref(ctx).sync_store(&mut self.g_norm_h);
    println!("DEBUG: worker: rank: {} |grad|: {:.6}", self.worker_rank(), self.g_norm_h[0]);*/
  }

  fn step(&mut self, step_size: f32) {
    self.operator.step(0, step_size, &mut self.grad_acc);
  }

  fn block(&mut self) {
    //unimplemented!();
  }
}
