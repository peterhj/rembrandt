use operator::{CompleteOperator, OpRead, OpWrite, OpCursor, OpCursorInner, ExtentMap};
use opt::sgd::parallel::{ParallelSgdOptWorker};
use opt::second::parallel::{ParallelSecondOptWorker};

//use array::{Shape};
use array_cuda::device::context::{DeviceContext};
use array_cuda::device::linalg::{VectorExt, AsyncVectorExt};
use array_cuda::device::memory::{DeviceBufferInitExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut, RawDeviceBuffer};
use comm_cuda::{RingDeviceBufCommBuilder, RingDeviceBufComm};
//use nccl::{NcclUniqueId, NcclComm, NcclSumOp};

use rand::{Rng, thread_rng};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering, fence};

#[derive(Clone)]
struct DevWorkerSharedData {
  shared_seed:  [u64; 2],
  //reduce_buf:   Mutex<f32>,
  reduce_count: Arc<AtomicUsize>,
}

/*#[derive(Clone)]
pub struct DeviceNcclAllreduceSgdOptWorkerBuilder {
  num_workers:  usize,
  comm_id:  NcclUniqueId,
  barrier:  Arc<Barrier>,
  shared:   Arc<DevWorkerSharedData>,
}

impl DeviceNcclAllreduceSgdOptWorkerBuilder {
  pub fn new(num_workers: usize) -> DeviceNcclAllreduceSgdOptWorkerBuilder {
    let shared = DevWorkerSharedData{
      shared_seed:  [thread_rng().next_u64(), thread_rng().next_u64()],
    };
    DeviceNcclAllreduceSgdOptWorkerBuilder{
      num_workers:  num_workers,
      comm_id:  NcclUniqueId::create().unwrap(),
      barrier:  Arc::new(Barrier::new(num_workers)),
      shared:   Arc::new(shared),
    }
  }

  pub fn into_worker(self, worker_rank: usize, context: Rc<DeviceContext>, operator: Box<CompleteOperator>) -> DeviceNcclAllreduceSgdOptWorker {
    let comm = match NcclComm::create(worker_rank, self.num_workers, self.comm_id) {
      Err(e) => panic!("failed to create nccl comm: {:?}", e),
      Ok(comm) => comm,
    };
    let params_len = operator.params_len();
    let pad = self.num_workers * 32;
    let padded_len = (params_len + pad - 1) / pad * pad;
    let ctx = &(*context).as_ref();
    DeviceNcclAllreduceSgdOptWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      context:      context.clone(),
      comm:         comm,
      barrier:      self.barrier,
      shared:       self.shared,
      operator:     operator,
      saved_param:  OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      src_grad:     OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      dst_grad:     OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      sig_chkpt:    false,
    }
  }
}

pub struct DeviceNcclAllreduceSgdOptWorker {
  worker_rank:  usize,
  num_workers:  usize,
  context:      Rc<DeviceContext>,
  comm:         NcclComm,
  barrier:      Arc<Barrier>,
  shared:       Arc<DevWorkerSharedData>,

  operator:     Box<CompleteOperator>,

  saved_param:  OpCursor<DeviceBuffer<f32>>,
  src_grad:     OpCursor<DeviceBuffer<f32>>,
  dst_grad:     OpCursor<DeviceBuffer<f32>>,

  sig_chkpt:    bool,
}

impl ParallelSgdOptWorker for DeviceNcclAllreduceSgdOptWorker {
  fn operator(&mut self) -> &mut CompleteOperator {
    &mut *self.operator
  }

  fn shared_seed(&mut self) -> [u64; 2] {
    self.shared.shared_seed
  }

  fn signal_checkpoint(&mut self) {
    self.sig_chkpt = true;
  }

  fn wait_checkpoint(&mut self) -> bool {
    if self.sig_chkpt {
      self.barrier.wait();
      self.sig_chkpt = false;
      true
    } else {
      false
    }
  }

  fn save_param(&mut self) {
    self.operator.write_param(0, &mut self.saved_param);
  }

  fn restore_param(&mut self) {
    self.operator.read_param(0, &mut self.saved_param);
  }

  fn stage_param(&mut self) {
    unimplemented!();
  }

  fn merge_param(&mut self) {
    unimplemented!();
  }

  fn sync_param(&mut self) {
    unimplemented!();
  }

  fn stage_grad(&mut self) {
    self.operator.write_grad(0, &mut self.src_grad);
  }

  fn merge_grad(&mut self) {
    self.operator.read_grad(0, &mut self.dst_grad);
  }

  fn sync_grad(&mut self) {
    let ctx = &(*self.context).as_ref();
    //ctx.sync();
    //self.barrier.wait();
    let params_len = self.src_grad.inner.len();
    unsafe { self.comm.allreduce(
        self.src_grad.inner.as_ref(ctx).as_ptr(), params_len,
        self.dst_grad.inner.as_ref_mut(ctx).as_mut_ptr(),
        NcclSumOp,
        ctx.stream.ptr,
    ).unwrap() };
    //ctx.sync();
    //self.barrier.wait();
    // FIXME(20160526): assuming all worker batches are evenly sized.
    self.dst_grad.inner.as_ref_mut(ctx).vector_scale(1.0 / self.num_workers as f32);
  }
}*/

impl OpCursorInner for RingDeviceBufComm<f32> {
  type Extra = ExtentMap;

  fn extra(&self) -> ExtentMap {
    let part_lens: Vec<_> = self.bufs[self.worker_rank()].iter().map(|buf| buf.len()).collect();
    ExtentMap::new(part_lens)
  }
}

impl OpRead for OpCursor<RingDeviceBufComm<f32>> {
  fn read<'a>(&'a mut self, offset: usize, dst: &mut DeviceBufferRefMut<'a, f32>) -> usize {
    let buf_len = dst.len();
    let &mut OpCursor{ref extra, ref inner} = self;
    extra.foreach_extent(offset, offset + buf_len, |part_idx, (part_lower, part_upper), (rel_lower, rel_upper)| {
      dst.mut_range(rel_lower, rel_upper)
        .copy_raw(&inner.bufs[inner.worker_rank()][part_idx].as_ref_range(part_lower, part_upper));
    });
    buf_len
  }

  fn accumulate_read<'a>(&'a mut self, offset: usize, alpha: f32, beta: f32, dst: &mut DeviceBufferRefMut<'a, f32>) -> usize {
    let buf_len = dst.len();
    let &mut OpCursor{ref extra, ref inner} = self;
    extra.foreach_extent(offset, offset + buf_len, |part_idx, (part_lower, part_upper), (rel_lower, rel_upper)| {
      let src = inner.bufs[inner.worker_rank()][part_idx].as_ref_range(part_lower, part_upper);
      dst.mut_range(rel_lower, rel_upper).vector_scale(beta);
      dst.mut_range(rel_lower, rel_upper).vector_add_raw(alpha, &src);
    });
    buf_len
  }
}

impl OpWrite for OpCursor<RingDeviceBufComm<f32>> {
  fn write<'a>(&'a mut self, offset: usize, src: &DeviceBufferRef<'a, f32>) -> usize {
    let buf_len = src.len();
    let &mut OpCursor{ref extra, ref inner} = self;
    extra.foreach_extent(offset, offset + buf_len, |part_idx, (part_lower, part_upper), (rel_lower, rel_upper)| {
      inner.bufs[inner.worker_rank()][part_idx].as_ref_range(part_lower, part_upper)
        .copy(&src.range(rel_lower, rel_upper));
    });
    buf_len
  }

  fn accumulate_write<'a>(&'a mut self, offset: usize, alpha: f32, beta: f32, src: &DeviceBufferRef<'a, f32>) -> usize {
    let buf_len = src.len();
    let &mut OpCursor{ref extra, ref inner} = self;
    extra.foreach_extent(offset, offset + buf_len, |part_idx, (part_lower, part_upper), (rel_lower, rel_upper)| {
      let mut dst = inner.bufs[inner.worker_rank()][part_idx].as_ref_range(part_lower, part_upper);
      dst.async_vector_scale(beta, &src.ctx);
      dst.async_vector_add(alpha, &src.range(rel_lower, rel_upper));
    });
    buf_len
  }
}

#[derive(Clone)]
pub struct DeviceAllreduceOptWorkerBuilder {
  num_workers:  usize,
  shared:       DevWorkerSharedData,
  comm_builder: RingDeviceBufCommBuilder<f32>,
}

impl DeviceAllreduceOptWorkerBuilder {
  pub fn new(num_workers: usize) -> DeviceAllreduceOptWorkerBuilder {
    let shared = DevWorkerSharedData{
      shared_seed:  [thread_rng().next_u64(), thread_rng().next_u64()],
      //reduce_buf:   Mutex::new(0.0),
      reduce_count: Arc::new(AtomicUsize::new(0)),
    };
    DeviceAllreduceOptWorkerBuilder{
      num_workers:  num_workers,
      shared:       shared,
      comm_builder: RingDeviceBufCommBuilder::new(num_workers),
    }
  }

  pub fn into_worker(self, worker_rank: usize, context: Rc<DeviceContext>, operator: Box<CompleteOperator>) -> DeviceAllreduceOptWorker {
    /*let comm = match NcclComm::create(worker_rank, self.num_workers, self.comm_id) {
      Err(e) => panic!("failed to create nccl comm: {:?}", e),
      Ok(comm) => comm,
    };*/
    let params_len = operator.params_len();
    match self.comm_builder.buf_len() {
      Some(buf_len) => assert_eq!(params_len, buf_len),
      None => self.comm_builder.set_buf_len(params_len),
    }
    let comm = OpCursor::new(self.comm_builder.into_comm(worker_rank, context.clone()));
    let pad = self.num_workers * 32;
    let padded_len = (params_len + pad - 1) / pad * pad;
    let ctx = &(*context).as_ref();
    DeviceAllreduceOptWorker{
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
}

pub struct DeviceAllreduceOptWorker {
  worker_rank:  usize,
  num_workers:  usize,
  context:      Rc<DeviceContext>,
  shared:       DevWorkerSharedData,
  operator:     Box<CompleteOperator>,
  comm:         OpCursor<RingDeviceBufComm<f32>>,
  grad_acc:     OpCursor<DeviceBuffer<f32>>,
  saved_param:  OpCursor<DeviceBuffer<f32>>,
  w_norm:       DeviceBuffer<f32>,
  w_norm_h:     Vec<f32>,
  g_norm:       DeviceBuffer<f32>,
  g_norm_h:     Vec<f32>,
  sig_chkpt:    bool,
}

impl ParallelSgdOptWorker for DeviceAllreduceOptWorker {
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
    self.shared.shared_seed
  }

  fn reduce_scalar(&self, count: usize) -> usize {
    if self.worker_rank == 0 {
      self.shared.reduce_count.store(0, Ordering::Release);
    }
    self.comm.inner.barrier();
    fence(Ordering::AcqRel);
    self.shared.reduce_count.fetch_add(count, Ordering::AcqRel);
    self.comm.inner.barrier();
    fence(Ordering::AcqRel);
    let x = self.shared.reduce_count.load(Ordering::Acquire);
    fence(Ordering::AcqRel);
    x
  }

  fn signal_checkpoint(&mut self) {
    self.sig_chkpt = true;
  }

  fn wait_checkpoint(&mut self) -> bool {
    if self.sig_chkpt {
      self.comm.inner.barrier();
      self.sig_chkpt = false;
      true
    } else {
      false
    }
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

  fn stage_param(&mut self) {
    unimplemented!();
  }

  fn sync_param(&mut self) {
    unimplemented!();
  }

  fn merge_param(&mut self) {
    unimplemented!();
  }

  fn stage_grad(&mut self) {
    //self.operator.write_grad(0, &mut self.comm);
    unimplemented!();
  }

  fn sync_grad(&mut self) {
    self.operator.write_grad(0, &mut self.comm);
    self.comm.inner.barrier();
    self.comm.inner.allreduce_average();
    self.operator.read_grad(0, &mut self.comm);
    self.comm.inner.barrier();
  }

  fn merge_grad(&mut self) {
    //self.operator.read_grad(0, &mut self.comm);
    unimplemented!();
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
}

impl ParallelSecondOptWorker for DeviceAllreduceOptWorker {
  //fn stage<'a>(&'a mut self, src: &'a DeviceBufferRef<'a, f32>) {
  fn stage<'ctx>(&'ctx mut self, src: &DeviceBufferRef<'ctx, f32>) {
    self.comm.write(0, src);
  }

  fn sync(&mut self) {
    self.comm.inner.barrier();
    self.comm.inner.allreduce_average();
  }

  //fn merge<'a>(&'a mut self, dst: &'a mut DeviceBufferRefMut<'a, f32>) {
  fn merge<'ctx>(&'ctx mut self, dst: &mut DeviceBufferRefMut<'ctx, f32>) {
    self.comm.read(0, dst);
  }

  fn read_step<'ctx>(&'ctx mut self, dst: &mut DeviceBufferRefMut<'ctx, f32>) {
    self.grad_acc.read(0, dst);
  }
}
