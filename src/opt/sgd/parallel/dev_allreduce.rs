use operator::{CompleteOperator, OpRead, OpWrite, OpCursor, OpCursorInner, ExtentMap};
use opt::sgd::parallel::{ParallelSgdOptWorker};

//use array::{Shape};
use array_cuda::device::context::{DeviceContext};
use array_cuda::device::linalg::{VectorExt};
use array_cuda::device::memory::{DeviceBufferInitExt, DeviceBuffer};
use comm_cuda::{RingDeviceBufCommBuilder, RingDeviceBufComm};
use nccl::{NcclUniqueId, NcclComm, NcclSumOp};

use rand::{Rng, thread_rng};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::sync::{Arc, Barrier};

struct DevWorkerSharedData {
  shared_seed:  [u64; 2],
}

#[derive(Clone)]
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
}

#[derive(Clone)]
pub struct DeviceAllreduceSgdOptWorkerBuilder {
  num_workers:  usize,
  comm_id:  NcclUniqueId,
  barrier:  Arc<Barrier>,
  shared:   Arc<DevWorkerSharedData>,
  comm_builder: RingDeviceBufCommBuilder<f32>,
}

impl DeviceAllreduceSgdOptWorkerBuilder {
  pub fn new(num_workers: usize) -> DeviceAllreduceSgdOptWorkerBuilder {
    let shared = DevWorkerSharedData{
      shared_seed:  [thread_rng().next_u64(), thread_rng().next_u64()],
    };
    DeviceAllreduceSgdOptWorkerBuilder{
      num_workers:  num_workers,
      comm_id:  NcclUniqueId::create().unwrap(),
      barrier:  Arc::new(Barrier::new(num_workers)),
      shared:   Arc::new(shared),
      comm_builder: RingDeviceBufCommBuilder::new(num_workers),
    }
  }

  pub fn into_worker(self, worker_rank: usize, context: Rc<DeviceContext>, operator: Box<CompleteOperator>) -> DeviceAllreduceSgdOptWorker {
    /*let comm = match NcclComm::create(worker_rank, self.num_workers, self.comm_id) {
      Err(e) => panic!("failed to create nccl comm: {:?}", e),
      Ok(comm) => comm,
    };*/
    let params_len = operator.params_len();
    match self.comm_builder.buf_len() {
      Some(buf_len) => assert_eq!(params_len, buf_len),
      None => self.comm_builder.set_buf_len(params_len),
    }
    let comm = self.comm_builder.into_comm(worker_rank, context.clone());
    let pad = self.num_workers * 32;
    let padded_len = (params_len + pad - 1) / pad * pad;
    let ctx = &(*context).as_ref();
    DeviceAllreduceSgdOptWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      context:      context.clone(),
      //comm:         comm,
      barrier:      self.barrier,
      shared:       self.shared,
      comm:         comm,
      operator:     operator,
      saved_param:  OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      //src_grad:     OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      //dst_grad:     OpCursor::new(DeviceBuffer::zeros(padded_len, ctx)),
      sig_chkpt:    false,
    }
  }
}

impl OpCursorInner for RingDeviceBufComm<f32> {
  type Extra = ExtentMap;

  fn extra(&self) -> ExtentMap {
    //let part_lens: Vec<_> = self.iter().map(|buf| buf.len()).collect();
    let part_lens = vec![];
    ExtentMap::new(part_lens)
  }
}

pub struct DeviceAllreduceSgdOptWorker {
  worker_rank:  usize,
  num_workers:  usize,
  context:      Rc<DeviceContext>,
  //comm:         NcclComm,
  barrier:      Arc<Barrier>,
  shared:       Arc<DevWorkerSharedData>,
  comm:         RingDeviceBufComm<f32>,
  //comm:         OpCursor<RingDeviceBufComm<f32>>,

  operator:     Box<CompleteOperator>,

  saved_param:  OpCursor<DeviceBuffer<f32>>,
  //src_grad:     OpCursor<DeviceBuffer<f32>>,
  //dst_grad:     OpCursor<DeviceBuffer<f32>>,

  sig_chkpt:    bool,
}

impl ParallelSgdOptWorker for DeviceAllreduceSgdOptWorker {
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
    // FIXME(20160607)
    //self.operator.write_grad(0, &mut self.src_grad);
  }

  fn merge_grad(&mut self) {
    // FIXME(20160607)
    //self.operator.read_grad(0, &mut self.dst_grad);
  }

  fn sync_grad(&mut self) {
    let ctx = &(*self.context).as_ref();
    // FIXME(20160607)
    /*//ctx.sync();
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
    self.dst_grad.inner.as_ref_mut(ctx).vector_scale(1.0 / self.num_workers as f32);*/
  }
}
