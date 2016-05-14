use operator::{
  Operator, InputOperator,
  OperatorNode, OperatorConfig,
  OpCapability, OpPhase,
  Regularization,
  Data3dOperatorConfig,
  AffineOperatorConfig,
  Conv2dOperatorConfig,
  Pool2dOperatorConfig,
  DropoutOperatorConfig,
};
use operator::comm::{
  //CommWorkerBuilder,
  CommWorker,
  GossipConfig,
  ParameterServerConfig,
};
use operator::loss::{
  LossOperator,
  CategoricalLossConfig,
};
use operator::worker::{
  OperatorWorkerBuilder,
  OperatorWorker,
  SequentialOperatorConfig,
};
use worker::{MpiDistCommWorker};

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::comm::{ReduceOperation, AverageReduceOperation, for_all_devices};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::ext::{DeviceAsyncNumExt};
use array_cuda::device::linalg::{AsyncBlasVectorExt};
use array_cuda::device::memory::{RawDeviceBuffer};
use array_new::{AsyncArray};
use rng::xorshift::{Xorshiftplus128Rng};
use worker_::{WorkerData};

use mpi::{Mpi, MpiComm, MpiGroup, MpiWindow, MpiWindowLockMode, MpiRequest, MpiRequestList, MpiSumOp};
//use procgroup::{ProcGroup};
use threadpool::{ThreadPool};

use rand::{Rng, SeedableRng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{min};
use std::collections::{HashSet};
use std::ffi::{CString};
use std::io::{Write};
use std::iter::{FromIterator, repeat};
use std::marker::{PhantomData};
use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use std::sync::mpsc::{Sender, Receiver, TryRecvError, channel};
use std::thread::{JoinHandle, sleep, spawn};
use std::time::{Duration};
use vec_map::{VecMap};

/*#[derive(Clone)]
pub struct MpiDistSyncAllreduceCommWorkerBuilder {
}

impl MpiDistSyncAllreduceCommWorkerBuilder {
  pub fn into_worker(self, context: Rc<DeviceContext>)
}*/

pub struct MpiDistSyncAllreduceCommWorker {
  pub worker_data:  WorkerData,
  pub mpi:          Mpi,
  context:      Rc<DeviceContext>,

  buf_len:      usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  com_interval: usize,

  origin_buf:   RawDeviceBuffer<f32>,
  target_buf:   RawDeviceBuffer<f32>,
  origin_buf_h: Vec<f32>,
  target_buf_h: Vec<f32>,

  send_reqs:    MpiRequestList,

  avg_reduce:   AverageReduceOperation<f32>,

  shared_seed:  [u64; 2],
  shared_rng:   Xorshiftplus128Rng,
  ranks_perm:   Vec<usize>,
  iter_counter: usize,
  bar_signal: bool,
}

impl MpiDistSyncAllreduceCommWorker {
  pub fn new(paramserver_cfg: ParameterServerConfig, context: Rc<DeviceContext>) -> MpiDistSyncAllreduceCommWorker {
    // XXX(20160415): Empirically determined message length.
    //let msg_len = 32 * 1024;
    let msg_len = 1 * 1024 * 1024;
    let num_buf_msgs = (paramserver_cfg.buf_size + msg_len - 1) / msg_len;
    //let buf_len = num_buf_msgs * msg_len;
    let buf_len = paramserver_cfg.buf_size;

    let ctx = &(*context).as_ref();
    let origin_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    let target_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };

    //let mpi = Mpi::new();
    let mpi = Mpi::new_serialized();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let mut origin_buf_h = Vec::with_capacity(buf_len);
    unsafe { origin_buf_h.set_len(buf_len) };
    let mut target_buf_h = Vec::with_capacity(buf_len);
    unsafe { target_buf_h.set_len(buf_len) };

    // FIXME(20160416): Need to do multithreaded MPI:
    // - have a sending thread which acquires the target window locks, sends
    //   the messages, and waits on the send completion
    // - have a separate receiving thread which receives the messages from one
    //   and only one source at a time, then upon completion performs the
    //   reduction and notifies the sending thread
    // - each time .communicate() is called, sending (active) thread notifies
    //   the receiving (passive) thread
    // - one round of gossip consists of both a send and a receive
    // - all loads and stores between device and host are performed on the
    //   active thread

    let mut shared_seed = [0, 0];
    if worker_rank == 0 {
      shared_seed = [thread_rng().next_u64(), thread_rng().next_u64()];
    }
    mpi.broadcast(&mut shared_seed, 0);

    MpiDistSyncAllreduceCommWorker{
      worker_data:  WorkerData::new(worker_rank, num_workers),
      context:      context.clone(),
      mpi:          Mpi,
      buf_len:      buf_len,
      msg_len:      msg_len,
      num_buf_msgs: num_buf_msgs,
      com_interval: paramserver_cfg.com_interval,
      origin_buf:   origin_buf,
      target_buf:   target_buf,
      origin_buf_h: origin_buf_h,
      target_buf_h: target_buf_h,
      send_reqs:    MpiRequestList::new(),
      avg_reduce:   AverageReduceOperation::new(0),
      shared_seed:  shared_seed,
      shared_rng:   Xorshiftplus128Rng::from_seed(shared_seed),
      ranks_perm:   (0 .. num_workers).collect(),
      iter_counter: 0,
      bar_signal: false,
    }
  }
}

impl MpiDistCommWorker for MpiDistSyncAllreduceCommWorker {
  fn mpi(&self) -> &Mpi {
    &self.mpi
  }
}

impl CommWorker for MpiDistSyncAllreduceCommWorker {
  fn worker_data(&self) -> &WorkerData {
    &self.worker_data
  }

  fn next(&mut self) -> bool {
    // FIXME(20160412)
    self.iter_counter += 1;
    true
  }

  fn signal_barrier(&mut self) {
    self.bar_signal = true;
  }

  fn wait_barrier(&mut self) -> bool {
    if self.bar_signal {
      Mpi::barrier_().unwrap();
      self.bar_signal = false;
      true
    } else {
      false
    }
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    /*if self.iter_counter % self.com_interval != 0 {
      return;
    }*/
    let ctx = &(*self.context).as_ref();
    let data_len = data.len();
    let data = data.as_view(ctx).data;
    data.raw_send(
        &self.origin_buf.as_ref_range(offset, offset + data_len),
    );
  }

  fn complete_load(&mut self) {
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }

  fn communicate_first(&mut self) {
    // Do nothing.
  }

  fn communicate(&mut self, _repeat: bool /*, ctx: &DeviceCtxRef*/) {
    /*if self.iter_counter % self.com_interval != 0 {
      return;
    }*/
    let ctx = &(*self.context).as_ref();
    self.origin_buf.sync_store(&mut self.origin_buf_h, ctx);
    ctx.sync();
    self.send_reqs.clear();
    //Mpi::barrier_().unwrap();
    for msg in 0 .. self.num_buf_msgs {
      /*Mpi::allreduce_(
          &self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len],
          &mut self.target_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len],
          MpiSumOp,
      ).unwrap();*/
      let send_req = match Mpi::nonblocking_allreduce_(
          &self.origin_buf_h[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          &mut self.target_buf_h[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          MpiSumOp)
      {
        Err(e) => panic!("nonblocking allreduce failed: {:?}", e),
        Ok(req) => req,
      };
      self.send_reqs.append(send_req);
    }
    self.send_reqs.wait_all().unwrap();
    //Mpi::barrier_().unwrap();
    self.target_buf.sync_load(&self.target_buf_h, ctx);
    self.target_buf.as_ref().async_vector_scale(1.0 / self.worker_data.num_workers() as f32, ctx);
    ctx.sync();
  }

  fn communicate_exact(&mut self) {
    let ctx = &(*self.context).as_ref();
    self.origin_buf.sync_store(&mut self.origin_buf_h, ctx);
    ctx.sync();
    self.send_reqs.clear();
    for msg in 0 .. self.num_buf_msgs {
      let send_req = match Mpi::nonblocking_allreduce_(
          &self.origin_buf_h[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          &mut self.target_buf_h[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          MpiSumOp)
      {
        Err(e) => panic!("nonblocking allreduce failed: {:?}", e),
        Ok(req) => req,
      };
      self.send_reqs.append(send_req);
    }
    self.send_reqs.wait_all().unwrap();
    Mpi::barrier_().unwrap();
    self.target_buf.sync_load(&self.target_buf_h, ctx);
    self.target_buf.as_ref().async_vector_scale(1.0 / self.worker_data.num_workers() as f32, ctx);
    ctx.sync();
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    /*if self.iter_counter % self.com_interval != 0 {
      return;
    }*/
    let ctx = &(*self.context).as_ref();
    let data_len = data.len();
    let mut data = data.as_view_mut(ctx).data;
    data.raw_recv(
        &self.target_buf.as_ref_range(offset, offset + data_len),
    );
  }

  fn complete_store(&mut self) {
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }

  fn allreduce(&mut self, src_data: &[f32], dst_data: &mut [f32]) {
    assert_eq!(src_data.len(), dst_data.len());
    let n = src_data.len();
    let num_msgs = (n + self.msg_len - 1) / self.msg_len;
    for msg in 0 .. num_msgs {
      Mpi::allreduce_(
          &src_data[msg * self.msg_len .. min(n, (msg+1) * self.msg_len)],
          &mut dst_data[msg * self.msg_len .. min(n, (msg+1) * self.msg_len)],
          MpiSumOp,
      ).unwrap();
    }
    Mpi::barrier_().unwrap();
  }
}
