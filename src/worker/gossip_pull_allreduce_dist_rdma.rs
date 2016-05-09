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
use array_cuda::device::linalg::{BlasVectorExt, AsyncBlasVectorExt};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer, RawDeviceBuffer};
use array_new::{AsyncArray};
use rng::xorshift::{Xorshiftplus128Rng};
use worker_::{WorkerData};

use mpi::{Mpi, MpiComm, MpiGroup, MpiRequestList, MpiSumOp, MpiWindowLockMode, MpiWindow, MpiMemory, MpiOwnedWindow};
//use procgroup::{ProcGroup};
use threadpool::{ThreadPool};

use rand::{Rng, SeedableRng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{min};
use std::collections::{HashSet};
use std::ffi::{CString};
use std::fs::{OpenOptions, copy, create_dir_all, read_link, remove_file};
use std::io::{Read, BufRead, Write, BufReader};
use std::iter::{FromIterator, repeat};
use std::marker::{PhantomData};
use std::os::unix::fs::{symlink};
use std::path::{Path, PathBuf};
use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering, fence};
use std::sync::mpsc::{Sender, Receiver, TryRecvError, channel};
use std::thread::{JoinHandle, sleep, spawn};
use std::time::{Duration};
use vec_map::{VecMap};

pub struct MpiDistAsyncPullGossipAllreduceCommWorker {
  worker_data:  WorkerData,
  context:      Rc<DeviceContext>,
  mpi:          Mpi,

  buf_len:      usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  /*rdma_buf_len: usize,
  num_rdma_buf_msgs:    usize,*/
  //com_interval: usize,

  src_buf:      DeviceBuffer<f32>,
  allsrc_buf_h: Vec<f32>,
  alldst_buf_h: Vec<f32>,
  pull_target_win:    MpiOwnedWindow<f32, MpiMemory<f32>>,
  pull_origin_buf_h:  MpiMemory<f32>,
  reduce_buf_h: Vec<f32>,
  dst_buf:      DeviceBuffer<f32>,

  group_size:   usize,
  group_rank:   usize,
  num_groups:   usize,
  groups:       Vec<MpiGroup>,
  group_comms:  Vec<MpiComm>,

  checkpt_sig:  bool,

  send_reqs:    MpiRequestList,
  recv_count:   Arc<AtomicUsize>,
  bar_signal:   Arc<AtomicBool>,

  avg_reduce:   AverageReduceOperation<f32>,

  shared_seed:  [u64; 2],
  shared_rng:   Xorshiftplus128Rng,
  local_rng:    Xorshiftplus128Rng,
  //ranks_perm:   Vec<usize>,
  ranks_range:  Range<usize>,
  group_idxs_range: Range<usize>,
  iter_counter: usize,
  //recv_success: bool,
}

impl MpiDistAsyncPullGossipAllreduceCommWorker {
  pub fn new(group_size: usize, msg_len: usize, req_buf_len: usize, context: Rc<DeviceContext>) -> MpiDistAsyncPullGossipAllreduceCommWorker {
    // XXX(20160415): Empirically determined message length.
    /*//let msg_len = 32 * 1024;
    let msg_len = 1 * 1024 * 1024;
    //let msg_len = 4 * 1024 * 1024;*/
    let num_buf_msgs = (req_buf_len + msg_len - 1) / msg_len;
    let buf_len = req_buf_len;
    // XXX(20160501): For RDMA specifically, the buffer length is increased
    // by one for a normalization value.
    /*let num_rdma_buf_msgs = (gossip_cfg.buf_size + 1 + msg_len - 1) / msg_len;
    let rdma_buf_len = gossip_cfg.buf_size + 1; //num_rdma_buf_msgs * msg_len;*/

    let mpi = Mpi::new_serialized();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let ctx = &(*context).as_ref();
    let dev_idx = ctx.device();

    let src_buf = DeviceBuffer::zeros(buf_len, ctx);
    let dst_buf = DeviceBuffer::zeros(buf_len, ctx);

    let mut allsrc_buf_h = Vec::with_capacity(buf_len);
    for _ in 0 .. buf_len {
      allsrc_buf_h.push(0.0);
    }
    let mut alldst_buf_h = Vec::with_capacity(buf_len);
    for _ in 0 .. buf_len {
      alldst_buf_h.push(0.0);
    }

    let pull_target_buf_h = MpiMemory::alloc_(buf_len).unwrap();
    let pull_target_win = MpiOwnedWindow::create_(pull_target_buf_h).unwrap();
    let pull_origin_buf_h = MpiMemory::alloc_(buf_len).unwrap();
    let mut reduce_buf_h = Vec::with_capacity(buf_len);
    for _ in 0 .. buf_len {
      reduce_buf_h.push(0.0);
    }

    assert_eq!(0, num_workers % group_size);
    let group_rank = worker_rank / group_size;
    let num_groups = num_workers / group_size;
    let world_group = match MpiComm::world().group_() {
      Err(e) => panic!("failed to get world group: {:?}", e),
      Ok(group) => group,
    };
    let mut groups = Vec::with_capacity(num_groups);
    for group_idx in 0 .. num_groups {
      let group = match world_group.ranges(&[(group_idx * group_size, (group_idx+1) * group_size, 1)]) {
        Err(e) => panic!("failed to create subgroup from range: {:?}", e),
        Ok(group) => group,
      };
      groups.push(group);
    }
    let mut group_comms = Vec::with_capacity(num_groups);
    for group_idx in 0 .. num_groups {
      /*let group_comm = match if group_idx == group_rank {
        MpiComm::world().create_(&groups[group_idx])
      } else {
        MpiComm::world().create_(&MpiGroup::empty_())
      } {*/
      let group_comm = match MpiComm::world().create_(&groups[group_idx]) {
        Err(e) => panic!("failed to create subgroup comm: {:?}", e),
        Ok(comm) => comm,
      };
      group_comms.push(group_comm);
    }

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

    let recv_count = Arc::new(AtomicUsize::new(0));
    let bar_signal = Arc::new(AtomicBool::new(false));

    let mut shared_seed = [0, 0];
    if worker_rank == 0 {
      shared_seed = [thread_rng().next_u64(), thread_rng().next_u64()];
    }
    if num_workers > 1 {
      mpi.broadcast(&mut shared_seed, 0);
    }

    MpiDistAsyncPullGossipAllreduceCommWorker{
      worker_data:  WorkerData::new(worker_rank, num_workers),
      context:      context.clone(),
      mpi:          Mpi,
      buf_len:      buf_len,
      msg_len:      msg_len,
      num_buf_msgs: num_buf_msgs,
      /*rdma_buf_len: rdma_buf_len,
      num_rdma_buf_msgs:    num_rdma_buf_msgs,*/
      //com_interval: 1,
      src_buf:      src_buf,
      allsrc_buf_h: allsrc_buf_h,
      alldst_buf_h: alldst_buf_h,
      pull_target_win:    pull_target_win,
      pull_origin_buf_h:  pull_origin_buf_h,
      reduce_buf_h: reduce_buf_h,
      dst_buf:      dst_buf,
      group_size:   group_size,
      group_rank:   group_rank,
      num_groups:   num_groups,
      groups:       groups,
      group_comms:  group_comms,
      checkpt_sig:  false,
      send_reqs:    MpiRequestList::new(),
      recv_count:   recv_count,
      bar_signal:   bar_signal,
      avg_reduce:   AverageReduceOperation::new(0),
      shared_seed:  shared_seed,
      shared_rng:   Xorshiftplus128Rng::from_seed(shared_seed),
      local_rng:    Xorshiftplus128Rng::new(&mut thread_rng()),
      //ranks_perm:   (0 .. num_workers).collect(),
      ranks_range:  Range::new(0, num_workers),
      group_idxs_range: Range::new(0, num_groups),
      iter_counter: 0,
      //recv_success: true,
    }
  }
}

impl MpiDistCommWorker for MpiDistAsyncPullGossipAllreduceCommWorker {
  fn mpi(&self) -> &Mpi {
    &self.mpi
  }
}

impl CommWorker for MpiDistAsyncPullGossipAllreduceCommWorker {
  fn worker_data(&self) -> &WorkerData {
    &self.worker_data
  }

  fn next(&mut self) -> bool {
    // FIXME(20160412)
    self.iter_counter += 1;
    true
  }

  fn signal_barrier(&mut self) {
    if self.worker_data.worker_rank() == 0 {
      let worker_rank = self.worker_data.worker_rank();
      let num_workers = self.worker_data.num_workers();

      let dummy_buf: Vec<u8> = Vec::with_capacity(64);
      self.send_reqs.clear();
      for r in 0 .. num_workers {
        if r == worker_rank {
          continue;
        }
        // FIXME(20160501): send to world communicator.
        let send_req = match MpiComm::world().nonblocking_sync_send(&dummy_buf, r, 2) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: active thread: failed to do initial send: {:?}", e),
        };
        self.send_reqs.append(send_req);
      }
      self.send_reqs.wait_all();

      self.checkpt_sig = true;
    }
  }

  fn wait_barrier(&mut self) -> bool {
    if self.worker_data.worker_rank() != 0 {
      let mut chk_recv_rank = None;
      match MpiComm::world().nonblocking_probe(None, Some(2)) {
        Err(e) => panic!("async gossip: passive thread: failed to do first recv: {:?}", e),
        Ok(None) => {}
        Ok(Some(status)) => {
          chk_recv_rank = Some(status.src_rank);
        }
      }
      if let Some(chk_recv_rank) = chk_recv_rank {
        let mut dummy_buf: Vec<u8> = Vec::with_capacity(64);
        let mut recv_req = match MpiComm::world().nonblocking_recv(&mut dummy_buf, Some(chk_recv_rank), Some(2)) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
        };
        recv_req.wait().unwrap();
        self.checkpt_sig = true;
      }
    }

    if !self.checkpt_sig {
      false
    } else {
      Mpi::barrier_().unwrap();
      self.checkpt_sig = false;
      true
    }
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    let ctx = &(*self.context).as_ref();
    let data_len = data.len();
    let data = data.as_view(ctx).data;
    data.send(
        &mut self.src_buf.as_ref_mut_range(offset, offset + data_len, ctx),
    );
  }

  fn complete_load(&mut self) {
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }

  fn communicate_first(&mut self) {
    // Do nothing.
  }

  fn communicate(&mut self) {
    let self_rank = self.worker_data.worker_rank();
    let group_rank = self.group_rank;

    let ctx = &(*self.context).as_ref();
    self.src_buf.as_ref_mut(ctx).sync_store(&mut self.allsrc_buf_h);

    // FIXME(20160508): do a reduce, not an allreduce.
    self.send_reqs.clear();
    for msg in 0 .. self.num_buf_msgs {
      let req = match self.group_comms[group_rank].nonblocking_reduce_(
          &self.allsrc_buf_h[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          &mut self.alldst_buf_h[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          MpiSumOp, 0)
      {
        Err(e) => panic!("nonblocking reduce failed: {:?}", e),
        Ok(req) => req,
      };
      self.send_reqs.append(req);
    }
    self.send_reqs.wait_all().unwrap();

    if self_rank % self.group_size == 0 {
      self.pull_target_win.lock(self_rank, MpiWindowLockMode::Exclusive).unwrap();
      self.pull_target_win.as_mut_slice().copy_from_slice(&self.alldst_buf_h);
      self.pull_target_win.unlock(self_rank).unwrap();

      //let recv_rank = self.ranks_range.ind_sample(&mut self.local_rng);
      let recv_group_idx = self.group_idxs_range.ind_sample(&mut self.local_rng);
      let recv_rank = recv_group_idx * self.group_size;

      if self_rank != recv_rank {
        self.pull_target_win.lock(recv_rank, MpiWindowLockMode::Shared).unwrap();
        let mut origin_buf_h = self.pull_origin_buf_h.as_mut();
        for msg in 0 .. self.num_buf_msgs {
          unsafe { self.pull_target_win.get_(
              origin_buf_h.as_mut_ptr().offset((msg * self.msg_len) as isize),
              min(self.msg_len, self.buf_len - msg * self.msg_len),
              recv_rank,
              msg * self.msg_len,
          ) }.unwrap();
        }
        self.pull_target_win.unlock(recv_rank).unwrap();
      }

      if self_rank == recv_rank {
        self.src_buf.as_ref(ctx).send(&mut self.dst_buf.as_ref_mut(ctx));
      } else {
        self.dst_buf.as_ref_mut(ctx).sync_load(self.pull_origin_buf_h.as_ref());
        self.dst_buf.as_ref_mut(ctx).row_vector_sum(1.0, &self.src_buf.as_ref(ctx));
        self.dst_buf.as_ref_mut(ctx).row_vector_scale(0.5);
      }
      ctx.sync();
    }

    // FIXME(20160508): do a broadcast.
    self.send_reqs.clear();
    for msg in 0 .. self.num_buf_msgs {
      let req = match self.group_comms[group_rank].nonblocking_broadcast_(
          &mut self.alldst_buf_h[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          0)
      {
        Err(e) => panic!("nonblocking broadcast failed: {:?}", e),
        Ok(req) => req,
      };
      self.send_reqs.append(req);
    }
    self.send_reqs.wait_all().unwrap();
  }

  fn communicate_exact(&mut self) {
    let ctx = &(*self.context).as_ref();
    self.src_buf.as_ref(ctx).sync_store(self.pull_origin_buf_h.as_mut());
    ctx.sync();
    for msg in 0 .. self.num_buf_msgs {
      Mpi::allreduce_(
          &self.pull_origin_buf_h.as_ref()[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          &mut self.reduce_buf_h[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          MpiSumOp,
      ).unwrap();
    }
    Mpi::barrier_().unwrap();
    self.dst_buf.as_ref_mut(ctx).sync_load(&self.reduce_buf_h);
    self.dst_buf.as_ref_mut(ctx).row_vector_scale(1.0 / self.worker_data.num_workers() as f32);
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

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    let ctx = &(*self.context).as_ref();
    let data_len = data.len();
    let mut data = data.as_view_mut(ctx).data;
    data.recv(
        &self.dst_buf.as_ref_range(offset, offset + data_len, ctx),
    );
  }

  fn complete_store(&mut self) {
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }
}
