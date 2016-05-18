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
//use worker::allreduce_dist::{MpiDistSyncAllreduceCommWorker};

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::comm::{ReduceOperation, AverageReduceOperation, for_all_devices};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::ext::{DeviceAsyncNumExt};
use array_cuda::device::linalg::{BlasVectorExt, AsyncBlasVectorExt};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer, RawDeviceBuffer};
use array_new::{AsyncArray};
use rng::xorshift::{Xorshiftplus128Rng};
use worker_::{WorkerData};

use mpi::{Mpi, MpiComm, MpiRequestList, MpiSumOp, MpiWindowLockMode, MpiWindow, MpiMemory, MpiOwnedWindow};
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

enum AsyncPullGossipAct2PassMsg {
  Quit,
  Pause,
  Resume,
}

enum AsyncPullGossipPass2ActMsg {
  AckPause,
}

enum AsyncPullGossipPassiveState {
  Receiving,
  Paused,
}

struct MpiDistAsyncPullGossipPassiveWorker {
  worker_rank:  usize,
  num_workers:  usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  rdma_buf_len: usize,
  state:        AsyncPullGossipPassiveState,
  context:      DeviceContext,
  reduce_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  //target_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  //target_buf_h: Arc<Mutex<Vec<f32>>>,
  norm_buf_h:   Arc<Mutex<f32>>,
  rdma_buf_h:   Vec<f32>,
  rdma_win:     Arc<Mutex<MpiWindow<f32>>>,
  act2pass_rx:  Receiver<AsyncPullGossipAct2PassMsg>,
  pass2act_tx:  Sender<AsyncPullGossipPass2ActMsg>,
  //client_conns: VecMap<MpiComm>,
  recv_reqs:    MpiRequestList,
  recv_count:   Arc<AtomicUsize>,
  bar_signal:   Arc<AtomicBool>,
  rng:          Xorshiftplus128Rng,
}

impl MpiDistAsyncPullGossipPassiveWorker {
  pub fn run(mut self) {
    let sleep_duration = Duration::new(0, 50_000);
    let mut poll_ranks: Vec<_> = (0 .. self.num_workers).collect();

    'poll_loop: loop {
      match self.act2pass_rx.try_recv() {
        Err(TryRecvError::Empty) => {}
        Err(e) => panic!("async gossip: passive thread: failed to poll receiver: {:?}", e),

        Ok(AsyncPullGossipAct2PassMsg::Pause) => {
          // FIXME(20160501): transfer the rdma buffer to gpu; need an exclusive
          // lock here.
          {
            let mut rdma_win = self.rdma_win.lock().unwrap();
            //let mut target_buf_h = self.target_buf_h.lock().unwrap();
            rdma_win.lock(self.worker_rank, MpiWindowLockMode::Exclusive).unwrap();
            /*unsafe { rdma_win.get_(
                self.target_buf_h.as_mut_ptr(),
                self.target_buf_h.len(),
                self.worker_rank,
                0,
            ) };*/
            let mut norm_buf_h = self.norm_buf_h.lock().unwrap();
            *norm_buf_h = self.rdma_buf_h[0];
            let ctx = &self.context.as_ref();
            let mut reduce_buf = self.reduce_buf.lock().unwrap();
            reduce_buf.sync_load(&self.rdma_buf_h[1 .. ], ctx);
            rdma_win.unlock(self.worker_rank).unwrap();
          }

          /*{
            let ctx = &self.context.as_ref();
            //let mut target_buf_h = self.target_buf_h.lock().unwrap();
            let mut reduce_buf = self.reduce_buf.lock().unwrap();
            reduce_buf.sync_load(&self.target_buf_h, ctx);
          }*/

          self.state = AsyncPullGossipPassiveState::Paused;
          match self.pass2act_tx.send(AsyncPullGossipPass2ActMsg::AckPause) {
            Err(e) => panic!("async gossip: passive thread: failed to send AckPause: {:?}", e),
            Ok(_) => {}
          }
        }

        Ok(AsyncPullGossipAct2PassMsg::Resume) => {
          {
            let mut rdma_win = self.rdma_win.lock().unwrap();
            rdma_win.lock(self.worker_rank, MpiWindowLockMode::Exclusive).unwrap();
            for i in 0 .. self.rdma_buf_len {
              self.rdma_buf_h[i] = 0.0;
            }
            rdma_win.unlock(self.worker_rank).unwrap();
          }

          self.state = AsyncPullGossipPassiveState::Receiving;
        }

        Ok(_) => unimplemented!(),
      }

      match self.state {
        AsyncPullGossipPassiveState::Receiving => {}
        AsyncPullGossipPassiveState::Paused => {
          sleep(sleep_duration);
          continue 'poll_loop;
        }
      }

      // FIXME(20160501): handle checkpoint signal.
      /*let mut chk_recv_rank = None;
      self.rng.shuffle(&mut poll_ranks);
      'chk_probe_loop: for &r in poll_ranks.iter() {
        if r == self.worker_rank {
          continue 'chk_probe_loop;
        }
        match self.client_conns[r].nonblocking_probe(Some(0), Some(2)) {
          Err(e) => panic!("async gossip: passive thread: failed to do first recv: {:?}", e),
          Ok(None) => {}
          Ok(Some(_)) => {
            chk_recv_rank = Some(r);
            break 'chk_probe_loop;
          }
        };
      }
      if let Some(chk_recv_rank) = chk_recv_rank {
        let mut dummy_buf: Vec<u8> = Vec::with_capacity(64);
        let mut recv_req = match self.client_conns[chk_recv_rank].nonblocking_recv(&mut dummy_buf, Some(0), Some(2)) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
        };
        recv_req.wait().unwrap();
        self.bar_signal.store(true, Ordering::Release);
      }*/

      // FIXME(20160501): how to zero the rdma buffer?
      /*let mut recv_rank = None;
      self.rng.shuffle(&mut poll_ranks);
      'probe_loop: for &r in poll_ranks.iter() {
        if r == self.worker_rank {
          continue 'probe_loop;
        }
        match self.client_conns[r].nonblocking_probe(Some(0), Some(1)) {
          Err(e) => panic!("async gossip: passive thread: failed to do first recv: {:?}", e),
          Ok(None) => {}
          Ok(Some(_)) => {
            recv_rank = Some(r);
            break 'probe_loop;
          }
        };
      }
      if recv_rank.is_none() {
        sleep(sleep_duration);
        continue 'poll_loop;
      }

      {
        let mut target_buf_h = self.target_buf_h.lock().unwrap();
        let recv_rank = recv_rank.unwrap();
        self.recv_reqs.clear();
        let recv_req = match self.client_conns[recv_rank].nonblocking_recv(&mut target_buf_h[ .. self.msg_len], Some(0), Some(1)) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
        };
        self.recv_reqs.append(recv_req);
        self.recv_reqs.wait_all();
        for msg in 1 .. self.num_buf_msgs {
          let recv_req = match self.client_conns[recv_rank].nonblocking_recv(&mut target_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], Some(0), Some(0)) {
            Ok(req) => req,
            Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
          };
          self.recv_reqs.append(recv_req);
        }
        self.recv_reqs.wait_all();

        let ctx = &self.context.as_ref();
        let reduce_buf = self.reduce_buf.lock().unwrap();
        let mut target_buf = self.target_buf.lock().unwrap();

        if 0 == self.recv_count.load(Ordering::Acquire) {
          reduce_buf.as_ref().async_set_constant(0.0, ctx);
        }
        self.recv_count.fetch_add(1, Ordering::AcqRel);
        fence(Ordering::AcqRel);

        target_buf.sync_load(&target_buf_h, ctx);
        reduce_buf.as_ref().async_vector_add(1.0, &target_buf.as_ref(), ctx);
        ctx.sync();
      }*/

      sleep(sleep_duration);
    }
  }
}

pub struct MpiDistAsyncPullGossipCommWorker {
  worker_data:  WorkerData,
  context:      Rc<DeviceContext>,
  mpi:          Mpi,

  buf_len:      usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  rdma_buf_len: usize,
  num_rdma_buf_msgs:    usize,
  num_rounds:   usize,
  //com_interval: usize,

  /*origin_buf:   RawDeviceBuffer<f32>,
  reduce_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  //target_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  final_buf:    RawDeviceBuffer<f32>,
  origin_buf_h: Vec<f32>,
  //target_buf_h: Arc<Mutex<Vec<f32>>>,
  final_buf_h:  Vec<f32>,
  norm_buf_h:   Arc<Mutex<f32>>,
  rdma_win:     Arc<Mutex<MpiWindow<f32>>>,*/

  src_buf:      DeviceBuffer<f32>,
  pull_target_win:    MpiOwnedWindow<f32, MpiMemory<f32>>,
  pull_origin_buf_h:  MpiMemory<f32>,
  reduce_buf_h: Vec<f32>,
  target_buf:   DeviceBuffer<f32>,
  dst_buf:      DeviceBuffer<f32>,
  checkpt_sig:  bool,

  //client_conns: VecMap<MpiComm>,
  //server_conns: VecMap<MpiComm>,
  //server_ports: VecMap<CString>,

  /*act2pass_tx:  Sender<AsyncPullGossipAct2PassMsg>,
  pass2act_rx:  Receiver<AsyncPullGossipPass2ActMsg>,
  passive_thr:  JoinHandle<()>,*/
  send_reqs:    MpiRequestList,
  recv_count:   Arc<AtomicUsize>,
  bar_signal:   Arc<AtomicBool>,
  recv_ranks:   Vec<usize>,

  avg_reduce:   AverageReduceOperation<f32>,

  step_size:    f32,
  shared_seed:  [u64; 2],
  shared_rng:   Xorshiftplus128Rng,
  local_rng:    Xorshiftplus128Rng,
  //ranks_perm:   Vec<usize>,
  ranks_range:  Range<usize>,
  iter_counter: usize,
  //recv_success: bool,
}

impl MpiDistAsyncPullGossipCommWorker {
  pub fn new(msg_len: usize, gossip_cfg: GossipConfig, context: Rc<DeviceContext>) -> MpiDistAsyncPullGossipCommWorker {
    // XXX(20160415): Empirically determined message length.
    /*//let msg_len = 32 * 1024;
    let msg_len = 1 * 1024 * 1024;
    //let msg_len = 4 * 1024 * 1024;*/
    let num_buf_msgs = (gossip_cfg.buf_size + msg_len - 1) / msg_len;
    let buf_len = gossip_cfg.buf_size; //num_buf_msgs * msg_len;
    // XXX(20160501): For RDMA specifically, the buffer length is increased
    // by one for a normalization value.
    let num_rdma_buf_msgs = (gossip_cfg.buf_size + 1 + msg_len - 1) / msg_len;
    let rdma_buf_len = gossip_cfg.buf_size + 1; //num_rdma_buf_msgs * msg_len;
    assert!(gossip_cfg.num_rounds >= 1);

    let mpi = Mpi::new_serialized();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let ctx = &(*context).as_ref();
    let dev_idx = ctx.device();

    let src_buf = DeviceBuffer::zeros(buf_len, ctx);
    let target_buf = DeviceBuffer::zeros(buf_len, ctx);
    let dst_buf = DeviceBuffer::zeros(buf_len, ctx);

    let pull_target_buf_h = MpiMemory::alloc_(buf_len).unwrap();
    let pull_target_win = MpiOwnedWindow::create_(pull_target_buf_h).unwrap();
    let pull_origin_buf_h = MpiMemory::alloc_(buf_len).unwrap();
    let mut reduce_buf_h = Vec::with_capacity(buf_len);
    for _ in 0 .. buf_len {
      reduce_buf_h.push(0.0);
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

    MpiDistAsyncPullGossipCommWorker{
      worker_data:  WorkerData::new(worker_rank, num_workers),
      context:      context.clone(),
      mpi:          Mpi,
      buf_len:      buf_len,
      msg_len:      msg_len,
      num_buf_msgs: num_buf_msgs,
      rdma_buf_len: rdma_buf_len,
      num_rdma_buf_msgs:    num_rdma_buf_msgs,
      num_rounds:   gossip_cfg.num_rounds,
      src_buf:      src_buf,
      pull_target_win:    pull_target_win,
      pull_origin_buf_h:  pull_origin_buf_h,
      reduce_buf_h: reduce_buf_h,
      target_buf:   target_buf,
      dst_buf:      dst_buf,
      checkpt_sig:  false,
      send_reqs:    MpiRequestList::new(),
      recv_count:   recv_count,
      bar_signal:   bar_signal,
      recv_ranks:   vec![],
      avg_reduce:   AverageReduceOperation::new(0),
      step_size:    0.1,
      shared_seed:  shared_seed,
      shared_rng:   Xorshiftplus128Rng::from_seed(shared_seed),
      local_rng:    Xorshiftplus128Rng::new(&mut thread_rng()),
      //ranks_perm:   (0 .. num_workers).collect(),
      ranks_range:  Range::new(0, num_workers),
      iter_counter: 0,
      //recv_success: true,
    }
  }
}

impl MpiDistCommWorker for MpiDistAsyncPullGossipCommWorker {
  fn mpi(&self) -> &Mpi {
    &self.mpi
  }
}

impl CommWorker for MpiDistAsyncPullGossipCommWorker {
  fn worker_data(&self) -> &WorkerData {
    &self.worker_data
  }

  fn hack_set_step_size(&mut self, step_size: f32) {
    self.step_size = step_size;
  }

  fn next(&mut self) -> bool {
    // FIXME(20160412)
    self.iter_counter += 1;
    /*if self.worker_data.worker_rank() == 0 {
      println!("DEBUG: next: {}", self.iter_counter);
    }*/
    true
  }

  fn signal_barrier(&mut self) {
    if self.worker_data.worker_rank() == 0 {
      /*let prev_signal = self.bar_signal.compare_and_swap(false, true, Ordering::AcqRel);
      assert!(!prev_signal, "signal_barrier: tried to signal during an ongoing barrier");*/

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
        /*let send_req = match self.server_conns[r].nonblocking_sync_send(&dummy_buf, 0, 2) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: active thread: failed to do initial send: {:?}", e),
        };
        self.send_reqs.append(send_req);*/
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
    /*if self.iter_counter % self.com_interval != 0 {
      return;
    }*/
    let ctx = &(*self.context).as_ref();
    let data_len = data.len();
    let data = data.as_view(ctx).data;
    /*data.raw_send(
        &self.origin_buf.as_ref_range(offset, offset + data_len),
    );*/
    data.send(
        &mut self.src_buf.as_ref_mut_range(offset, offset + data_len, ctx),
    );
  }

  fn complete_load(&mut self) {
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }

  fn communicate_first(&mut self) {
    let self_rank = self.worker_data.worker_rank();

    let ctx = &(*self.context).as_ref();
    self.pull_target_win.lock(self_rank, MpiWindowLockMode::Exclusive).unwrap();
    self.src_buf.as_ref_mut(ctx).sync_store(self.pull_target_win.as_mut_slice());
    self.pull_target_win.unlock(self_rank).unwrap();
  }

  fn communicate(&mut self, repeat: bool /*, ctx: &DeviceCtxRef*/) {
    let self_rank = self.worker_data.worker_rank();

    let ctx = &(*self.context).as_ref();
    self.pull_target_win.lock(self_rank, MpiWindowLockMode::Exclusive).unwrap();
    self.src_buf.as_ref_mut(ctx).sync_store(self.pull_target_win.as_mut_slice());
    self.pull_target_win.unlock(self_rank).unwrap();

    if self.num_rounds > 1 {
      self.src_buf.as_ref(ctx).send(&mut self.dst_buf.as_ref_mut(ctx));
    }

    if !repeat {
      self.recv_ranks.clear();
    }

    for round in 0 .. self.num_rounds {
      let recv_rank = if !repeat {
        let recv_rank = self.ranks_range.ind_sample(&mut self.local_rng);
        self.recv_ranks.push(recv_rank);
        recv_rank
      } else {
        self.recv_ranks[round]
      };

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

      if self.num_rounds == 1 {
        if self_rank == recv_rank {
          self.src_buf.as_ref(ctx).send(&mut self.dst_buf.as_ref_mut(ctx));
        } else {
          // XXX(20160517): This version has a constantly weighted average.
          /*self.dst_buf.as_ref_mut(ctx).sync_load(self.pull_origin_buf_h.as_ref());
          self.dst_buf.as_ref_mut(ctx).row_vector_sum(1.0, &self.src_buf.as_ref(ctx));
          self.dst_buf.as_ref_mut(ctx).row_vector_scale(0.5);*/

          // XXX(20160517): This version weights the average with the step size.
          self.dst_buf.as_ref_mut(ctx).sync_load(self.pull_origin_buf_h.as_ref());
          self.dst_buf.as_ref_mut(ctx).row_vector_scale(5.0 * self.step_size);
          self.dst_buf.as_ref_mut(ctx).row_vector_sum(1.0 - 5.0 * self.step_size, &self.src_buf.as_ref(ctx));
        }
        ctx.sync();
      } else if self.num_rounds > 1 {
        if self_rank == recv_rank {
          self.dst_buf.as_ref_mut(ctx).row_vector_sum(1.0, &self.src_buf.as_ref(ctx));
        } else {
          self.target_buf.as_ref_mut(ctx).sync_load(self.pull_origin_buf_h.as_ref());
          self.dst_buf.as_ref_mut(ctx).row_vector_sum(1.0, &self.target_buf.as_ref(ctx));
        }
      }
    }

    if self.num_rounds > 1 {
      self.dst_buf.as_ref_mut(ctx).row_vector_scale(1.0 / (self.num_rounds as f32 + 1.0));
      ctx.sync();
    }

    /*self.pull_target_win.lock(self_rank, MpiWindowLockMode::Exclusive).unwrap();
    self.dst_buf.as_ref_mut(ctx).sync_store(self.pull_target_win.as_mut_slice());
    self.pull_target_win.unlock(self_rank).unwrap();*/
  }

  fn communicate_exact(&mut self) {
    let ctx = &(*self.context).as_ref();
    self.src_buf.as_ref(ctx).sync_store(self.pull_origin_buf_h.as_mut());
    //ctx.sync();
    self.send_reqs.clear();
    for msg in 0 .. self.num_buf_msgs {
      //Mpi::allreduce_(
      let req = Mpi::nonblocking_allreduce_(
          &self.pull_origin_buf_h.as_ref()[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          &mut self.reduce_buf_h[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          MpiSumOp,
      ).unwrap();
      self.send_reqs.append(req);
    }
    self.send_reqs.wait_all().unwrap();
    //Mpi::barrier_().unwrap();
    self.dst_buf.as_ref_mut(ctx).sync_load(&self.reduce_buf_h);
    self.dst_buf.as_ref_mut(ctx).row_vector_scale(1.0 / self.worker_data.num_workers() as f32);
    ctx.blocking_sync();

    /*let ctx = &(*self.context).as_ref();
    self.origin_buf.sync_store(&mut self.origin_buf_h, ctx);
    ctx.sync();
    for msg in 0 .. self.num_buf_msgs {
      Mpi::allreduce_(
          &self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len],
          &mut self.final_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len],
          MpiSumOp,
      ).unwrap();
    }
    Mpi::barrier_().unwrap();
    self.final_buf.sync_load(&self.final_buf_h, ctx);
    self.final_buf.as_ref().async_vector_scale(1.0 / self.worker_data.num_workers() as f32, ctx);
    ctx.sync();*/
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
    /*if self.iter_counter % self.com_interval != 0 {
      return;
    }*/
    let ctx = &(*self.context).as_ref();
    let data_len = data.len();
    let mut data = data.as_view_mut(ctx).data;
    /*data.raw_recv(
        &self.final_buf.as_ref_range(offset, offset + data_len),
    );*/
    data.recv(
        &self.dst_buf.as_ref_range(offset, offset + data_len, ctx),
    );
  }

  fn complete_store(&mut self) {
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }
}
