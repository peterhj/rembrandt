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
use array_cuda::device::linalg::{AsyncBlasVectorExt};
use array_cuda::device::memory::{RawDeviceBuffer};
use array_new::{AsyncArray};
use rng::xorshift::{Xorshiftplus128Rng};
use worker_::{WorkerData};

use mpi::{Mpi, MpiComm, MpiRequestList, MpiSumOp, MpiWindowLockMode, MpiWindow, MpiOwnedWindow};
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

enum AsyncPushGossipAct2PassMsg {
  Quit,
  Pause,
  Resume,
}

enum AsyncPushGossipPass2ActMsg {
  AckPause,
}

enum AsyncPushGossipPassiveState {
  Receiving,
  Paused,
}

struct MpiDistAsyncPushGossipPassiveWorker {
  worker_rank:  usize,
  num_workers:  usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  state:        AsyncPushGossipPassiveState,
  context:      DeviceContext,
  reduce_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  target_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  target_buf_h: Arc<Mutex<Vec<f32>>>,
  rdma_buf_h:   Vec<f32>,
  rdma_win:     Arc<Mutex<MpiWindow<f32>>>,
  act2pass_rx:  Receiver<AsyncPushGossipAct2PassMsg>,
  pass2act_tx:  Sender<AsyncPushGossipPass2ActMsg>,
  //client_conns: VecMap<MpiComm>,
  recv_reqs:    MpiRequestList,
  recv_count:   Arc<AtomicUsize>,
  bar_signal:   Arc<AtomicBool>,
  rng:          Xorshiftplus128Rng,
}

impl MpiDistAsyncPushGossipPassiveWorker {
  pub fn run(mut self) {
    let sleep_duration = Duration::new(0, 50_000);
    let mut poll_ranks: Vec<_> = (0 .. self.num_workers).collect();

    'poll_loop: loop {
      match self.act2pass_rx.try_recv() {
        Err(TryRecvError::Empty) => {}
        Err(e) => panic!("async gossip: passive thread: failed to poll receiver: {:?}", e),

        Ok(AsyncPushGossipAct2PassMsg::Pause) => {
          // FIXME(20160501): transfer the rdma buffer to gpu; need an exclusive
          // lock here.
          {
            let mut rdma_win = self.rdma_win.lock().unwrap();
            let mut target_buf_h = self.target_buf_h.lock().unwrap();
            rdma_win.lock(self.worker_rank, MpiWindowLockMode::Exclusive).unwrap();
            unsafe { rdma_win.get_(
                target_buf_h.as_mut_ptr(),
                target_buf_h.len(),
                self.worker_rank,
                0,
            ) };
            rdma_win.unlock(self.worker_rank).unwrap();
          }

          {
            let ctx = &self.context.as_ref();
            let mut target_buf_h = self.target_buf_h.lock().unwrap();
            let mut reduce_buf = self.reduce_buf.lock().unwrap();
            reduce_buf.sync_load(&target_buf_h, ctx);
          }

          self.state = AsyncPushGossipPassiveState::Paused;
          match self.pass2act_tx.send(AsyncPushGossipPass2ActMsg::AckPause) {
            Err(e) => panic!("async gossip: passive thread: failed to send AckPause: {:?}", e),
            Ok(_) => {}
          }
        }

        Ok(AsyncPushGossipAct2PassMsg::Resume) => {
          self.state = AsyncPushGossipPassiveState::Receiving;
        }

        Ok(_) => unimplemented!(),
      }

      match self.state {
        AsyncPushGossipPassiveState::Receiving => {}
        AsyncPushGossipPassiveState::Paused => {
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

pub struct MpiDistAsyncPushGossipCommWorker {
  worker_data:  WorkerData,
  context:      Rc<DeviceContext>,
  mpi:          Mpi,

  buf_len:      usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  com_interval: usize,

  origin_buf:   RawDeviceBuffer<f32>,
  reduce_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  target_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  final_buf:    RawDeviceBuffer<f32>,
  origin_buf_h: Vec<f32>,
  target_buf_h: Arc<Mutex<Vec<f32>>>,
  final_buf_h:  Vec<f32>,
  rdma_win:     Arc<Mutex<MpiWindow<f32>>>,

  //client_conns: VecMap<MpiComm>,
  //server_conns: VecMap<MpiComm>,
  //server_ports: VecMap<CString>,

  act2pass_tx:  Sender<AsyncPushGossipAct2PassMsg>,
  pass2act_rx:  Receiver<AsyncPushGossipPass2ActMsg>,
  passive_thr:  JoinHandle<()>,
  send_reqs:    MpiRequestList,
  recv_count:   Arc<AtomicUsize>,
  bar_signal:   Arc<AtomicBool>,

  avg_reduce:   AverageReduceOperation<f32>,

  shared_seed:  [u64; 2],
  shared_rng:   Xorshiftplus128Rng,
  local_rng:    Xorshiftplus128Rng,
  //ranks_perm:   Vec<usize>,
  ranks_range:  Range<usize>,
  iter_counter: usize,
  //recv_success: bool,
}

impl MpiDistAsyncPushGossipCommWorker {
  pub fn new(gossip_cfg: GossipConfig, context: Rc<DeviceContext>) -> MpiDistAsyncPushGossipCommWorker {
    // XXX(20160415): Empirically determined message length.
    let msg_len = 32 * 1024;
    let num_buf_msgs = (gossip_cfg.buf_size + msg_len - 1) / msg_len;
    let buf_len = num_buf_msgs * msg_len;
    // XXX(20160501): For RDMA specifically, the buffer length is increased
    // by one for a normalization value.
    let num_rdma_buf_msgs = (gossip_cfg.buf_size + 1 + msg_len - 1) / msg_len;
    let rdma_buf_len = num_rdma_buf_msgs * msg_len;

    let mpi = Mpi::new_serialized();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let ctx = &(*context).as_ref();
    let dev_idx = ctx.device();
    let origin_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    origin_buf.as_ref().async_set_constant(0.0, ctx);
    let reduce_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    reduce_buf.as_ref().async_set_constant(0.0, ctx);
    let target_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    target_buf.as_ref().async_set_constant(0.0, ctx);
    let final_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    final_buf.as_ref().async_set_constant(0.0, ctx);
    ctx.sync();
    let reduce_buf = Arc::new(Mutex::new(reduce_buf));
    let target_buf = Arc::new(Mutex::new(target_buf));

    let mut origin_buf_h = Vec::with_capacity(buf_len);
    unsafe { origin_buf_h.set_len(buf_len) };
    let mut target_buf_h = Vec::with_capacity(buf_len);
    unsafe { target_buf_h.set_len(buf_len) };
    let mut final_buf_h = Vec::with_capacity(buf_len);
    unsafe { final_buf_h.set_len(buf_len) };
    let target_buf_h = Arc::new(Mutex::new(target_buf_h));

    let mut rdma_buf_h = Vec::with_capacity(rdma_buf_len);
    for _ in 0 .. rdma_buf_len {
      rdma_buf_h.push(0.0);
    }
    let rdma_win = unsafe { MpiWindow::create_(rdma_buf_h.as_mut_ptr(), rdma_buf_h.len()) }.unwrap();
    let rdma_win = Arc::new(Mutex::new(rdma_win));

    /*let service_port = Mpi::open_port_().unwrap();
    let mut service_name_buf = vec![];
    write!(&mut service_name_buf, "rembrandt_server_{}", worker_rank);
    let service_name = CString::new(service_name_buf).unwrap();
    //println!("DEBUG: rank: {} service name: {:?}", worker_rank, service_name);
    if num_workers > 1 {
      Mpi::publish_service_(&service_name, false, true, &service_port).unwrap();
      Mpi::barrier_().unwrap();
    }

    let mut client_conns = VecMap::with_capacity(num_workers);
    let mut server_conns = VecMap::with_capacity(num_workers);
    let mut server_ports = VecMap::with_capacity(num_workers);
    // FIXME(20160419): it's quadratic, but I tried the naive non-quadratic
    // version and it didn't work...
    for server_r in 0 .. num_workers {
      for client_r in 0 .. num_workers {
        if client_r == server_r {
          continue;
        }
        if worker_rank == server_r {
          //println!("DEBUG: server rank: {} connected to client rank: {}", worker_rank, client_r);
          let client_conn = MpiComm::accept(&service_port).unwrap();
          client_conns.insert(client_r, client_conn);
        }
        if worker_rank == client_r {
          //println!("DEBUG: client rank: {} connected to server rank: {}", worker_rank, server_r);
          let mut server_name_buf = vec![];
          write!(&mut server_name_buf, "rembrandt_server_{}", server_r);
          let server_name = CString::new(server_name_buf).unwrap();
          let server_port = Mpi::lookup_service_(&server_name).unwrap();
          let server_conn = MpiComm::connect(&server_port).unwrap();
          server_conns.insert(server_r, server_conn);
          server_ports.insert(server_r, server_port);
        }
        /*if worker_rank == server_r {
          println!("DEBUG: connected: ({}, {})", server_r, client_r);
        }*/
        Mpi::barrier_().unwrap();
      }
    }*/

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

    let (act2pass_tx, act2pass_rx) = channel();
    let (pass2act_tx, pass2act_rx) = channel();
    let recv_count = Arc::new(AtomicUsize::new(0));
    let bar_signal = Arc::new(AtomicBool::new(false));

    let passive_thr = {
      let reduce_buf = reduce_buf.clone();
      let target_buf = target_buf.clone();
      let target_buf_h = target_buf_h.clone();
      //let rdma_buf_h = rdma_buf_h;
      let rdma_win = rdma_win.clone();
      let recv_count = recv_count.clone();
      let bar_signal = bar_signal.clone();
      spawn(move || {
        let passive_worker = MpiDistAsyncPushGossipPassiveWorker{
          worker_rank:  worker_rank,
          num_workers:  num_workers,
          msg_len:      msg_len,
          num_buf_msgs: num_buf_msgs,
          state:        AsyncPushGossipPassiveState::Receiving,
          context:      DeviceContext::new(dev_idx),
          reduce_buf:   reduce_buf,
          target_buf:   target_buf,
          target_buf_h: target_buf_h,
          rdma_buf_h:   rdma_buf_h,
          rdma_win:     rdma_win,
          act2pass_rx:  act2pass_rx,
          pass2act_tx:  pass2act_tx,
          //client_conns: client_conns,
          recv_reqs:    MpiRequestList::new(),
          recv_count:   recv_count,
          bar_signal:   bar_signal,
          rng:          Xorshiftplus128Rng::new(&mut thread_rng()),
        };
        passive_worker.run();
      })
    };

    let mut shared_seed = [0, 0];
    if worker_rank == 0 {
      shared_seed = [thread_rng().next_u64(), thread_rng().next_u64()];
    }
    if num_workers > 1 {
      mpi.broadcast(&mut shared_seed, 0);
    }

    MpiDistAsyncPushGossipCommWorker{
      worker_data:  WorkerData::new(worker_rank, num_workers),
      context:      context.clone(),
      mpi:          Mpi,
      buf_len:      buf_len,
      msg_len:      msg_len,
      num_buf_msgs: num_buf_msgs,
      com_interval: 1,
      /*world_group:  world_group,
      solo_groups:  solo_groups,
      pair_groups:  pair_groups,*/
      origin_buf:   origin_buf,
      reduce_buf:   reduce_buf,
      target_buf:   target_buf,
      final_buf:    final_buf,
      origin_buf_h: origin_buf_h,
      target_buf_h: target_buf_h,
      final_buf_h:  final_buf_h,
      //target_win_h: target_win_h,
      rdma_win:     rdma_win,
      //client_conns: client_conns,
      //server_conns: server_conns,
      //server_ports: server_ports,
      act2pass_tx:  act2pass_tx,
      pass2act_rx:  pass2act_rx,
      passive_thr:  passive_thr,
      send_reqs:    MpiRequestList::new(),
      recv_count:   recv_count,
      bar_signal:   bar_signal,
      avg_reduce:   AverageReduceOperation::new(0),
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

impl MpiDistCommWorker for MpiDistAsyncPushGossipCommWorker {
  fn mpi(&self) -> &Mpi {
    &self.mpi
  }
}

impl CommWorker for MpiDistAsyncPushGossipCommWorker {
  fn worker_data(&self) -> &WorkerData {
    &self.worker_data
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
      let prev_signal = self.bar_signal.compare_and_swap(false, true, Ordering::AcqRel);
      assert!(!prev_signal, "signal_barrier: tried to signal during an ongoing barrier");

      let worker_rank = self.worker_data.worker_rank();
      let num_workers = self.worker_data.num_workers();

      let dummy_buf: Vec<u8> = Vec::with_capacity(64);

      self.send_reqs.clear();
      for r in 0 .. num_workers {
        if r == worker_rank {
          continue;
        }
        // FIXME(20160501): send to world communicator.
        /*let send_req = match self.server_conns[r].nonblocking_sync_send(&dummy_buf, 0, 2) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: active thread: failed to do initial send: {:?}", e),
        };
        self.send_reqs.append(send_req);*/
      }
      self.send_reqs.wait_all();
    }
  }

  fn wait_barrier(&mut self) -> bool {
    let bar_signal = self.bar_signal.load(Ordering::Acquire);
    if !bar_signal {
      false
    } else {
      Mpi::barrier_().unwrap();
      self.bar_signal.store(false, Ordering::Release);
      true
    }
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }
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

  fn communicate(&mut self/*, ctx: &DeviceCtxRef*/) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let ctx = &(*self.context).as_ref();

    let self_rank = self.worker_data.worker_rank();
    let send_rank = self.ranks_range.ind_sample(&mut self.local_rng);

    if self_rank != send_rank {
      self.origin_buf.sync_store(&mut self.origin_buf_h, ctx);

      // FIXME(20160501): transfer the rdma buffer to gpu; need an exclusive
      // lock here.
      let rdma_win = self.rdma_win.lock().unwrap();
      rdma_win.lock(send_rank, MpiWindowLockMode::Exclusive).unwrap();
      unsafe { rdma_win.put_accumulate_(
          self.origin_buf_h.as_ptr(),
          self.origin_buf_h.len(),
          send_rank,
          0,
          MpiSumOp,
      ) }.unwrap();
      rdma_win.unlock(send_rank).unwrap();

      /*//println!("DEBUG: async gossip: active thread ({}): round: {} initial send rank: {}", self_rank, self.iter_counter, send_rank);
      self.send_reqs.clear();
      let send_req = match self.server_conns[send_rank].nonblocking_sync_send(&self.origin_buf_h[ .. self.msg_len], 0, 1) {
        Ok(req) => req,
        Err(e) => panic!("async gossip: active thread: failed to do initial send: {:?}", e),
      };
      self.send_reqs.append(send_req);
      self.send_reqs.wait_all();
      //println!("DEBUG: async gossip: active thread ({}): round: {} remaining sends rank: {}", self_rank, self.iter_counter, send_rank);
      for msg in 1 .. self.num_buf_msgs {
        let send_req = match self.server_conns[send_rank].nonblocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], 0, 0) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: active thread: failed to do nonblocking send: {:?}", e),
        };
        self.send_reqs.append(send_req);
      }
      self.send_reqs.wait_all();*/
    }

    match self.act2pass_tx.send(AsyncPushGossipAct2PassMsg::Pause) {
      Err(e) => panic!("async gossip: active thread: failed to send Pause to passive thread: {:?}", e),
      Ok(_) => {}
    }
    match self.pass2act_rx.recv() {
      Err(e) => {
        panic!("async gossip: active thread: failed to recv AckPause from passive thread: {:?}", e);
      }
      Ok(AsyncPushGossipPass2ActMsg::AckPause) => {
        // Do nothing.
      }
    }

    {
      //let target_buf = self.target_buf.lock().unwrap();
      let reduce_buf = self.reduce_buf.lock().unwrap();
      let recv_count = self.recv_count.load(Ordering::Acquire);

      let self_weight = if self_rank == send_rank {
        2.0
      } else {
        1.0
      };
      /*if self_rank == 0 {
        println!("DEBUG: async push gossip: round: {} recv count: {}", self.iter_counter, recv_count);
      }*/
      if recv_count == 0 {
        self.origin_buf.raw_send(&self.final_buf, ctx);
      } else {
        reduce_buf.as_ref().async_vector_add(self_weight, &self.origin_buf.as_ref(), ctx);
        reduce_buf.as_ref().async_vector_scale(1.0 / (recv_count as f32 + self_weight), ctx);
        reduce_buf.raw_send(&self.final_buf, ctx);
      }

      ctx.sync();

      // Reset the common atomic counter.
      self.recv_count.store(0, Ordering::Release);
      fence(Ordering::AcqRel);
    }

    match self.act2pass_tx.send(AsyncPushGossipAct2PassMsg::Resume) {
      Ok(_) => {}
      Err(e) => panic!("async gossip: active thread: failed to send Resume to passive thread: {:?}", e),
    }
  }

  fn communicate_exact(&mut self) {
    let ctx = &(*self.context).as_ref();
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
    if self.iter_counter % self.com_interval != 0 {
      return;
    }
    let ctx = &(*self.context).as_ref();
    let data_len = data.len();
    let mut data = data.as_view_mut(ctx).data;
    data.raw_recv(
        &self.final_buf.as_ref_range(offset, offset + data_len),
    );
  }

  fn complete_store(&mut self) {
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }
}
