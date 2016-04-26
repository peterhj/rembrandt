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

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::comm::{ReduceOperation, AverageReduceOperation, for_all_devices};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::ext::{DeviceAsyncNumExt};
use array_cuda::device::linalg::{AsyncBlasVectorExt};
use array_cuda::device::memory::{RawDeviceBuffer};
use array_new::{AsyncArray};
use rng::xorshift::{Xorshiftplus128Rng};
use worker_::{WorkerData};

use mpi::{Mpi, MpiComm, MpiRequestList, MpiSumOp};
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

struct MpiDistElasticServerCentralWorker {
  worker_rank:  usize,
  num_workers:  usize,
  msg_len:      usize,
  num_buf_msgs: usize,

  state:        AsyncPushGossipPassiveState,
  context:      DeviceContext,

  reduce_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  target_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  target_buf_h: Arc<Mutex<Vec<f32>>>,

  center_buf:   RawDeviceBuffer<f32>,
  center_buf_h: Vec<f32>,

  //act2pass_rx:  Receiver<AsyncPushGossipAct2PassMsg>,
  //pass2act_tx:  Sender<AsyncPushGossipPass2ActMsg>,
  client_conns: VecMap<MpiComm>,
  recv_reqs:    MpiRequestList,
  recv_count:   Arc<AtomicUsize>,
  bar_signal:   Arc<AtomicBool>,

  rng:          Xorshiftplus128Rng,
}

impl MpiDistElasticServerCentralWorker {
  pub fn run(&mut self) {
    let sleep_duration = Duration::new(0, 50_000);
    let mut poll_ranks: Vec<_> = (0 .. self.num_workers).collect();

    'poll_loop: loop {
      match self.act2pass_rx.try_recv() {
        Err(TryRecvError::Empty) => {}
        Err(e) => panic!("async gossip: passive thread: failed to poll receiver: {:?}", e),

        Ok(AsyncPushGossipAct2PassMsg::Pause) => {
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

      let mut chk_recv_rank = None;
      // FIXME(20160422): could also randomize the order of checked ranks,
      // may make things more unbiased?
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
      }

      let mut recv_rank = None;
      // FIXME(20160422): could also randomize the order of checked ranks,
      // may make things more unbiased?
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
      }

      sleep(sleep_duration);
  }
}

pub struct MpiDistElasticServerCommWorker {
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

  server_conn:  MpiComm,
  server_port:  VecMapCString,

  //act2pass_tx:  Sender<AsyncPushGossipAct2PassMsg>,
  //pass2act_rx:  Receiver<AsyncPushGossipPass2ActMsg>,
  central_thr:  Option<JoinHandle<()>>,
  send_reqs:    MpiRequestList,
  recv_count:   Arc<AtomicUsize>,
  bar_signal:   Arc<AtomicBool>,

  avg_reduce:   AverageReduceOperation<f32>,

  shared_seed:  [u64; 2],
  shared_rng:   Xorshiftplus128Rng,
  local_rng:    Xorshiftplus128Rng,
  ranks_range:  Range<usize>,
  iter_counter: usize,
}

impl MpiDistElasticServerCommWorker {
  pub fn new(gossip_cfg: GossipConfig, context: Rc<DeviceContext>) -> MpiDistElasticServerCommWorker {
    // XXX(20160415): Empirically determined message length.
    let msg_len = 32 * 1024;
    let num_buf_msgs = (gossip_cfg.buf_size + msg_len - 1) / msg_len;
    let buf_len = num_buf_msgs * msg_len;

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

    let mpi = Mpi::new();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let mut origin_buf_h = Vec::with_capacity(buf_len);
    unsafe { origin_buf_h.set_len(buf_len) };
    let mut target_buf_h = Vec::with_capacity(buf_len);
    unsafe { target_buf_h.set_len(buf_len) };
    let mut final_buf_h = Vec::with_capacity(buf_len);
    unsafe { final_buf_h.set_len(buf_len) };
    let target_buf_h = Arc::new(Mutex::new(target_buf_h));

    let mut server_name = None;
    let mut server_conn = None;
    let mut server_port = None;

    if worker_rank == 0 {
      let service_port = Mpi::open_port_().unwrap();
      let mut service_name_buf = vec![];
      write!(&mut service_name_buf, "rembrandt_server_{}", worker_rank);
      let service_name = CString::new(service_name_buf).unwrap();
      //println!("DEBUG: rank: {} service name: {:?}", worker_rank, service_name);
      if num_workers > 1 {
        Mpi::publish_service_(&service_name, false, true, &service_port).unwrap();
      }
    }
    Mpi::barrier_().unwrap();

    let mut client_conns = VecMap::with_capacity(num_workers);
    let mut server_conns = VecMap::with_capacity(num_workers);
    let mut server_ports = VecMap::with_capacity(num_workers);
    for client_r in 0 .. num_workers {
      // FIXME(20160426): how to handle when the worker is the server?
      if client_r == 0 {
        continue;
      }
      if worker_rank == 0 {
        let client_conn = MpiComm::accept(&service_port).unwrap();
        client_conns.insert(client_r, client_conn);
      }
      if worker_rank == client_r {
        let mut server_name_buf = vec![];
        write!(&mut server_name_buf, "rembrandt_server_{}", 0);
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
          act2pass_rx:  act2pass_rx,
          pass2act_tx:  pass2act_tx,
          client_conns: client_conns,
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
      //client_conns: client_conns,
      server_conns: server_conns,
      server_ports: server_ports,
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
