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
  ParameterServerConfig,
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
use std::thread::{JoinHandle, sleep, sleep_ms, spawn};
use std::time::{Duration};
use vec_map::{VecMap};

struct MpiDistElasticServerCentralWorker {
  worker_rank:  usize,
  num_workers:  usize,
  msg_len:      usize,
  num_buf_msgs: usize,

  //state:        AsyncPushGossipPassiveState,
  context:      DeviceContext,

  //reduce_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  //target_buf:   RawDeviceBuffer<f32>,
  target_buf:   DeviceBuffer<f32>,
  target_buf_h: Vec<f32>,

  //center_buf:   RawDeviceBuffer<f32>,
  center_buf:   DeviceBuffer<f32>,
  center_buf_h: Vec<f32>,

  central_bar:  Arc<Barrier>,

  //act2pass_rx:  Receiver<AsyncPushGossipAct2PassMsg>,
  //pass2act_tx:  Sender<AsyncPushGossipPass2ActMsg>,
  //service_name: CString,
  //service_port: CString,
  //client_conns: VecMap<MpiComm>,
  recv_reqs:    MpiRequestList,
  //recv_count:   Arc<AtomicUsize>,
  bar_signal:   Arc<AtomicBool>,

  rng:          Xorshiftplus128Rng,
}

impl MpiDistElasticServerCentralWorker {
  pub fn run(&mut self) {
    //println!("DEBUG: elastic server: central: running...");

    let ctx = &self.context.as_ref();
    let sleep_duration = Duration::new(0, 50_000);
    //let mut poll_ranks: Vec<_> = (0 .. self.num_workers).collect();

    let mut dummy_buf: Vec<u8> = Vec::with_capacity(64);

    // FIXME(20160428): initialization w/ initial params.
    println!("DEBUG: elastic server: central: init params...");
    self.recv_reqs.clear();
    for msg in 0 .. self.num_buf_msgs {
      let recv_req = match MpiComm::world().nonblocking_recv(&mut self.center_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], Some(0), Some(0x31)) {
        Ok(req) => req,
        Err(e) => panic!("elastic server: central: failed to do nonblocking recv: {:?}", e),
      };
      self.recv_reqs.append(recv_req);
    }
    println!("DEBUG: elastic server: central: init params: wait...");
    self.recv_reqs.wait_all().unwrap();
    self.center_buf.as_ref_mut(ctx).sync_load(&self.center_buf_h);
    println!("DEBUG: elastic server: central: init params: done");

    'poll_loop: loop {
      let mut ctrl_recv_rank = None;
      match MpiComm::world().nonblocking_probe(None, Some(0x21)) {
        Err(e) => panic!("elastic server: central: failed to do control recv: {:?}", e),
        Ok(None) => {}
        Ok(Some(status)) => {
          ctrl_recv_rank = Some(status.src_rank);
        }
      }
      if let Some(ctrl_recv_rank) = ctrl_recv_rank {
        //println!("DEBUG: elastic server: central: received ctrl msg");
        // FIXME(20160428): for simplicity, only allow rank 0 to direct the
        // server to do stuff.
        assert_eq!(0, ctrl_recv_rank);

        //println!("DEBUG: elastic server: central: receive ctrl msg...");
        //self.recv_reqs.clear();
        let mut recv_req = match MpiComm::world().nonblocking_recv(&mut dummy_buf[ .. 0], Some(0), Some(0x21)) {
          Ok(req) => req,
          Err(e) => panic!(),
        };
        //self.recv_reqs.append(recv_req);
        //self.recv_reqs.wait_all();
        recv_req.wait().unwrap();

        //println!("DEBUG: elastic server: central: signal clients to barrier...");
        self.recv_reqs.clear();
        for r in 0 .. self.num_workers {
          let send_req = match MpiComm::world().nonblocking_send(&dummy_buf[ .. 0], r, 0x22) {
            Ok(req) => req,
            Err(e) => panic!(),
          };
          self.recv_reqs.append(send_req);
        }
        //send_req.wait().unwrap();
        self.recv_reqs.wait_all().unwrap();

        //println!("DEBUG: elastic server: central: send center params to rank 0...");
        self.recv_reqs.clear();
        let send_req = match MpiComm::world().nonblocking_sync_send(&self.center_buf_h[ .. self.msg_len], 0, 0x20) {
          Ok(req) => req,
          Err(e) => panic!(),
        };
        self.recv_reqs.append(send_req);
        self.recv_reqs.wait_all().unwrap();
        for msg in 1 .. self.num_buf_msgs {
          let send_req = match MpiComm::world().nonblocking_send(&self.center_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], 0, 0x20) {
            Ok(req) => req,
            Err(e) => panic!(),
          };
          self.recv_reqs.append(send_req);
        }
        self.recv_reqs.wait_all().unwrap();
        //println!("DEBUG: elastic server: central: done ctrl");

        // XXX(20160428): now the broadcast happens on the client side.

        /*for r in 0 .. self.num_workers {
          for msg in 0 .. self.num_buf_msgs {
            let send_req = match MpiComm::world().nonblocking_send(&dummy_buf[ .. 0], r, 0x20) {
              Ok(req) => req,
              Err(e) => panic!(),
            };
            self.recv_reqs.append(send_req);
            unimplemented!();
          }
        }
        self.recv_reqs.wait_all();*/
      }

      //println!("DEBUG: elastic server: central: probing...");
      let mut recv_rank = None;
      match MpiComm::world().nonblocking_probe(None, Some(1)) {
        Err(e) => panic!("elastic server: central: failed to do first recv: {:?}", e),
        Ok(None) => {}
        Ok(Some(status)) => {
          recv_rank = Some(status.src_rank);
        }
      }
      if recv_rank.is_none() {
        sleep(sleep_duration);
        continue 'poll_loop;
      }
      //println!("DEBUG: elastic server: central: found source: {}", recv_rank.unwrap());

      {
        //let mut target_buf_h = self.target_buf_h.lock().unwrap();
        let recv_rank = recv_rank.unwrap();
        self.recv_reqs.clear();
        let recv_req = match MpiComm::world().nonblocking_recv(&mut self.target_buf_h[ .. self.msg_len], Some(recv_rank), Some(1)) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
        };
        self.recv_reqs.append(recv_req);
        self.recv_reqs.wait_all();
        for msg in 1 .. self.num_buf_msgs {
          let recv_req = match MpiComm::world().nonblocking_recv(&mut self.target_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], Some(recv_rank), Some(0)) {
            Ok(req) => req,
            Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
          };
          self.recv_reqs.append(recv_req);
        }
        self.recv_reqs.wait_all();

        //let reduce_buf = self.reduce_buf.lock().unwrap();
        //let mut target_buf = self.target_buf.lock().unwrap();

        /*if 0 == self.recv_count.load(Ordering::Acquire) {
          reduce_buf.as_ref().async_set_constant(0.0, ctx);
        }
        self.recv_count.fetch_add(1, Ordering::AcqRel);
        fence(Ordering::AcqRel);*/

        //let mut target_buf_h = self.target_buf_h.lock().unwrap();
        //let recv_rank = recv_rank.unwrap();
        //self.recv_reqs.clear();
        for msg in 0 .. self.num_buf_msgs {
          let send_req = match MpiComm::world().nonblocking_send(&self.center_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], recv_rank, 0x10) {
            Ok(req) => req,
            Err(e) => panic!("async gossip: passive thread: failed to do nonblocking send: {:?}", e),
          };
          self.recv_reqs.append(send_req);
        }
        self.recv_reqs.wait_all();

        let beta = 0.8 / self.num_workers as f32;
        self.target_buf.as_ref_mut(ctx).sync_load(&self.target_buf_h);
        //reduce_buf.as_ref().async_vector_add(1.0, &target_buf.as_ref(), ctx);
        self.center_buf.as_ref_mut(ctx).row_vector_scale(1.0 - beta);
        self.center_buf.as_ref_mut(ctx).row_vector_sum(beta, &self.target_buf.as_ref(ctx));
        //ctx.sync();
        self.center_buf.as_ref(ctx).sync_store(&mut self.center_buf_h);
      }

      //sleep(sleep_duration);
    }
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

  origin_buf:   DeviceBuffer<f32>,
  //reduce_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  //target_buf:   Arc<Mutex<RawDeviceBuffer<f32>>>,
  target_buf:   DeviceBuffer<f32>,
  final_buf:    DeviceBuffer<f32>,
  origin_buf_h: Vec<f32>,
  //target_buf_h: Arc<Mutex<Vec<f32>>>,
  target_buf_h: Vec<f32>,
  //final_buf_h:  Vec<f32>,

  central_bar:  Option<Arc<Barrier>>,

  //server_conn:  MpiComm,
  //server_port:  CString,

  //act2pass_tx:  Sender<AsyncPushGossipAct2PassMsg>,
  //pass2act_rx:  Receiver<AsyncPushGossipPass2ActMsg>,
  central_thr:  Option<JoinHandle<()>>,
  send_reqs:    MpiRequestList,
  //recv_count:   Arc<AtomicUsize>,
  bar_signal:   Arc<AtomicBool>,

  avg_reduce:   AverageReduceOperation<f32>,

  shared_seed:  [u64; 2],
  shared_rng:   Xorshiftplus128Rng,
  local_rng:    Xorshiftplus128Rng,
  ranks_range:  Range<usize>,
  iter_counter: usize,
  rank0_init:   bool,
  rank0_signal: bool,
}

impl MpiDistElasticServerCommWorker {
  pub fn new(paramserver_cfg: ParameterServerConfig, context: Rc<DeviceContext>) -> MpiDistElasticServerCommWorker {
    // XXX(20160415): Empirically determined message length.
    let msg_len = 32 * 1024;
    let num_buf_msgs = (paramserver_cfg.buf_size + msg_len - 1) / msg_len;
    let buf_len = num_buf_msgs * msg_len;

    let ctx = &(*context).as_ref();
    let dev_idx = ctx.device();
    /*let origin_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    origin_buf.as_ref().async_set_constant(0.0, ctx);
    //let reduce_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    //reduce_buf.as_ref().async_set_constant(0.0, ctx);
    let target_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    target_buf.as_ref().async_set_constant(0.0, ctx);
    let final_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    final_buf.as_ref().async_set_constant(0.0, ctx);*/
    let origin_buf = DeviceBuffer::zeros(buf_len, ctx);
    let target_buf = DeviceBuffer::zeros(buf_len, ctx);
    let final_buf = DeviceBuffer::zeros(buf_len, ctx);
    ctx.sync();
    //let reduce_buf = Arc::new(Mutex::new(reduce_buf));
    //let target_buf = Arc::new(Mutex::new(target_buf));

    /*let center_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    center_buf.as_ref().async_set_constant(0.0, ctx);
    let mut center_buf_h = Vec::with_capacity(buf_len);
    for _ in 0 .. buf_len {
      center_buf_h.push(0.0);
    }*/

    let mpi = Mpi::new();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let mut origin_buf_h = Vec::with_capacity(buf_len);
    for _ in 0 .. buf_len {
      origin_buf_h.push(0.0);
    }
    let mut target_buf_h = Vec::with_capacity(buf_len);
    for _ in 0 .. buf_len {
      target_buf_h.push(0.0);
    }
    /*let mut final_buf_h = Vec::with_capacity(buf_len);
    for _ in 0 .. buf_len {
      final_buf_h.push(0.0);
    }*/
    //let target_buf_h = Arc::new(Mutex::new(target_buf_h));

    //let mut server_name = None;
    //let mut server_conn = None;

    /*let mut service_name_buf = vec![];
    write!(&mut service_name_buf, "rembrandt_server_central");
    let service_name = CString::new(service_name_buf).unwrap();
    //println!("DEBUG: rank: {} service name: {:?}", worker_rank, service_name);*/

    /*let mut server_port = None;
    if worker_rank == 0 {
      let service_port = Mpi::open_port_().unwrap();
      Mpi::publish_service_(&service_name, false, true, &service_port).unwrap();
      server_port = Some(service_port);
    }
    Mpi::barrier_().unwrap();*/

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

    let central_bar = if worker_rank == 0 {
      Some(Arc::new(Barrier::new(2)))
    } else {
      None
    };

    //let (act2pass_tx, act2pass_rx) = channel();
    //let (pass2act_tx, pass2act_rx) = channel();
    //let recv_count = Arc::new(AtomicUsize::new(0));
    let bar_signal = Arc::new(AtomicBool::new(false));

    let central_thr = if worker_rank == 0 {
      //let reduce_buf = reduce_buf.clone();
      //let target_buf = target_buf.clone();
      //let target_buf_h = target_buf_h.clone();
      //let service_name = service_name.clone();
      let central_bar = central_bar.clone().unwrap();
      //let recv_count = recv_count.clone();
      let bar_signal = bar_signal.clone();
      Some(spawn(move || {
        let context = DeviceContext::new(dev_idx);

        //let center_buf = unsafe { RawDeviceBuffer::new(buf_len, &context.as_ref()) };
        //center_buf.as_ref().async_set_constant(0.0, &context.as_ref());

        let target_buf = DeviceBuffer::zeros(buf_len, &context.as_ref());
        let mut target_buf_h = Vec::with_capacity(buf_len);
        for _ in 0 .. buf_len {
          target_buf_h.push(0.0);
        }

        let center_buf = DeviceBuffer::zeros(buf_len, &context.as_ref());
        let mut center_buf_h = Vec::with_capacity(buf_len);
        for _ in 0 .. buf_len {
          center_buf_h.push(0.0);
        }

        MpiDistElasticServerCentralWorker{
          worker_rank:  worker_rank,
          num_workers:  num_workers,
          msg_len:      msg_len,
          num_buf_msgs: num_buf_msgs,
          //state:        AsyncPushGossipPassiveState::Receiving,
          context:      context,
          //reduce_buf:   reduce_buf,
          target_buf:   target_buf,
          target_buf_h: target_buf_h,
          center_buf:   center_buf,
          center_buf_h: center_buf_h,
          //act2pass_rx:  act2pass_rx,
          //pass2act_tx:  pass2act_tx,
          central_bar:  central_bar,
          //service_name: service_name,
          //service_port: server_port.unwrap(),
          //client_conns: client_conns,
          //client_conns: VecMap::with_capacity(num_workers),
          recv_reqs:    MpiRequestList::new(),
          //recv_count:   recv_count,
          bar_signal:   bar_signal,
          rng:          Xorshiftplus128Rng::new(&mut thread_rng()),
        }.run();
      }))
    } else {
      None
    };

    /*//let mut client_conns = VecMap::with_capacity(num_workers);
    let mut server_conn = None;
    let mut server_port = None;

    if let Some(ref central_bar) = central_bar {
      central_bar.wait();
    }
    Mpi::barrier_().unwrap();
    for client_r in 0 .. num_workers {
      if worker_rank == client_r {
        println!("DEBUG: connecting: {}", client_r);
        sleep_ms(500);
        /*let mut server_name_buf = vec![];
        write!(&mut server_name_buf, "rembrandt_server_{}", 0);
        let server_name = CString::new(server_name_buf).unwrap();*/
        println!("DEBUG: try to lookup: {}", client_r);
        let service_port = Mpi::lookup_service_(&service_name).unwrap();
        println!("DEBUG: try to connect: {}", client_r);
        let service_conn = MpiComm::connect(&service_port).unwrap();
        server_conn = Some(service_conn);
        server_port = Some(service_port);
        println!("DEBUG: connected: ({})", client_r);
      }
      if let Some(ref central_bar) = central_bar {
        central_bar.wait();
      }
      Mpi::barrier_().unwrap();
    }*/

    let mut shared_seed = [0, 0];
    if worker_rank == 0 {
      shared_seed = [thread_rng().next_u64(), thread_rng().next_u64()];
    }
    if num_workers > 1 {
      mpi.broadcast(&mut shared_seed, 0);
    }

    MpiDistElasticServerCommWorker{
      worker_data:  WorkerData::new(worker_rank, num_workers),
      context:      context.clone(),
      mpi:          Mpi,
      buf_len:      buf_len,
      msg_len:      msg_len,
      num_buf_msgs: num_buf_msgs,
      com_interval: paramserver_cfg.com_interval,
      /*world_group:  world_group,
      solo_groups:  solo_groups,
      pair_groups:  pair_groups,*/
      origin_buf:   origin_buf,
      //reduce_buf:   reduce_buf,
      target_buf:   target_buf,
      final_buf:    final_buf,
      origin_buf_h: origin_buf_h,
      target_buf_h: target_buf_h,
      //final_buf_h:  final_buf_h,
      //target_win_h: target_win_h,
      //client_conns: client_conns,
      //server_conn:  server_conn.unwrap(),
      //server_port:  server_port.unwrap(),
      //act2pass_tx:  act2pass_tx,
      //pass2act_rx:  pass2act_rx,
      central_bar:  central_bar,
      central_thr:  central_thr,
      send_reqs:    MpiRequestList::new(),
      //recv_count:   recv_count,
      bar_signal:   bar_signal,
      avg_reduce:   AverageReduceOperation::new(0),
      shared_seed:  shared_seed,
      shared_rng:   Xorshiftplus128Rng::from_seed(shared_seed),
      local_rng:    Xorshiftplus128Rng::new(&mut thread_rng()),
      //ranks_perm:   (0 .. num_workers).collect(),
      ranks_range:  Range::new(0, num_workers),
      iter_counter: 0,
      rank0_init:   false,
      rank0_signal: false,
      //recv_success: true,
    }
  }
}

impl MpiDistCommWorker for MpiDistElasticServerCommWorker {
  fn mpi(&self) -> &Mpi {
    &self.mpi
  }
}

impl CommWorker for MpiDistElasticServerCommWorker {
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
    assert_eq!(0, self.worker_data.worker_rank());
    let mut dummy_buf: Vec<u8> = Vec::with_capacity(64);
    self.send_reqs.clear();
    let send_req = match MpiComm::world().nonblocking_sync_send(&mut dummy_buf[ .. 0], 0, 0x21) {
      Ok(req) => req,
      Err(e) => panic!(),
    };
    self.send_reqs.append(send_req);
    self.send_reqs.wait_all();
    self.rank0_signal = true;

    /*let prev_signal = self.bar_signal.compare_and_swap(false, true, Ordering::AcqRel);
    assert!(!prev_signal, "signal_barrier: tried to signal during an ongoing barrier");

    let worker_rank = self.worker_data.worker_rank();
    let num_workers = self.worker_data.num_workers();

    let dummy_buf: Vec<u8> = Vec::with_capacity(64);

    self.send_reqs.clear();
    for r in 0 .. num_workers {
      if r == worker_rank {
        continue;
      }
      let send_req = match self.server_conns[r].nonblocking_sync_send(&dummy_buf, 0, 2) {
        Ok(req) => req,
        Err(e) => panic!("async gossip: active thread: failed to do initial send: {:?}", e),
      };
      self.send_reqs.append(send_req);
    }
    self.send_reqs.wait_all();*/
  }

  fn wait_barrier(&mut self) -> bool {
    if self.rank0_signal {
      self.rank0_signal = false;
      return true;
    }
    match MpiComm::world().nonblocking_probe(Some(0), Some(0x22)) {
      Err(e) => panic!("elastic server: central: failed to do first recv: {:?}", e),
      Ok(None) => {
        false
      }
      Ok(Some(status)) => {
        true
      }
    }

    /*let bar_signal = self.bar_signal.load(Ordering::Acquire);
    if !bar_signal {
      false
    } else {
      Mpi::barrier_().unwrap();
      self.bar_signal.store(false, Ordering::Release);
      true
    }*/
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }
    let ctx = &(*self.context).as_ref();
    let data_len = data.len();
    let data = data.as_view(ctx).data;
    //data.raw_send(
    data.send(
        &mut self.origin_buf.as_ref_mut_range(offset, offset + data_len, ctx),
    );
  }

  fn complete_load(&mut self) {
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }

  fn communicate_first(&mut self) {
    if self.worker_data.worker_rank() == 0 {
      assert!(!self.rank0_init);
      let ctx = &(*self.context).as_ref();
      self.origin_buf.as_ref(ctx).sync_store(&mut self.origin_buf_h);
      self.send_reqs.clear();
      let send_req = match MpiComm::world().nonblocking_sync_send(&self.origin_buf_h[ .. self.msg_len], 0, 0x31) {
        Ok(req) => req,
        Err(e) => panic!("elastic server: active thread: failed to do nonblocking send: {:?}", e),
      };
      self.send_reqs.append(send_req);
      self.send_reqs.wait_all();
      for msg in 1 .. self.num_buf_msgs {
        let send_req = match MpiComm::world().nonblocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], 0, 0x31) {
          Ok(req) => req,
          Err(e) => panic!("elastic server: active thread: failed to do nonblocking send: {:?}", e),
        };
        self.send_reqs.append(send_req);
      }
      self.send_reqs.wait_all();
      self.rank0_init = true;
    }
    Mpi::barrier_().unwrap();
  }

  fn communicate(&mut self/*, ctx: &DeviceCtxRef*/) {
    let self_rank = self.worker_data.worker_rank();
    //let send_rank = self.ranks_range.ind_sample(&mut self.local_rng);

    if self_rank == 0 {
      assert!(self.rank0_init);
    }

    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let ctx = &(*self.context).as_ref();

    //self.origin_buf.sync_store(&mut self.origin_buf_h, ctx);
    self.origin_buf.as_ref(ctx).sync_store(&mut self.origin_buf_h);

    //println!("DEBUG: elastic server: active thread ({}): round: {} initial send", self_rank, self.iter_counter);
    self.send_reqs.clear();
    let send_req = match MpiComm::world().nonblocking_sync_send(&self.origin_buf_h[ .. self.msg_len], 0, 1) {
      Ok(req) => req,
      Err(e) => panic!("elastic server: active thread: failed to do initial send: {:?}", e),
    };
    self.send_reqs.append(send_req);
    self.send_reqs.wait_all();
    //println!("DEBUG: elastic server: active thread ({}): round: {} remaining sends", self_rank, self.iter_counter);
    for msg in 1 .. self.num_buf_msgs {
      let send_req = match MpiComm::world().nonblocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], 0, 0) {
        Ok(req) => req,
        Err(e) => panic!("elastic server: active thread: failed to do nonblocking send: {:?}", e),
      };
      self.send_reqs.append(send_req);
    }
    self.send_reqs.wait_all();

    for msg in 0 .. self.num_buf_msgs {
      let recv_req = match MpiComm::world().nonblocking_recv(&mut self.target_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], Some(0), Some(0x10)) {
        Ok(req) => req,
        Err(e) => panic!("elastic server: active thread: failed to do nonblocking recv: {:?}", e),
      };
      self.send_reqs.append(recv_req);
    }
    self.send_reqs.wait_all();

    let beta = 0.8 / self.worker_data.num_workers() as f32;
    self.target_buf.as_ref_mut(ctx).sync_load(&self.target_buf_h);
    self.origin_buf.as_ref(ctx).send(&mut self.final_buf.as_ref_mut(ctx));
    self.final_buf.as_ref_mut(ctx).row_vector_scale(1.0 - beta);
    self.final_buf.as_ref_mut(ctx).row_vector_sum(beta, &self.target_buf.as_ref(ctx));
    ctx.sync();
  }

  fn communicate_exact(&mut self) {
    let ctx = &(*self.context).as_ref();
    //self.origin_buf.as_ref(ctx).sync_store(&mut self.target_buf_h);

    let mut dummy_buf: Vec<u8> = Vec::with_capacity(64);

    //println!("DEBUG: ");
    let mut recv_req = match MpiComm::world().nonblocking_recv(&mut dummy_buf[ .. 0], Some(0), Some(0x22)) {
      Ok(req) => req,
      Err(e) => panic!(),
    };
    recv_req.wait().unwrap();

    if self.worker_data.worker_rank() == 0 {
      self.send_reqs.clear();
      for msg in 0 .. self.num_buf_msgs {
        let recv_req = match MpiComm::world().nonblocking_recv(&mut self.target_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], Some(0), Some(0x20)) {
          Ok(req) => req,
          Err(e) => panic!("elastic server: active thread: failed to do nonblocking recv: {:?}", e),
        };
        self.send_reqs.append(recv_req);
      }
      self.send_reqs.wait_all();
    }
    Mpi::barrier_().unwrap();

    self.send_reqs.clear();
    for msg in 0 .. self.num_buf_msgs {
      // FIXME(20160428): can't directly do bcast here because rank 0 has both
      // client and server, instead just let server send to just rank 0 and then
      // do the bcast.
      //unimplemented!();
      let bcast_req = match Mpi::nonblocking_broadcast_(
          &mut self.target_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len],
          0)
      {
        Ok(req) => req,
        Err(e) => panic!(),
      };
      self.send_reqs.append(bcast_req);
    }
    self.send_reqs.wait_all().unwrap();
    Mpi::barrier_().unwrap();

    self.final_buf.as_ref_mut(ctx).sync_load(&self.target_buf_h);
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
    //data.raw_recv(
    data.recv(
        &self.final_buf.as_ref_range(offset, offset + data_len, ctx),
    );
  }

  fn complete_store(&mut self) {
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }
}
