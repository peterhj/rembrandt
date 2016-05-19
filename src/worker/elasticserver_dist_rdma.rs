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

enum ElasticServerAct2PassMsg {
  Quit,
  Pause,
  Resume,
}

enum ElasticServerPass2ActMsg {
  AckPause,
}

enum ElasticServerPassiveState {
  Ready,
  Receiving{target_rank: usize},
  DoneReceiving{target_rank: usize},
  //Broadcast,
}

impl ElasticServerPassiveState {
  pub fn from_raw(x: u64) -> ElasticServerPassiveState {
    let state = (x & 0xffff_ffff) as u32;
    let target_rank = ((x >> 32) & 0xffff_ffff) as u32;
    match state {
      0 => ElasticServerPassiveState::Ready,
      1 => ElasticServerPassiveState::Receiving{target_rank: target_rank as usize},
      2 => ElasticServerPassiveState::DoneReceiving{target_rank: target_rank as usize},
      //3 => ElasticServerPassiveState::Broadcast,
      _ => unreachable!(),
    }
  }

  pub fn into_raw(&self) -> u64 {
    let (state, target_rank) = match self {
      &ElasticServerPassiveState::Ready                       => (0, 0),
      &ElasticServerPassiveState::Receiving{target_rank}      => (1, target_rank as u32),
      &ElasticServerPassiveState::DoneReceiving{target_rank}  => (2, target_rank as u32),
      //&ElasticServerPassiveState::Broadcast                   => (3, 0),
    };
    let x = state | ((target_rank as u64) << 32);
    x
  }
}

struct MpiDistElasticServerPassiveWorker {
  worker_rank:  usize,
  num_workers:  usize,
  buf_len:      usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  //num_buf_msgs: usize,
  //rdma_buf_len: usize,
  //state:        ElasticServerPassiveState,
  context:      DeviceContext,
  req_list:     MpiRequestList,
  //state_target: MpiOwnedWindow<u64, MpiMemory<u64>>,
  state_target: Arc<Mutex<MpiOwnedWindow<u64, MpiMemory<u64>>>>,
  state_origin: MpiMemory<u64>,
  center_target:    Arc<Mutex<MpiOwnedWindow<f32, MpiMemory<f32>>>>,
  //dst_target:       Arc<Mutex<MpiOwnedWindow<f32, MpiMemory<f32>>>>,
  center_origin:    MpiMemory<f32>,
  //center_buf:       Arc<Mutex<DeviceBuffer<f32>>>,
  center_buf:       Arc<Mutex<RawDeviceBuffer<f32>>>,
  local_target: MpiOwnedWindow<f32, MpiMemory<f32>>,
  local_origin: MpiMemory<f32>,
  //local_buf:    DeviceBuffer<f32>,
  local_buf:    RawDeviceBuffer<f32>,
}

impl MpiDistElasticServerPassiveWorker {
  pub fn run(mut self) {
    let sleep_duration = Duration::new(0, 50_000);
    //let mut poll_ranks: Vec<_> = (0 .. self.num_workers).collect();

    'poll_loop: loop {
      {
        let mut state_target = self.state_target.lock().unwrap();
        state_target.lock(0, MpiWindowLockMode::Shared).unwrap();
        unsafe { state_target.get_(
            self.state_origin.as_mut().as_mut_ptr(),
            1, 0, 0,
        ).unwrap() };
        //let rawstate = self.state_target.as_slice()[0]
        state_target.unlock(0).unwrap();
        //ElasticServerPassiveState::from_raw(rawstate)
      }
      let state = ElasticServerPassiveState::from_raw(self.state_origin.as_ref()[0]);

      match state {
        ElasticServerPassiveState::Ready => {}

        ElasticServerPassiveState::Receiving{target_rank} => {}

        ElasticServerPassiveState::DoneReceiving{target_rank} => {
          let ctx = &self.context.as_ref();

          /*{
            let mut dst_target = self.dst_target.lock().unwrap();
            dst_target.lock(target_rank, MpiWindowLockMode::Exclusive).unwrap();
            for msg in 0 .. self.num_msgs {
              dst_target.put_(
                  self.center_origin.as_ref().as_ptr(),
                  min(self.msg_len, self.buf_len - msg * self.msg_len),
                  target_rank,
                  msg * self.msg_len,
              ).unwrap();
            }
            dst_target.unlock(target_rank);
          }*/

          {
            let mut center_buf = self.center_buf.lock().unwrap();
            //center_buf.as_ref_mut(ctx).sync_load(self.center_origin.as_ref());
            //self.local_buf.as_ref_mut(ctx).sync_load(self.local_origin.as_ref());
            center_buf.sync_load(self.center_origin.as_ref(), ctx);
            self.local_buf.sync_load(self.local_origin.as_ref(), ctx);
            //center_buf.as_ref_mut(ctx).row_vector_scale(1.0 - 0.8 / self.num_workers as f32);
            //center_buf.as_ref_mut(ctx).row_vector_sum(0.8 / self.num_workers as f32, &self.local_buf.as_ref(ctx));
            center_buf.as_ref().async_vector_scale(1.0 - 0.8 / self.num_workers as f32, ctx);
            center_buf.as_ref().async_vector_add(0.8 / self.num_workers as f32, &self.local_buf.as_ref(), ctx);
            ctx.blocking_sync();
          }

          self.state_origin.as_mut()[0] = ElasticServerPassiveState::Ready.into_raw();
          let mut state_target = self.state_target.lock().unwrap();
          state_target.lock(0, MpiWindowLockMode::Exclusive).unwrap();
          unsafe { state_target.put_(
              self.state_origin.as_ref().as_ptr(),
              1, 0, 0,
          ).unwrap() };
          state_target.unlock(0).unwrap();
        }
      }

      sleep(sleep_duration);
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
  /*rdma_buf_len: usize,
  num_rdma_buf_msgs:    usize,
  num_rounds:   usize,*/
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
  //push_target_win:    Arc<Mutex<MpiOwnedWindow<f32, MpiMemory<f32>>>>,
  pull_target_win:    MpiOwnedWindow<f32, MpiMemory<f32>>,
  pull_origin_buf_h:  MpiMemory<f32>,
  //pull_target_win:    MpiOwnedWindow<f32, MpiMemory<f32>>,
  //pull_origin_buf_h:  MpiMemory<f32>,
  reduce_buf_h: Vec<f32>,
  target_buf:   DeviceBuffer<f32>,
  dst_buf:      DeviceBuffer<f32>,
  //center_target:    Option<MpiOwnedWindow<f32, MpiMemory<f32>>>,
  center_origin:    MpiMemory<f32>,
  center_target:    Arc<Mutex<MpiOwnedWindow<f32, MpiMemory<f32>>>>,
  center_buf:       Option<Arc<Mutex<RawDeviceBuffer<f32>>>>,
  active_center_buf:    DeviceBuffer<f32>,
  local_origin:     MpiMemory<f32>,
  local_target:     Arc<Mutex<MpiOwnedWindow<f32, MpiMemory<f32>>>>,
  //local_buf:        Option<Arc<Mutex<RawDeviceBuffer<f32>>>>,
  //state_target:     Option<MpiOwnedWindow<u64, MpiMemory<u64>>>,
  state_target:     Arc<Mutex<MpiOwnedWindow<u64, MpiMemory<u64>>>>,
  state_origin:     MpiMemory<u64>,
  state_compare:    MpiMemory<u64>,
  state_result:     MpiMemory<u64>,
  checkpt_sig:  bool,

  center_thr:   Option<JoinHandle<()>>,
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

impl MpiDistElasticServerCommWorker {
  pub fn new(msg_len: usize, gossip_cfg: GossipConfig, context: Rc<DeviceContext>) -> MpiDistElasticServerCommWorker {
    // XXX(20160415): Empirically determined message length.
    /*//let msg_len = 32 * 1024;
    let msg_len = 1 * 1024 * 1024;
    //let msg_len = 4 * 1024 * 1024;*/
    let num_buf_msgs = (gossip_cfg.buf_size + msg_len - 1) / msg_len;
    let buf_len = gossip_cfg.buf_size; //num_buf_msgs * msg_len;
    // XXX(20160501): For RDMA specifically, the buffer length is increased
    // by one for a normalization value.
    /*let num_rdma_buf_msgs = (gossip_cfg.buf_size + 1 + msg_len - 1) / msg_len;
    let rdma_buf_len = gossip_cfg.buf_size + 1; //num_rdma_buf_msgs * msg_len;
    assert!(gossip_cfg.num_rounds >= 1);*/

    let mpi = Mpi::new_serialized();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let ctx = &(*context).as_ref();
    let dev_idx = ctx.device();

    let src_buf = DeviceBuffer::zeros(buf_len, ctx);
    let target_buf = DeviceBuffer::zeros(buf_len, ctx);
    let dst_buf = DeviceBuffer::zeros(buf_len, ctx);

    let pull_target_buf_h = MpiMemory::alloc_(buf_len).unwrap();
    //let pull_target_win = Arc::new(Mutex::new(MpiOwnedWindow::create_(pull_target_buf_h).unwrap()));
    let pull_target_win = MpiOwnedWindow::create_(pull_target_buf_h).unwrap();
    let pull_origin_buf_h = MpiMemory::alloc_(buf_len).unwrap();
    let mut reduce_buf_h = Vec::with_capacity(buf_len);
    for _ in 0 .. buf_len {
      reduce_buf_h.push(0.0);
    }

    //let center_target = MpiOwnedWindow<f32, MpiMemory<f32>>;
    let center_target_buf_h = MpiMemory::alloc_(buf_len).unwrap();
    let center_target = MpiOwnedWindow::create_(center_target_buf_h).unwrap();
    let center_target = Arc::new(Mutex::new(center_target));
    /*let (active_center_target, passive_center_target) = if worker_rank == 0 {
      (None, Some(center_target))
    } else {
      (Some(center_target), None)
    };*/
    let center_buf = if worker_rank == 0 {
      Some(Arc::new(Mutex::new(unsafe { RawDeviceBuffer::new(buf_len, ctx) })))
    } else {
      None
    };
    let active_center_buf = DeviceBuffer::zeros(buf_len, ctx);

    let local_target_buf_h = MpiMemory::alloc_(buf_len).unwrap();
    let local_target = MpiOwnedWindow::create_(local_target_buf_h).unwrap();
    let local_target = Arc::new(Mutex::new(local_target));

    let state_target = MpiOwnedWindow::create_(MpiMemory::alloc_(1).unwrap()).unwrap();
    let state_target = Arc::new(Mutex::new(state_target));
    /*let (active_state_target, passive_state_target) = if worker_rank == 0 {
      (None, Some(MpiOwnedWindow::create_(MpiMemory::alloc_(1).unwrap()).unwrap()))
    } else {
      (Some(MpiOwnedWindow::create_(MpiMemory::alloc_(1).unwrap()).unwrap()), None)
    };*/

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

    let center_thr = if worker_rank == 0 {
      let center_target = center_target.clone();
      let center_buf = center_buf.clone().unwrap();
      let state_target = state_target.clone();
      Some(spawn(move || {
        let context = DeviceContext::new(dev_idx);

        let center_origin = MpiMemory::alloc_(buf_len).unwrap();
        //let center_buf = DeviceBuffer::zeros(buf_len, &context.as_ref());

        let local_target_buf_h = MpiMemory::<f32>::alloc_(buf_len).unwrap();
        let local_target = MpiOwnedWindow::create_(local_target_buf_h).unwrap();
        let local_origin = MpiMemory::alloc_(buf_len).unwrap();
        //let local_buf = DeviceBuffer::zeros(buf_len, &context.as_ref());
        let local_buf = unsafe { RawDeviceBuffer::new(buf_len, &context.as_ref()) };

        //let pull_target_win = pull_target_win.clone();

        MpiDistElasticServerPassiveWorker{
  /*worker_rank:  usize,
  num_workers:  usize,
  buf_len:      usize,
  msg_len:      usize,
  num_msgs:     usize,
  //num_buf_msgs: usize,
  //rdma_buf_len: usize,
  //state:        ElasticServerPassiveState,
  context:      DeviceContext,
  req_list:     MpiRequestList,
  state_target: MpiOwnedWindow<u64, MpiMemory<u64>>,
  state_origin: MpiMemory<u64>,
  //center_target:    MpiOwnedWindow<f32, MpiMemory<f32>>,
  //dst_target:       Arc<Mutex<MpiOwnedWindow<f32, MpiMemory<f32>>>>,
  center_origin:    MpiMemory<f32>,
  center_buf:       Arc<Mutex<DeviceBuffer<f32>>>,
  local_target: MpiOwnedWindow<f32, MpiMemory<f32>>,
  local_origin: MpiMemory<f32>,
  local_buf:    DeviceBuffer<f32>,*/
          worker_rank:  worker_rank,
          num_workers:  num_workers,
          buf_len:      buf_len,
          msg_len:      msg_len,
          num_buf_msgs: num_buf_msgs,
          //state:        AsyncPushGossipPassiveState::Receiving,
          context:      context,
          req_list:     MpiRequestList::new(),
          //state_target: MpiOwnedWindow::create_(MpiMemory::alloc_(1).unwrap()).unwrap(),
          //state_target: passive_state_target.unwrap(),
          state_target: state_target,
          state_origin: MpiMemory::alloc_(1).unwrap(),
          //dst_target:   push_target_win,
          center_target:    center_target,
          center_origin:    center_origin,
          center_buf:       center_buf,
          local_target: local_target,
          local_origin: local_origin,
          local_buf:    local_buf,
        }.run();
      }))
    } else {
      None
    };

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
      /*rdma_buf_len: rdma_buf_len,
      num_rdma_buf_msgs:    num_rdma_buf_msgs,
      num_rounds:   gossip_cfg.num_rounds,*/
      src_buf:      src_buf,
      pull_target_win:    pull_target_win,
      pull_origin_buf_h:  pull_origin_buf_h,
      reduce_buf_h: reduce_buf_h,
      target_buf:   target_buf,
      dst_buf:      dst_buf,
      center_origin:    MpiMemory::alloc_(buf_len).unwrap(),
      //center_target:    active_center_target,
      center_target:    center_target,
      center_buf:       center_buf,
      active_center_buf:    active_center_buf,
      local_origin:     MpiMemory::alloc_(buf_len).unwrap(),
      local_target:     local_target,
      //local_buf:        DeviceBuffer::zeros(buf_len, ctx),
      //state_target:     active_state_target,
      state_target:     state_target,
      state_origin:     MpiMemory::alloc_(1).unwrap(),
      state_compare:    MpiMemory::alloc_(1).unwrap(),
      state_result:     MpiMemory::alloc_(1).unwrap(),
      checkpt_sig:  false,
      send_reqs:    MpiRequestList::new(),
      center_thr:   center_thr,
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

impl MpiDistCommWorker for MpiDistElasticServerCommWorker {
  fn mpi(&self) -> &Mpi {
    &self.mpi
  }
}

impl CommWorker for MpiDistElasticServerCommWorker {
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

  fn communicate(&mut self, _repeat: bool /*, ctx: &DeviceCtxRef*/) {
    let sleep_duration = Duration::new(0, 50_000);
    let self_rank = self.worker_data.worker_rank();
    let num_workers = self.worker_data.num_workers();

    let ctx = &(*self.context).as_ref();
    self.pull_target_win.lock(self_rank, MpiWindowLockMode::Exclusive).unwrap();
    self.src_buf.as_ref_mut(ctx).sync_store(self.pull_target_win.as_mut_slice());
    self.pull_target_win.unlock(self_rank).unwrap();

    let ready_rawstate = ElasticServerPassiveState::Ready.into_raw();
    let recv_rawstate = ElasticServerPassiveState::Receiving{target_rank: self_rank}.into_raw();
    let done_rawstate = ElasticServerPassiveState::DoneReceiving{target_rank: self_rank}.into_raw();

    loop {
      self.state_origin.as_mut()[0] = recv_rawstate;
      self.state_compare.as_mut()[0] = ready_rawstate;
      self.state_result.as_mut()[0] = 0xffff_ffff_ffff_ffff;
      {
        let mut state_target = self.state_target.lock().unwrap();
        state_target.lock(0, MpiWindowLockMode::Exclusive).unwrap();
        unsafe { state_target.compare_and_swap_(
            self.state_origin.as_ref().as_ptr(),
            self.state_compare.as_ref().as_ptr(),
            self.state_result.as_mut().as_mut_ptr(),
            0, 0,
        ).unwrap() };
        state_target.unlock(0).unwrap();
      }
      if self.state_result.as_ref()[0] == ready_rawstate {
        break;
      }
      sleep(sleep_duration);
    }

    {
      let mut local_target = self.local_target.lock().unwrap();
      local_target.lock(0, MpiWindowLockMode::Exclusive).unwrap();
      for msg in 0 .. self.num_buf_msgs {
        unsafe { local_target.put_(
            self.local_origin.as_ref().as_ptr().offset((msg * self.msg_len) as isize),
            min(self.msg_len, self.buf_len - msg * self.msg_len),
            0,
            msg * self.msg_len,
        ).unwrap() };
      }
      local_target.unlock(0);
    }

    {
      let mut center_target = self.center_target.lock().unwrap();
      center_target.lock(0, MpiWindowLockMode::Shared).unwrap();
      for msg in 0 .. self.num_buf_msgs {
        unsafe { center_target.get_(
            self.center_origin.as_mut().as_mut_ptr().offset((msg * self.msg_len) as isize),
            min(self.msg_len, self.buf_len - msg * self.msg_len),
            0,
            msg * self.msg_len,
        ).unwrap() };
      }
      center_target.unlock(0);
    }

    self.state_origin.as_mut()[0] = done_rawstate;
    self.state_compare.as_mut()[0] = recv_rawstate;
    self.state_result.as_mut()[0] = 0xffff_ffff_ffff_ffff;
    {
      let mut state_target = self.state_target.lock().unwrap();
      state_target.lock(0, MpiWindowLockMode::Exclusive).unwrap();
      unsafe { state_target.compare_and_swap_(
          self.state_origin.as_ref().as_ptr(),
          self.state_compare.as_ref().as_ptr(),
          self.state_result.as_mut().as_mut_ptr(),
          0, 0,
      ).unwrap() };
      state_target.unlock(0).unwrap();
    }
    assert_eq!(self.state_result.as_ref()[0], recv_rawstate);

    self.active_center_buf.as_ref_mut(ctx).sync_load(self.center_origin.as_ref());
    self.dst_buf.as_ref_mut(ctx).row_vector_scale(1.0 - 0.8 / num_workers as f32);
    self.dst_buf.as_ref_mut(ctx).row_vector_sum(0.8 / num_workers as f32, &self.active_center_buf.as_ref(ctx));

    /*if self_rank != recv_rank {
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
    }*/

    ctx.sync();
  }

  fn communicate_exact(&mut self) {
    let ctx = &(*self.context).as_ref();
    //self.src_buf.as_ref(ctx).sync_store(self.pull_origin_buf_h.as_mut());
    if self.worker_data.worker_rank() == 0 {
      let mut center_buf = self.center_buf.as_mut().unwrap().lock().unwrap();
      center_buf.sync_store(self.center_origin.as_mut(), ctx);
    }
    //ctx.sync();
    self.send_reqs.clear();
    for msg in 0 .. self.num_buf_msgs {
      let req = MpiComm::world().nonblocking_broadcast_(
          &mut self.center_origin.as_mut()[msg * self.msg_len .. min(self.buf_len, (msg+1) * self.msg_len)],
          0,
      ).unwrap();
      self.send_reqs.append(req);
    }
    self.send_reqs.wait_all().unwrap();
    //Mpi::barrier_().unwrap();
    self.dst_buf.as_ref_mut(ctx).sync_load(self.center_origin.as_ref());
    ctx.blocking_sync();
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
