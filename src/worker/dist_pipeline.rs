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
  CommWorkerBuilder, CommWorker,
  GossipConfig,
};
use operator::loss::{
  LossOperator,
  CategoricalLossConfig,
};
use operator::worker::{
  OperatorWorkerBuilder,
  OperatorWorker,
  PipelineOperatorConfig,
};

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::comm::{ReduceOperation, AverageReduceOperation, for_all_devices};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::memory::{RawDeviceBuffer};
use array_new::{AsyncArray};
use rng::xorshift::{Xorshiftplus128Rng};
use worker_::{WorkerData};

use mpi::{Mpi, MpiComm, MpiGroup, MpiWindow, MpiWindowLockMode, MpiRequest, MpiRequestList};
//use procgroup::{ProcGroup};
use threadpool::{ThreadPool};

use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell};
use std::collections::{HashSet};
use std::ffi::{CString};
use std::io::{Write};
use std::iter::{FromIterator, repeat};
use std::marker::{PhantomData};
use std::rc::{Rc};
use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::thread::{JoinHandle, sleep, spawn};
use std::time::{Duration};
use vec_map::{VecMap};

/*#[derive(Clone)]
pub struct MpiDistOldSyncGossipCommWorkerBuilder {
  num_workers:  usize,
  num_rounds:   usize,
  period:       usize,
  //barrier:      Arc<Barrier>,
  shared_bufs:  Vec<Arc<RawDeviceBuffer<f32>>>,
  //shared_rns:   Arc<Vec<AtomicUsize>>,
  shared_seed:  [u64; 2],
}

impl MpiDistOldSyncGossipCommWorkerBuilder {
  pub fn new(config: GossipConfig, /*contexts: &[DeviceContext]*/) -> MpiDistOldSyncGossipCommWorkerBuilder {
    // FIXME(20160412)
    unimplemented!();
  }
}*/

pub struct MpiDistDownpourServerCommWorker;

pub struct MpiDistElasticServerCommWorker;

pub struct MpiDistOldSyncGossipCommWorker {
  worker_data:  WorkerData,
  context:      Rc<DeviceContext>,
  mpi:          Mpi,

  buf_len:      usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  com_interval: usize,

  world_group:  MpiGroup,
  solo_groups:  Vec<MpiGroup>,
  pair_groups:  Vec<MpiGroup>,

  origin_buf:   RawDeviceBuffer<f32>,
  target_buf:   RawDeviceBuffer<f32>,
  origin_buf_h: Vec<f32>,
  target_buf_h: Vec<f32>,
  target_win_h: MpiWindow<f32>,
  avg_reduce:   AverageReduceOperation<f32>,

  rng:          Xorshiftplus128Rng,
  ranks_perm:   Vec<usize>,
  iter_counter: usize,
}

impl MpiDistOldSyncGossipCommWorker {
  pub fn new(gossip_cfg: GossipConfig, context: Rc<DeviceContext>) -> MpiDistOldSyncGossipCommWorker {
    // XXX(20160415): Empirically determined message length.
    let msg_len = 32 * 1024;
    let num_buf_msgs = (gossip_cfg.buf_size + msg_len - 1) / msg_len;
    let buf_len = num_buf_msgs * msg_len;

    let ctx = &(*context).as_ref();
    let origin_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    let target_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };

    let mpi = Mpi::new();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let mut origin_buf_h = Vec::with_capacity(buf_len);
    unsafe { origin_buf_h.set_len(buf_len) };
    let mut target_buf_h = Vec::with_capacity(buf_len);
    unsafe { target_buf_h.set_len(buf_len) };
    let target_win_h = match unsafe { MpiWindow::create(target_buf_h.as_mut_ptr(), target_buf_h.len(), &mpi) } {
      Ok(win) => win,
      Err(e) => panic!("comm worker: failed to create MPI window"),
    };

    let world_group = mpi.world_group();
    let mut solo_groups = Vec::with_capacity(num_workers);
    for r in 0 .. num_workers {
      solo_groups.push(world_group.ranges(&[(r, r+1, 1)]));
    }
    let mut pair_groups = Vec::with_capacity(num_workers);
    for r in 0 .. num_workers {
      if r != worker_rank {
        pair_groups.push(world_group.ranges(&[(worker_rank, worker_rank+1, 1), (r, r+1, 1)]));
      } else {
        pair_groups.push(world_group.ranges(&[(worker_rank, worker_rank+1, 1)]));
      }
    }

    MpiDistOldSyncGossipCommWorker{
      worker_data:  WorkerData::new(worker_rank, num_workers),
      context:      context.clone(),
      mpi:          Mpi,
      buf_len:      buf_len,
      msg_len:      msg_len,
      num_buf_msgs: num_buf_msgs,
      com_interval: 1,
      world_group:  world_group,
      solo_groups:  solo_groups,
      pair_groups:  pair_groups,
      origin_buf:   origin_buf,
      target_buf:   target_buf,
      origin_buf_h: origin_buf_h,
      target_buf_h: target_buf_h,
      target_win_h: target_win_h,
      avg_reduce:   AverageReduceOperation::new(0),
      rng:          Xorshiftplus128Rng::new(&mut thread_rng()),
      ranks_perm:   (0 .. num_workers).collect(),
      iter_counter: 0,
    }
  }
}

impl CommWorker for MpiDistOldSyncGossipCommWorker {
  fn next(&mut self) -> bool {
    // FIXME(20160412)
    self.iter_counter += 1;
    /*if self.worker_data.worker_rank() == 0 {
      println!("DEBUG: next: {}", self.iter_counter);
    }*/
    true
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let data = data.as_view(ctx).data;
    data.raw_send(
        //&self.target_buf.as_ref_range(offset, offset + data.len()),
        &self.origin_buf.as_ref_range(offset, offset + data.len()),
    );
  }

  fn communicate(&mut self, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    // FIXME(20160412): steps for distributed gossip:
    // - first, sync on the previous loads (this can happen in the caller)
    // - optionally, wait on a world barrier
    // - pick a target according to a permutation
    // - call a fence-protected RMA get
    // - wait on a final world barrier

    self.rng.shuffle(&mut self.ranks_perm);
    let self_rank = self.worker_data.worker_rank();
    let send_rank = self.ranks_perm[self_rank];
    let mut recv_rank = self_rank;
    for r in 0 .. self.worker_data.num_workers() {
      if self.ranks_perm[r] == self_rank {
        recv_rank = r;
        break;
      }
    }

    if self_rank == send_rank || self_rank == recv_rank {
      assert_eq!(self_rank, send_rank);
      assert_eq!(self_rank, recv_rank);
      //self.target_buf.raw_send(&self.origin_buf, ctx);
      self.origin_buf.raw_send(&self.target_buf, ctx);
      return;
    }

    //self.target_buf.sync_store(&mut self.target_buf_h, ctx);
    self.origin_buf.sync_store(&mut self.origin_buf_h, ctx);

    // FIXME(20160415): getting communication to work.
    // FIXME(20160416): nesting the start-complete-post-wait primitives lead to
    // deadlock.
    self.target_win_h.fence();
    //self.target_win_h.post(&self.solo_groups[recv_rank]);
    //self.target_win_h.start(&self.solo_groups[send_rank]);
    for msg in 0 .. self.num_buf_msgs {
      let offset = msg * self.msg_len;
      //match unsafe { self.target_win_h.rma_get(self.origin_buf_h.as_mut_ptr().offset(offset as isize), self.msg_len, send_rank, offset, &self.mpi) } {
      match unsafe { self.target_win_h.rma_put(self.origin_buf_h.as_ptr().offset(offset as isize), self.msg_len, send_rank, offset, &self.mpi) } {
        Ok(_) => {}
        Err(e) => panic!("mpi dist comm worker: failed to call rma_get: {:?}", e),
      }
    }
    self.target_win_h.fence();
    //self.target_win_h.complete();
    //self.target_win_h.wait();

    // FIXME(20160415): load from the correct buffer.
    //self.origin_buf.sync_load(&self.origin_buf_h, ctx);
    self.target_buf.sync_load(&self.target_buf_h, ctx);
    self.avg_reduce.reduce(
        /*&(self.target_buf).as_ref(),
        &(self.origin_buf).as_ref(),*/
        &(self.origin_buf).as_ref(),
        &(self.target_buf).as_ref(),
        ctx,
    );
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let mut data = data.as_view_mut(ctx).data;
    let data_len = data.len();
    data.raw_recv(
        //&self.origin_buf.as_ref_range(offset, offset + data_len),
        &self.target_buf.as_ref_range(offset, offset + data_len),
    );
  }
}

enum OldAsyncGossipAct2PassMsg {
  Quit,
  StartRound,
}

enum OldAsyncGossipPass2ActMsg {
  DoneRound,
}

pub struct MpiDistOldAsyncGossipCommWorker {
  worker_data:  WorkerData,
  context:      Rc<DeviceContext>,
  mpi:          Mpi,

  buf_len:      usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  com_interval: usize,

  /*world_group:  MpiGroup,
  solo_groups:  Vec<MpiGroup>,
  pair_groups:  Vec<MpiGroup>,*/

  origin_buf:   RawDeviceBuffer<f32>,
  target_buf:   RawDeviceBuffer<f32>,
  origin_buf_h: Vec<f32>,
  //target_buf_h: Vec<f32>,
  target_buf_h: Arc<Mutex<Vec<f32>>>,
  target_win_h: MpiWindow<f32>,

  act2pass_tx:  Sender<OldAsyncGossipAct2PassMsg>,
  pass2act_rx:  Receiver<OldAsyncGossipPass2ActMsg>,
  passive_thr:  JoinHandle<()>,
  send_reqs:    MpiRequestList,

  avg_reduce:   AverageReduceOperation<f32>,

  rng:          Xorshiftplus128Rng,
  ranks_perm:   Vec<usize>,
  iter_counter: usize,
}

impl MpiDistOldAsyncGossipCommWorker {
  pub fn new(gossip_cfg: GossipConfig, context: Rc<DeviceContext>) -> MpiDistOldAsyncGossipCommWorker {
    // XXX(20160415): Empirically determined message length.
    let msg_len = 32 * 1024;
    let num_buf_msgs = (gossip_cfg.buf_size + msg_len - 1) / msg_len;
    let buf_len = num_buf_msgs * msg_len;

    let ctx = &(*context).as_ref();
    let origin_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    let target_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };

    let mpi = Mpi::new();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    /*let world_group = mpi.world_group();
    let mut solo_groups = Vec::with_capacity(num_workers);
    for r in 0 .. num_workers {
      solo_groups.push(world_group.ranges(&[(r, r+1, 1)]));
    }
    let mut pair_groups = Vec::with_capacity(num_workers);
    for r in 0 .. num_workers {
      if r != worker_rank {
        pair_groups.push(world_group.ranges(&[(worker_rank, worker_rank+1, 1), (r, r+1, 1)]));
      } else {
        pair_groups.push(world_group.ranges(&[(worker_rank, worker_rank+1, 1)]));
      }
    }*/

    let mut origin_buf_h = Vec::with_capacity(buf_len);
    unsafe { origin_buf_h.set_len(buf_len) };
    let mut target_buf_h = Vec::with_capacity(buf_len);
    unsafe { target_buf_h.set_len(buf_len) };
    let target_win_h = match unsafe { MpiWindow::create(target_buf_h.as_mut_ptr(), target_buf_h.len(), &mpi) } {
      Ok(win) => win,
      Err(e) => panic!("comm worker: failed to create MPI window"),
    };
    let target_buf_h = Arc::new(Mutex::new(target_buf_h));

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
    let passive_thr = {
      let target_buf_h = target_buf_h.clone();
      spawn(move || {
        let msg_len = msg_len;
        let num_buf_msgs = num_buf_msgs;
        let mut recv_reqs = MpiRequestList::new();
        loop {
          println!("DEBUG: async gossip: passive thread ({}): waiting for round start", worker_rank);
          match act2pass_rx.recv() {
            Err(_) => {
              println!("DEBUG: async gossip: passive thread ({}): loop terminated early", worker_rank);
              break;
            }
            Ok(OldAsyncGossipAct2PassMsg::Quit) => {
              break;
            }
            Ok(OldAsyncGossipAct2PassMsg::StartRound) => {
              println!("DEBUG: async gossip: passive thread ({}): starting gossip round", worker_rank);
              {
                let mut target_buf_h = target_buf_h.lock().unwrap();
                //println!("DEBUG: async gossip: passive thread ({}): first recv", worker_rank);
                let status = match Mpi::blocking_recv(&mut target_buf_h[ .. msg_len], None) {
                  Ok(status) => status,
                  Err(e) => panic!("async gossip: passive thread: failed to do first recv: {:?}", e),
                };
                let recv_rank = status.src_rank;
                println!("DEBUG: async gossip: passive thread ({}): recv rank: {}", worker_rank, recv_rank);
                recv_reqs.clear();
                println!("DEBUG: async gossip: passive thread ({}): start recv", worker_rank);
                for msg in 1 .. num_buf_msgs {
                  //Mpi::blocking_recv(&mut target_buf_h[msg * msg_len .. (msg+1) * msg_len], Some(recv_rank)).unwrap();
                  //println!("DEBUG: async gossip: passive thread ({}): did recv {}/{}", worker_rank, msg, num_buf_msgs);
                  let recv_req = match MpiRequest::nonblocking_recv(&mut target_buf_h[msg * msg_len .. (msg+1) * msg_len], Some(recv_rank), None) {
                    Ok(req) => req,
                    Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
                  };
                  recv_reqs.append(recv_req);
                }
                println!("DEBUG: async gossip: passive thread ({}): finish recv", worker_rank);
              }
              recv_reqs.wait_all();
              pass2act_tx.send(OldAsyncGossipPass2ActMsg::DoneRound).unwrap();
            }
          }
        }
      })
    };

    MpiDistOldAsyncGossipCommWorker{
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
      target_buf:   target_buf,
      origin_buf_h: origin_buf_h,
      target_buf_h: target_buf_h,
      target_win_h: target_win_h,
      act2pass_tx:  act2pass_tx,
      pass2act_rx:  pass2act_rx,
      passive_thr:  passive_thr,
      send_reqs:    MpiRequestList::new(),
      avg_reduce:   AverageReduceOperation::new(0),
      rng:          Xorshiftplus128Rng::new(&mut thread_rng()),
      ranks_perm:   (0 .. num_workers).collect(),
      iter_counter: 0,
    }
  }
}

impl CommWorker for MpiDistOldAsyncGossipCommWorker {
  fn next(&mut self) -> bool {
    // FIXME(20160412)
    self.iter_counter += 1;
    /*if self.worker_data.worker_rank() == 0 {
      println!("DEBUG: next: {}", self.iter_counter);
    }*/
    true
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let data = data.as_view(ctx).data;
    data.raw_send(
        //&self.target_buf.as_ref_range(offset, offset + data.len()),
        &self.origin_buf.as_ref_range(offset, offset + data.len()),
    );
  }

  fn communicate(&mut self, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    // FIXME(20160412): steps for distributed gossip:
    // - first, sync on the previous loads (this can happen in the caller)
    // - optionally, wait on a world barrier
    // - pick a target according to a permutation
    // - call a fence-protected RMA get
    // - wait on a final world barrier

    self.rng.shuffle(&mut self.ranks_perm);
    let self_rank = self.worker_data.worker_rank();
    let send_rank = self.ranks_perm[self_rank];

    /*let mut recv_rank = self_rank;
    for r in 0 .. self.worker_data.num_workers() {
      if self.ranks_perm[r] == self_rank {
        recv_rank = r;
        break;
      }
    }

    if self_rank == send_rank || self_rank == recv_rank {
      assert_eq!(self_rank, send_rank);
      assert_eq!(self_rank, recv_rank);
      //self.target_buf.raw_send(&self.origin_buf, ctx);
      self.origin_buf.raw_send(&self.target_buf, ctx);
      return;
    }*/

    /*if self_rank == send_rank {
      //self.target_buf.raw_send(&self.origin_buf, ctx);
      self.origin_buf.raw_send(&self.target_buf, ctx);
      return;
    }*/

    //self.target_buf.sync_store(&mut self.target_buf_h, ctx);
    self.origin_buf.sync_store(&mut self.origin_buf_h, ctx);

    // FIXME(20160415): getting communication to work.

    println!("DEBUG: async gossip: active thread ({}): starting gossip round", self_rank);
    self.act2pass_tx.send(OldAsyncGossipAct2PassMsg::StartRound).unwrap();

    println!("DEBUG: async gossip: active thread ({}): send rank: {}", self_rank, send_rank);
    //self.target_win_h.lock(send_rank, MpiWindowLockMode::Exclusive).unwrap();
    self.send_reqs.clear();
    for msg in 0 .. self.num_buf_msgs {
      //Mpi::blocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], send_rank).unwrap();
      //println!("DEBUG: async gossip: passive thread ({}): did send {}/{}", self_rank, msg, self.num_buf_msgs);
      let send_req = match MpiRequest::nonblocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], send_rank, None) {
        Ok(req) => req,
        Err(e) => panic!("async gossip: active thread: failed to do nonblocking send: {:?}", e),
      };
      self.send_reqs.append(send_req);
    }
    /*{
      let mut target_buf_h = self.target_buf_h.lock().unwrap();
      let status = match Mpi::blocking_recv(&mut target_buf_h[ .. self.msg_len], None) {
        Ok(status) => status,
        Err(e) => panic!("async gossip: passive thread: failed to do first recv: {:?}", e),
      };
      let recv_rank = status.src_rank;
      for msg in 1 .. self.num_buf_msgs {
        //Mpi::blocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], send_rank).unwrap();
        //println!("DEBUG: async gossip: passive thread ({}): did send {}/{}", self_rank, msg, self.num_buf_msgs);
        let send_req = match MpiRequest::nonblocking_recv(&mut target_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], Some(recv_rank)) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: active thread: failed to do nonblocking send: {:?}", e),
        };
        self.send_reqs.append(send_req);
      }
    }*/
    self.send_reqs.wait_all();
    //self.target_win_h.unlock(send_rank).unwrap();

    println!("DEBUG: async gossip: active thread ({}): waiting for passive response", self_rank);
    match self.pass2act_rx.recv() {
      Err(e) => {
        panic!("async gossip: active thread: failed to receive msg from passive thread: {:?}", e);
      }
      Ok(OldAsyncGossipPass2ActMsg::DoneRound) => {}
    }
    println!("DEBUG: async gossip: active thread ({}): got passive response", self_rank);

    // FIXME(20160416): this pattern is known to deadlock!
    /*let mut send_req = match MpiRequest::nonblocking_send(&self.origin_buf_h, send_rank) {
      Ok(req) => req,
      Err(e) => panic!("failed to do nonblocking send: {:?}", e),
    };
    let mut recv_req = match MpiRequest::nonblocking_recv(&mut self.target_buf_h, Some(recv_rank)) {
      Ok(req) => req,
      Err(e) => panic!("failed to do nonblocking recv: {:?}", e),
    };
    //self.mpi.barrier();
    send_req.wait();
    recv_req.wait();*/

    /*println!("DEBUG: rank: {} start send_recv", self_rank);
    self.mpi.send_recv(&self.origin_buf_h, send_rank, &mut self.target_buf_h, Some(recv_rank));
    println!("DEBUG: rank: {} completed send_recv", self_rank);
    self.mpi.barrier();*/

    /*// FIXME(20160416): nesting the start-complete-post-wait primitives lead to
    // deadlock.
    self.target_win_h.fence();
    //self.target_win_h.post(&self.solo_groups[recv_rank]);
    //self.target_win_h.start(&self.solo_groups[send_rank]);
    for msg in 0 .. self.num_buf_msgs {
      let offset = msg * self.msg_len;
      //match unsafe { self.target_win_h.rma_get(self.origin_buf_h.as_mut_ptr().offset(offset as isize), self.msg_len, send_rank, offset, &self.mpi) } {
      match unsafe { self.target_win_h.rma_put(self.origin_buf_h.as_ptr().offset(offset as isize), self.msg_len, send_rank, offset, &self.mpi) } {
        Ok(_) => {}
        Err(e) => panic!("mpi dist comm worker: failed to call rma_get: {:?}", e),
      }
    }
    self.target_win_h.fence();
    //self.target_win_h.complete();
    //self.target_win_h.wait();*/

    // FIXME(20160415): load from the correct buffer.
    //self.origin_buf.sync_load(&self.origin_buf_h, ctx);
    {
      println!("DEBUG: async gossip: active thread ({}): acquiring target buf lock", self_rank);
      let target_buf_h = self.target_buf_h.lock().unwrap();
      println!("DEBUG: async gossip: active thread ({}): lock acquired", self_rank);
      self.target_buf.sync_load(&target_buf_h, ctx);
    }
    self.avg_reduce.reduce(
        /*&(self.target_buf).as_ref(),
        &(self.origin_buf).as_ref(),*/
        &(self.origin_buf).as_ref(),
        &(self.target_buf).as_ref(),
        ctx,
    );
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let mut data = data.as_view_mut(ctx).data;
    let data_len = data.len();
    data.raw_recv(
        //&self.origin_buf.as_ref_range(offset, offset + data_len),
        &self.target_buf.as_ref_range(offset, offset + data_len),
    );
  }
}

enum SyncGossipAct2PassMsg {
  Quit,
  StartRound{clock: usize, recv_rank: usize},
}

enum SyncGossipPass2ActMsg {
  DoneRound{clock: usize, success: bool},
}

struct MpiDistSyncGossipPassiveWorker {
  worker_rank:  usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  target_buf_h: Arc<Mutex<Vec<f32>>>,
  act2pass_rx:  Receiver<SyncGossipAct2PassMsg>,
  pass2act_tx:  Sender<SyncGossipPass2ActMsg>,
  client_conns: VecMap<MpiComm>,
  recv_reqs:    MpiRequestList,
}

impl MpiDistSyncGossipPassiveWorker {
  pub fn run(mut self) {
    loop {
      match self.act2pass_rx.recv() {
        Err(_) => {
          println!("DEBUG: async gossip: passive thread ({}): loop terminated early", self.worker_rank);
          break;
        }
        Ok(SyncGossipAct2PassMsg::Quit) => {
          break;
        }
        Ok(SyncGossipAct2PassMsg::StartRound{clock, recv_rank}) => {
          //println!("DEBUG: async gossip: passive thread ({}): round: {} starting gossip round", worker_rank, clock);
          //println!("DEBUG: async gossip: passive thread ({}): first recv", worker_rank);
          //println!("DEBUG: async gossip: passive thread ({}): round: {} recv rank: {}", worker_rank, clock, recv_rank);
          //println!("DEBUG: async gossip: passive thread ({}): round: {} acquire lock for recv", worker_rank, clock);
          let mut target_buf_h = self.target_buf_h.lock().unwrap();
          //println!("DEBUG: async gossip: passive thread ({}): round: {} start recv", worker_rank, clock);
          self.recv_reqs.clear();
          //let recv_req = match client_conns[recv_rank].nonblocking_recv(&mut target_buf_h[ .. msg_len], Some(0), Some(1)) {
          let recv_req = match self.client_conns[recv_rank].nonblocking_recv(&mut target_buf_h[ .. self.msg_len], Some(0), Some(clock as i32)) {
            Ok(req) => req,
            Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
          };
          self.recv_reqs.append(recv_req);
          for msg in 1 .. self.num_buf_msgs {
            //Mpi::blocking_recv(&mut target_buf_h[msg * msg_len .. (msg+1) * msg_len], Some(recv_rank)).unwrap();
            //println!("DEBUG: async gossip: passive thread ({}): did recv {}/{}", worker_rank, msg, num_buf_msgs);
            let recv_req = match self.client_conns[recv_rank].nonblocking_recv(&mut target_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], Some(0), Some(0)) {
              Ok(req) => req,
              Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
            };
            self.recv_reqs.append(recv_req);
          }
          self.recv_reqs.wait_all();
          //println!("DEBUG: async gossip: passive thread ({}): round: {} finish recv", worker_rank, clock);
          self.pass2act_tx.send(SyncGossipPass2ActMsg::DoneRound{clock: clock, success: true}).unwrap();
        }
      }
    }
  }
}

pub struct MpiDistSyncGossipCommWorker {
  worker_data:  WorkerData,
  context:      Rc<DeviceContext>,
  mpi:          Mpi,

  buf_len:      usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  com_interval: usize,

  origin_buf:   RawDeviceBuffer<f32>,
  target_buf:   RawDeviceBuffer<f32>,
  origin_buf_h: Vec<f32>,
  //target_buf_h: Vec<f32>,
  target_buf_h: Arc<Mutex<Vec<f32>>>,
  //target_win_h: MpiWindow<f32>,

  //client_conns: VecMap<MpiComm>,
  server_conns: VecMap<MpiComm>,
  server_ports: VecMap<CString>,

  act2pass_tx:  Sender<SyncGossipAct2PassMsg>,
  pass2act_rx:  Receiver<SyncGossipPass2ActMsg>,
  passive_thr:  JoinHandle<()>,
  send_reqs:    MpiRequestList,

  avg_reduce:   AverageReduceOperation<f32>,

  shared_seed:  [u64; 2],
  shared_rng:   Xorshiftplus128Rng,
  ranks_perm:   Vec<usize>,
  iter_counter: usize,
}

impl MpiDistSyncGossipCommWorker {
  pub fn new(gossip_cfg: GossipConfig, context: Rc<DeviceContext>) -> MpiDistSyncGossipCommWorker {
    // XXX(20160415): Empirically determined message length.
    //let msg_len = 16; // FIXME(20160419): for debugging.
    let msg_len = 32 * 1024;
    let num_buf_msgs = (gossip_cfg.buf_size + msg_len - 1) / msg_len;
    let buf_len = num_buf_msgs * msg_len;
    //let num_buf_msgs = 1; // FIXME(20160419): for debugging.
    //let num_buf_msgs = 2; // FIXME(20160419): for debugging.

    let ctx = &(*context).as_ref();
    let origin_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    let target_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };

    let mpi = Mpi::new();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let mut origin_buf_h = Vec::with_capacity(buf_len);
    unsafe { origin_buf_h.set_len(buf_len) };
    let mut target_buf_h = Vec::with_capacity(buf_len);
    unsafe { target_buf_h.set_len(buf_len) };
    /*let target_win_h = match unsafe { MpiWindow::create(target_buf_h.as_mut_ptr(), target_buf_h.len(), &mpi) } {
      Ok(win) => win,
      Err(e) => panic!("comm worker: failed to create MPI window"),
    };*/
    let target_buf_h = Arc::new(Mutex::new(target_buf_h));

    let service_port = Mpi::open_port_().unwrap();
    let mut service_name_buf = vec![];
    write!(&mut service_name_buf, "rembrandt_server_{}", worker_rank);
    let service_name = CString::new(service_name_buf).unwrap();
    //println!("DEBUG: rank: {} service name: {:?}", worker_rank, service_name);
    Mpi::publish_service_(&service_name, false, true, &service_port).unwrap();
    Mpi::barrier_().unwrap();

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
    let passive_thr = {
      let target_buf_h = target_buf_h.clone();
      spawn(move || {
        let passive_worker = MpiDistSyncGossipPassiveWorker{
          worker_rank:  worker_rank,
          msg_len:      msg_len,
          num_buf_msgs: num_buf_msgs,
          target_buf_h: target_buf_h,
          act2pass_rx:  act2pass_rx,
          pass2act_tx:  pass2act_tx,
          client_conns: client_conns,
          recv_reqs:    MpiRequestList::new(),
        };
        passive_worker.run();
      })
    };

    let mut shared_seed = [0, 0];
    if worker_rank == 0 {
      shared_seed = [thread_rng().next_u64(), thread_rng().next_u64()];
    }
    mpi.broadcast(&mut shared_seed, 0);

    MpiDistSyncGossipCommWorker{
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
      target_buf:   target_buf,
      origin_buf_h: origin_buf_h,
      target_buf_h: target_buf_h,
      //target_win_h: target_win_h,
      //client_conns: client_conns,
      server_conns: server_conns,
      server_ports: server_ports,
      act2pass_tx:  act2pass_tx,
      pass2act_rx:  pass2act_rx,
      passive_thr:  passive_thr,
      send_reqs:    MpiRequestList::new(),
      avg_reduce:   AverageReduceOperation::new(0),
      shared_seed:  shared_seed,
      shared_rng:   Xorshiftplus128Rng::from_seed(shared_seed),
      ranks_perm:   (0 .. num_workers).collect(),
      iter_counter: 0,
    }
  }
}

impl CommWorker for MpiDistSyncGossipCommWorker {
  fn next(&mut self) -> bool {
    // FIXME(20160412)
    self.iter_counter += 1;
    /*if self.worker_data.worker_rank() == 0 {
      println!("DEBUG: next: {}", self.iter_counter);
    }*/
    true
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let data = data.as_view(ctx).data;
    data.raw_send(
        //&self.target_buf.as_ref_range(offset, offset + data.len()),
        &self.origin_buf.as_ref_range(offset, offset + data.len()),
    );
  }

  fn communicate(&mut self, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    // FIXME(20160412): steps for distributed gossip:
    // - first, sync on the previous loads (this can happen in the caller)
    // - optionally, wait on a world barrier
    // - pick a target according to a permutation
    // - call a fence-protected RMA get
    // - wait on a final world barrier

    self.shared_rng.shuffle(&mut self.ranks_perm);
    let self_rank = self.worker_data.worker_rank();
    let send_rank = self.ranks_perm[self_rank];

    let mut recv_rank = self_rank;
    for r in 0 .. self.worker_data.num_workers() {
      if self.ranks_perm[r] == self_rank {
        recv_rank = r;
        break;
      }
    }

    // FIXME(20160415): getting communication to work.

    if self_rank == send_rank {
      assert_eq!(send_rank, recv_rank);
      //self.target_buf.raw_send(&self.origin_buf, ctx);
      self.origin_buf.raw_send(&self.target_buf, ctx);

    } else {
      //println!("DEBUG: async gossip: active thread ({}): round: {} starting gossip round", self_rank, self.iter_counter);
      self.act2pass_tx.send(SyncGossipAct2PassMsg::StartRound{clock: self.iter_counter, recv_rank: recv_rank}).unwrap();

      if self_rank != send_rank {
        //self.target_buf.sync_store(&mut self.target_buf_h, ctx);
        self.origin_buf.sync_store(&mut self.origin_buf_h, ctx);

        //println!("DEBUG: async gossip: active thread ({}): round: {} initial send rank: {}", self_rank, self.iter_counter, send_rank);
        //self.target_win_h.lock(send_rank, MpiWindowLockMode::Exclusive).unwrap();
        self.send_reqs.clear();
        //let send_req = match self.server_conns[send_rank].nonblocking_send(&self.origin_buf_h[ .. self.msg_len], 0, 1) {
        //let send_req = match self.server_conns[send_rank].nonblocking_sync_send(&self.origin_buf_h[ .. self.msg_len], 0, 1) {
        let send_req = match self.server_conns[send_rank].nonblocking_sync_send(&self.origin_buf_h[ .. self.msg_len], 0, self.iter_counter as i32) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: active thread: failed to do initial send: {:?}", e),
        };
        self.send_reqs.append(send_req);
        self.send_reqs.wait_all();
        //println!("DEBUG: async gossip: active thread ({}): round: {} remaining sends rank: {}", self_rank, self.iter_counter, send_rank);
        for msg in 1 .. self.num_buf_msgs {
          //Mpi::blocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], send_rank).unwrap();
          //println!("DEBUG: async gossip: passive thread ({}): did send {}/{}", self_rank, msg, self.num_buf_msgs);
          let send_req = match self.server_conns[send_rank].nonblocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], 0, 0) {
            Ok(req) => req,
            Err(e) => panic!("async gossip: active thread: failed to do nonblocking send: {:?}", e),
          };
          self.send_reqs.append(send_req);
        }
        self.send_reqs.wait_all();
        //self.target_win_h.unlock(send_rank).unwrap();
      }

      //println!("DEBUG: async gossip: active thread ({}): round: {} waiting for passive response", self_rank, self.iter_counter);
      match self.pass2act_rx.recv() {
        Err(e) => {
          panic!("async gossip: active thread: failed to receive msg from passive thread: {:?}", e);
        }
        Ok(SyncGossipPass2ActMsg::DoneRound{clock, success}) => {
          assert_eq!(clock, self.iter_counter);
        }
      }
      //println!("DEBUG: async gossip: active thread ({}): round: {} got passive response", self_rank, self.iter_counter);

    }

    if self_rank == send_rank {
      return;
    }

    // FIXME(20160415): load from the correct buffer.
    //self.origin_buf.sync_load(&self.origin_buf_h, ctx);
    {
      //println!("DEBUG: async gossip: active thread ({}): round: {} acquire lock for load", self_rank, self.iter_counter);
      let target_buf_h = self.target_buf_h.lock().unwrap();
      //println!("DEBUG: async gossip: active thread ({}): lock acquired", self_rank);
      self.target_buf.sync_load(&target_buf_h, ctx);
    }
    //println!("DEBUG: async gossip: active thread ({}): round: {} reduce", self_rank, self.iter_counter);
    self.avg_reduce.reduce(
        /*&(self.target_buf).as_ref(),
        &(self.origin_buf).as_ref(),*/
        &(self.origin_buf).as_ref(),
        &(self.target_buf).as_ref(),
        ctx,
    );
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let mut data = data.as_view_mut(ctx).data;
    let data_len = data.len();
    data.raw_recv(
        //&self.origin_buf.as_ref_range(offset, offset + data_len),
        &self.target_buf.as_ref_range(offset, offset + data_len),
    );
  }
}

enum AsyncGossipAct2PassMsg {
  Quit,
  StartRound{clock: usize},
  StopRound{clock: usize},
}

enum AsyncGossipPass2ActMsg {
  DoneRound{clock: usize, success: bool},
}

struct MpiDistAsyncGossipPassiveWorker {
  worker_rank:  usize,
  num_workers:  usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  target_buf_h: Arc<Mutex<Vec<f32>>>,
  act2pass_rx:  Receiver<AsyncGossipAct2PassMsg>,
  pass2act_tx:  Sender<AsyncGossipPass2ActMsg>,
  client_conns: VecMap<MpiComm>,
  recv_reqs:    MpiRequestList,
  recv_count:   Arc<AtomicUsize>,
}

impl MpiDistAsyncGossipPassiveWorker {
  pub fn run(mut self) {
    let sleep_duration = Duration::new(0, 50_000);
    'run_loop: loop {
      match self.act2pass_rx.recv() {
        Err(_) => {
          println!("DEBUG: async gossip: passive thread ({}): loop terminated early", self.worker_rank);
          break;
        }
        Ok(AsyncGossipAct2PassMsg::Quit) => {
          break;
        }
        Ok(AsyncGossipAct2PassMsg::StartRound{clock}) => {
          println!("DEBUG: async gossip: passive thread ({}): round: {} starting gossip round", self.worker_rank, clock);
          let mut received = false;
          {
            //println!("DEBUG: async gossip: passive thread ({}): first recv", worker_rank);
            let mut recv_rank = None;
            'probe_loop: loop {
            //'probe_loop: for _ in 0 .. 100 {
              for r in 0 .. self.num_workers {
                if r == self.worker_rank {
                  continue;
                }
                match self.client_conns[r].nonblocking_probe(Some(0), Some(1)) {
                  Ok(None) => {}
                  Ok(Some(_)) => {
                    recv_rank = Some(r);
                    break 'probe_loop;
                  }
                  Err(e) => panic!("async gossip: passive thread: failed to do first recv: {:?}", e),
                };
              }
              sleep(sleep_duration);
              match self.act2pass_rx.recv() {
                Ok(AsyncGossipAct2PassMsg::StopRound{..}) => {
                  continue 'run_loop;
                }
                Ok(_) => unreachable!(),
                Err(e) => panic!("async gossip: passive thread: failed to peek in msg queue: {:?}", e)
              }
            }
            if let Some(recv_rank) = recv_rank {
              println!("DEBUG: async gossip: passive thread ({}): round: {} recv rank: {}", self.worker_rank, clock, recv_rank);
              received = true;
              println!("DEBUG: async gossip: passive thread ({}): round: {} acquire lock for recv", self.worker_rank, clock);
              let mut target_buf_h = self.target_buf_h.lock().unwrap();
              println!("DEBUG: async gossip: passive thread ({}): round: {} start recv", self.worker_rank, clock);
              self.recv_reqs.clear();
              let recv_req = match self.client_conns[recv_rank].nonblocking_recv(&mut target_buf_h[ .. self.msg_len], Some(0), Some(1)) {
                Ok(req) => req,
                Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
              };
              self.recv_reqs.append(recv_req);
              for msg in 1 .. self.num_buf_msgs {
                //Mpi::blocking_recv(&mut target_buf_h[msg * msg_len .. (msg+1) * msg_len], Some(recv_rank)).unwrap();
                //println!("DEBUG: async gossip: passive thread ({}): did recv {}/{}", worker_rank, msg, num_buf_msgs);
                let recv_req = match self.client_conns[recv_rank].nonblocking_recv(&mut target_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], Some(0), Some(0)) {
                  Ok(req) => req,
                  Err(e) => panic!("async gossip: passive thread: failed to do nonblocking recv: {:?}", e),
                };
                self.recv_reqs.append(recv_req);
              }
              self.recv_reqs.wait_all();
              println!("DEBUG: async gossip: passive thread ({}): round: {} finish recv", self.worker_rank, clock);
            } else {
              println!("DEBUG: async gossip: passive thread ({}): round: {} no recv", self.worker_rank, clock);
            }
          }
          self.pass2act_tx.send(AsyncGossipPass2ActMsg::DoneRound{clock: clock, success: received}).unwrap();
        }
        Ok(AsyncGossipAct2PassMsg::StopRound{clock}) => {
        }
      }
    }
  }
}

pub struct MpiDistAsyncGossipCommWorker {
  worker_data:  WorkerData,
  context:      Rc<DeviceContext>,
  mpi:          Mpi,

  buf_len:      usize,
  msg_len:      usize,
  num_buf_msgs: usize,
  com_interval: usize,

  origin_buf:   RawDeviceBuffer<f32>,
  target_buf:   RawDeviceBuffer<f32>,
  origin_buf_h: Vec<f32>,
  //target_buf_h: Vec<f32>,
  target_buf_h: Arc<Mutex<Vec<f32>>>,
  //target_win_h: MpiWindow<f32>,

  //client_conns: VecMap<MpiComm>,
  server_conns: VecMap<MpiComm>,
  server_ports: VecMap<CString>,

  act2pass_tx:  Sender<AsyncGossipAct2PassMsg>,
  pass2act_rx:  Receiver<AsyncGossipPass2ActMsg>,
  passive_thr:  JoinHandle<()>,
  send_reqs:    MpiRequestList,
  recv_count:   Arc<AtomicUsize>,

  avg_reduce:   AverageReduceOperation<f32>,

  shared_seed:  [u64; 2],
  rng:          Xorshiftplus128Rng,
  ranks_perm:   Vec<usize>,
  iter_counter: usize,
  recv_success: bool,
}

impl MpiDistAsyncGossipCommWorker {
  pub fn new(gossip_cfg: GossipConfig, context: Rc<DeviceContext>) -> MpiDistAsyncGossipCommWorker {
    // XXX(20160415): Empirically determined message length.
    let msg_len = 32 * 1024;
    let num_buf_msgs = (gossip_cfg.buf_size + msg_len - 1) / msg_len;
    let buf_len = num_buf_msgs * msg_len;
    // FIXME(20160419): for debugging.
    //let num_buf_msgs = 1;

    let ctx = &(*context).as_ref();
    let origin_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };
    let target_buf = unsafe { RawDeviceBuffer::new(buf_len, ctx) };

    let mpi = Mpi::new();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();

    let mut origin_buf_h = Vec::with_capacity(buf_len);
    unsafe { origin_buf_h.set_len(buf_len) };
    let mut target_buf_h = Vec::with_capacity(buf_len);
    unsafe { target_buf_h.set_len(buf_len) };
    /*let target_win_h = match unsafe { MpiWindow::create(target_buf_h.as_mut_ptr(), target_buf_h.len(), &mpi) } {
      Ok(win) => win,
      Err(e) => panic!("comm worker: failed to create MPI window"),
    };*/
    let target_buf_h = Arc::new(Mutex::new(target_buf_h));

    let service_port = Mpi::open_port_().unwrap();
    let mut service_name_buf = vec![];
    write!(&mut service_name_buf, "rembrandt_server_{}", worker_rank);
    let service_name = CString::new(service_name_buf).unwrap();
    //println!("DEBUG: rank: {} service name: {:?}", worker_rank, service_name);
    Mpi::publish_service_(&service_name, false, true, &service_port).unwrap();
    Mpi::barrier_().unwrap();

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
    let passive_thr = {
      let target_buf_h = target_buf_h.clone();
      let recv_count = recv_count.clone();
      spawn(move || {
        let passive_worker = MpiDistAsyncGossipPassiveWorker{
          worker_rank:  worker_rank,
          num_workers:  num_workers,
          msg_len:      msg_len,
          num_buf_msgs: num_buf_msgs,
          target_buf_h: target_buf_h,
          act2pass_rx:  act2pass_rx,
          pass2act_tx:  pass2act_tx,
          client_conns: client_conns,
          recv_reqs:    MpiRequestList::new(),
          recv_count:   recv_count,
        };
        passive_worker.run();
      })
    };

    let mut shared_seed = [0, 0];
    if worker_rank == 0 {
      shared_seed = [thread_rng().next_u64(), thread_rng().next_u64()];
    }
    mpi.broadcast(&mut shared_seed, 0);

    MpiDistAsyncGossipCommWorker{
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
      target_buf:   target_buf,
      origin_buf_h: origin_buf_h,
      target_buf_h: target_buf_h,
      //target_win_h: target_win_h,
      //client_conns: client_conns,
      server_conns: server_conns,
      server_ports: server_ports,
      act2pass_tx:  act2pass_tx,
      pass2act_rx:  pass2act_rx,
      passive_thr:  passive_thr,
      send_reqs:    MpiRequestList::new(),
      recv_count:   recv_count,
      avg_reduce:   AverageReduceOperation::new(0),
      shared_seed:  shared_seed,
      rng:          Xorshiftplus128Rng::from_seed(shared_seed),
      ranks_perm:   (0 .. num_workers).collect(),
      iter_counter: 0,
      recv_success: true,
    }
  }
}

impl CommWorker for MpiDistAsyncGossipCommWorker {
  fn next(&mut self) -> bool {
    // FIXME(20160412)
    self.iter_counter += 1;
    /*if self.worker_data.worker_rank() == 0 {
      println!("DEBUG: next: {}", self.iter_counter);
    }*/
    true
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let data = data.as_view(ctx).data;
    data.raw_send(
        //&self.target_buf.as_ref_range(offset, offset + data.len()),
        &self.origin_buf.as_ref_range(offset, offset + data.len()),
    );
  }

  fn communicate(&mut self, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    // FIXME(20160412): steps for distributed gossip:
    // - first, sync on the previous loads (this can happen in the caller)
    // - optionally, wait on a world barrier
    // - pick a target according to a permutation
    // - call a fence-protected RMA get
    // - wait on a final world barrier

    self.rng.shuffle(&mut self.ranks_perm);
    let self_rank = self.worker_data.worker_rank();
    let send_rank = self.ranks_perm[self_rank];

    /*let mut recv_rank = self_rank;
    for r in 0 .. self.worker_data.num_workers() {
      if self.ranks_perm[r] == self_rank {
        recv_rank = r;
        break;
      }
    }

    if self_rank == send_rank || self_rank == recv_rank {
      assert_eq!(self_rank, send_rank);
      assert_eq!(self_rank, recv_rank);
      //self.target_buf.raw_send(&self.origin_buf, ctx);
      self.origin_buf.raw_send(&self.target_buf, ctx);
      return;
    }*/

    // FIXME(20160415): getting communication to work.

    if self_rank == send_rank {
    //if self_rank == send_rank && self.recv_success {
      //self.target_buf.raw_send(&self.origin_buf, ctx);
      self.origin_buf.raw_send(&self.target_buf, ctx);
      //self.recv_success = true;
      return;
    }

    println!("DEBUG: async gossip: active thread ({}): round: {} starting gossip round", self_rank, self.iter_counter);
    self.act2pass_tx.send(AsyncGossipAct2PassMsg::StartRound{clock: self.iter_counter}).unwrap();

    if self_rank != send_rank {
    //if self_rank != send_rank && self.recv_success {
      //self.target_buf.sync_store(&mut self.target_buf_h, ctx);
      self.origin_buf.sync_store(&mut self.origin_buf_h, ctx);

      println!("DEBUG: async gossip: active thread ({}): round: {} initial send rank: {}", self_rank, self.iter_counter, send_rank);
      //self.target_win_h.lock(send_rank, MpiWindowLockMode::Exclusive).unwrap();
      self.send_reqs.clear();
      let send_req = match self.server_conns[send_rank].nonblocking_sync_send(&self.origin_buf_h[ .. self.msg_len], 0, 1) {
      //let send_req = match self.server_conns[send_rank].nonblocking_send(&self.origin_buf_h[ .. self.msg_len], 0, 1) {
        Ok(req) => req,
        Err(e) => panic!("async gossip: active thread: failed to do initial send: {:?}", e),
      };
      self.send_reqs.append(send_req);
      self.send_reqs.wait_all();
      println!("DEBUG: async gossip: active thread ({}): round: {} remaining sends rank: {}", self_rank, self.iter_counter, send_rank);
      for msg in 1 .. self.num_buf_msgs {
        //Mpi::blocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], send_rank).unwrap();
        //println!("DEBUG: async gossip: passive thread ({}): did send {}/{}", self_rank, msg, self.num_buf_msgs);
        let send_req = match self.server_conns[send_rank].nonblocking_send(&self.origin_buf_h[msg * self.msg_len .. (msg+1) * self.msg_len], 0, 0) {
          Ok(req) => req,
          Err(e) => panic!("async gossip: active thread: failed to do nonblocking send: {:?}", e),
        };
        self.send_reqs.append(send_req);
      }
      self.send_reqs.wait_all();
      //self.target_win_h.unlock(send_rank).unwrap();
    }

    println!("DEBUG: async gossip: active thread ({}): round: {} waiting for passive response", self_rank, self.iter_counter);
    match self.pass2act_rx.recv() {
      Err(e) => {
        panic!("async gossip: active thread: failed to receive msg from passive thread: {:?}", e);
      }
      Ok(AsyncGossipPass2ActMsg::DoneRound{clock, success}) => {
        assert_eq!(clock, self.iter_counter);
        self.recv_success = success;
      }
    }
    println!("DEBUG: async gossip: active thread ({}): round: {} got passive response", self_rank, self.iter_counter);

    // FIXME(20160419): doing a sync barrier here for debugging.
    //Mpi::barrier_();

    // FIXME(20160415): load from the correct buffer.
    //self.origin_buf.sync_load(&self.origin_buf_h, ctx);
    {
      println!("DEBUG: async gossip: active thread ({}): round: {} acquire lock for load", self_rank, self.iter_counter);
      let target_buf_h = self.target_buf_h.lock().unwrap();
      //println!("DEBUG: async gossip: active thread ({}): lock acquired", self_rank);
      self.target_buf.sync_load(&target_buf_h, ctx);
    }
    println!("DEBUG: async gossip: active thread ({}): round: {} reduce", self_rank, self.iter_counter);
    self.avg_reduce.reduce(
        /*&(self.target_buf).as_ref(),
        &(self.origin_buf).as_ref(),*/
        &(self.origin_buf).as_ref(),
        &(self.target_buf).as_ref(),
        ctx,
    );
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    if self.iter_counter % self.com_interval != 0 {
      return;
    }

    let mut data = data.as_view_mut(ctx).data;
    let data_len = data.len();
    data.raw_recv(
        //&self.origin_buf.as_ref_range(offset, offset + data_len),
        &self.target_buf.as_ref_range(offset, offset + data_len),
    );
  }
}

#[derive(Clone)]
pub struct MpiDistPipelineOperatorWorkerBuilder {
  //num_workers:  usize,
  batch_size:   usize,
  config:       PipelineOperatorConfig,
  capability:   OpCapability,
  shared_seed:  [u64; 2],
  // XXX: Contravariance.
  //_marker:      PhantomData<fn () -> Comm>,
}

/*impl Clone for MpiDistPipelineOperatorWorkerBuilder {
  fn clone(&self) -> MpiDistPipelineOperatorWorkerBuilder {
    MpiDistPipelineOperatorWorkerBuilder{
      //num_workers:  self.num_workers,
      batch_size:   self.batch_size,
      config:       self.config.clone(),
      capability:   self.capability,
      shared_seed:  self.shared_seed,
      //_marker:      PhantomData,
    }
  }
}*/

impl MpiDistPipelineOperatorWorkerBuilder {
  //pub fn new(num_workers: usize, batch_size: usize, config: PipelineOperatorConfig<Comm>, capability: OpCapability) -> MpiDistPipelineOperatorWorkerBuilder<Comm> {
  pub fn new(/*num_workers: usize,*/ batch_size: usize, config: PipelineOperatorConfig, capability: OpCapability) -> MpiDistPipelineOperatorWorkerBuilder {
    MpiDistPipelineOperatorWorkerBuilder{
      //num_workers:  num_workers,
      batch_size:   batch_size,
      config:       config,
      capability:   capability,
      shared_seed:  [thread_rng().next_u64(), thread_rng().next_u64()],
      //_marker:      PhantomData,
    }
  }
}

impl MpiDistPipelineOperatorWorkerBuilder {
  //type Worker = MpiDistPipelineOperatorWorker;

  pub fn into_worker(self, /*tid: usize,*/ context: Rc<DeviceContext>) -> MpiDistPipelineOperatorWorker {
    let config = self.config.clone();
    let total_params_len = config.params_len();

    let gossip_cfg = GossipConfig{
      num_rounds:   1,
      buf_size:     total_params_len,
    };
    let comm_worker = Rc::new(RefCell::new(MpiDistAsyncGossipCommWorker::new(gossip_cfg, context.clone())));
    let worker_data = comm_worker.borrow().worker_data.clone();
    let mut shared_seed = [0u64, 0u64];
    if worker_data.worker_rank() == 0 {
      shared_seed = self.shared_seed;
    }
    comm_worker.borrow().mpi.broadcast(&mut shared_seed, 0);

    let input_op = config.input_op.unwrap().build_input_operator::<MpiDistAsyncGossipCommWorker>(self.batch_size, context.clone());
    let mut hidden_ops: Vec<Box<Operator>> = vec![];
    let mut params_off = 0;
    for r in 0 .. config.hidden_ops.len() {
      let hidden_op = {
        let prev_op = match r {
          0 => input_op.downcast(),
          _ => &*hidden_ops[r-1],
        };
        // FIXME(20160412): used fixed MPI comm worker.
        config.hidden_ops[r].build_operator::<MpiDistAsyncGossipCommWorker>(self.batch_size, self.capability, params_off, Some(prev_op), Some(comm_worker.clone()), context.clone())
      };
      hidden_ops.push(hidden_op);
      params_off += config.hidden_ops[r].params_len();
    }
    assert_eq!(params_off, total_params_len);
    let loss_op = {
      let num_hidden_ops = hidden_ops.len();
      let prev_op = match num_hidden_ops {
        0 => input_op.downcast(),
        _ => &*hidden_ops[num_hidden_ops-1],
      };
      config.loss_op.unwrap().build_loss_operator::<MpiDistAsyncGossipCommWorker>(self.batch_size, Some(prev_op), context.clone())
    };

    MpiDistPipelineOperatorWorker{
      worker_data:  worker_data,
      batch_size:   self.batch_size,
      config:       self.config,
      //shared_seed:  self.shared_seed,
      shared_seed:  shared_seed,
      context:      context,
      comm_worker:  comm_worker,
      input_op:     input_op,
      hidden_ops:   hidden_ops,
      loss_op:      loss_op,
    }
  }
}

pub struct MpiDistPipelineOperatorWorker {
  worker_data:  WorkerData,
  batch_size:   usize,
  config:       PipelineOperatorConfig,
  shared_seed:  [u64; 2],

  context:      Rc<DeviceContext>,
  //comm_worker:  Rc<RefCell<Comm>>,
  comm_worker:  Rc<RefCell<MpiDistAsyncGossipCommWorker>>,
  input_op:     Box<InputOperator>,
  hidden_ops:   Vec<Box<Operator>>,
  loss_op:      Box<LossOperator>,
}

impl MpiDistPipelineOperatorWorker {
}

impl OperatorWorker for MpiDistPipelineOperatorWorker {
  fn num_workers(&self) -> usize {
    self.worker_data.num_workers()
  }

  fn worker_rank(&self) -> usize {
    self.worker_data.tid()
  }

  fn shared_seed(&self) -> [u64; 2] {
    self.shared_seed
  }

  fn input_operator(&mut self) -> &mut InputOperator {
    &mut *self.input_op
  }

  fn loss_count(&self) -> usize {
    1
  }

  fn loss_operator(&mut self, rank: usize) -> &mut LossOperator {
    assert_eq!(rank, 0);
    &mut *self.loss_op
  }

  fn wait_barrier(&self) {
    // FIXME(20160402)
    unimplemented!();
  }

  fn next(&mut self) {
    self.comm_worker.borrow_mut().next();
  }
}

impl Operator for MpiDistPipelineOperatorWorker {
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn init_params(&mut self, shared_seed: [u64; 2]) {
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    for op in self.hidden_ops.iter_mut() {
      let op_seed = [rng.next_u64(), rng.next_u64()];
      op.init_params(op_seed);
    }
  }

  fn read_params(&mut self, blob: &[u8]) -> usize {
    let mut offset = 0;
    for op in self.hidden_ops.iter_mut() {
      offset += op.read_params(&blob[offset .. ]);
    }
    offset
  }

  fn write_params(&mut self, blob: &mut Vec<u8>) {
    for op in self.hidden_ops.iter_mut() {
      op.write_params(blob);
    }
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    self.input_op.forward(batch_size, phase);
    for op in self.hidden_ops.iter_mut() {
      op.forward(batch_size, phase);
    }
    self.loss_op.forward(batch_size, phase);
  }

  fn backward(&mut self, batch_size: usize) {
    self.loss_op.backward(batch_size);
    for op in self.hidden_ops.iter_mut().rev() {
      op.backward(batch_size);
    }
  }

  fn regularize(&mut self, reg: Regularization) {
    for op in self.hidden_ops.iter_mut() {
      op.regularize(reg);
    }
  }

  fn accumulate_grads(&mut self, scale: f32, momentum: f32) {
    for op in self.hidden_ops.iter_mut() {
      op.accumulate_grads(scale, momentum);
    }
  }

  fn update_params(&mut self, scale: f32) {
    for op in self.hidden_ops.iter_mut() {
      op.update_params(scale);
    }
  }

  fn save_params(&mut self) {
    for op in self.hidden_ops.iter_mut() {
      op.save_params();
    }
  }

  fn restore_params(&mut self) {
    for op in self.hidden_ops.iter_mut() {
      op.restore_params();
    }
  }

  fn set_grads_with_params_diff(&mut self) {
    for op in self.hidden_ops.iter_mut() {
      op.set_grads_with_params_diff();
    }
  }

  fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    if self.num_workers() <= 1 {
      return;
    }
    for op in self.hidden_ops.iter_mut() {
      op.stage_params();
    }
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }

  fn sync_params(&mut self) {
    if self.num_workers() <= 1 {
      return;
    }
    {
      let ctx = &(*self.context).as_ref();
      self.comm_worker.borrow_mut().communicate(ctx);
    }
    for op in self.hidden_ops.iter_mut() {
      op.sync_params();
    }
    let ctx = &(*self.context).as_ref();
    ctx.sync();
  }

  fn reset_grads(&mut self, scale: f32) {
    for op in self.hidden_ops.iter_mut() {
      op.reset_grads(scale);
    }
  }

  fn reset(&mut self) {
    for op in self.hidden_ops.iter_mut() {
      op.reset();
    }
  }
}
