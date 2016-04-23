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
