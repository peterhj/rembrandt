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

use mpi::{Mpi, MpiWindow};
//use procgroup::{ProcGroup};
use threadpool::{ThreadPool};

use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell};
use std::collections::{HashSet};
use std::iter::{FromIterator, repeat};
use std::marker::{PhantomData};
use std::rc::{Rc};
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering};

/*#[derive(Clone)]
pub struct MpiDistSyncGossipCommWorkerBuilder {
  num_workers:  usize,
  num_rounds:   usize,
  period:       usize,
  //barrier:      Arc<Barrier>,
  shared_bufs:  Vec<Arc<RawDeviceBuffer<f32>>>,
  //shared_rns:   Arc<Vec<AtomicUsize>>,
  shared_seed:  [u64; 2],
}

impl MpiDistSyncGossipCommWorkerBuilder {
  pub fn new(config: GossipConfig, /*contexts: &[DeviceContext]*/) -> MpiDistSyncGossipCommWorkerBuilder {
    // FIXME(20160412)
    unimplemented!();
  }
}*/

pub struct MpiDistSyncGossipCommWorker {
  worker_data:  WorkerData,
  context:      Rc<DeviceContext>,
  mpi:          Mpi,

  origin_buf:   RawDeviceBuffer<f32>,
  target_buf:   RawDeviceBuffer<f32>,
  origin_buf_h: Vec<f32>,
  target_buf_h: Vec<f32>,
  target_win_h: MpiWindow<f32>,

  rng:          Xorshiftplus128Rng,
  ranks_perm:   Vec<usize>,
  iter_counter: usize,
}

impl MpiDistSyncGossipCommWorker {
  pub fn new(gossip_cfg: GossipConfig, context: Rc<DeviceContext>) -> MpiDistSyncGossipCommWorker {
    let buf_size = gossip_cfg.buf_size;

    let ctx = &(*context).as_ref();
    let origin_buf = unsafe { RawDeviceBuffer::new(buf_size, ctx) };
    let target_buf = unsafe { RawDeviceBuffer::new(buf_size, ctx) };

    let mpi = Mpi::new();
    let worker_rank = mpi.rank();
    let num_workers = mpi.size();
    let mut origin_buf_h = Vec::with_capacity(buf_size);
    unsafe { origin_buf_h.set_len(buf_size) };
    let mut target_buf_h = Vec::with_capacity(buf_size);
    unsafe { target_buf_h.set_len(buf_size) };
    let target_win_h = match unsafe { MpiWindow::create(target_buf_h.as_mut_ptr(), target_buf_h.len(), &mpi) } {
      Ok(win) => win,
      Err(e) => panic!("comm worker: failed to create MPI window"),
    };

    MpiDistSyncGossipCommWorker{
      worker_data:  WorkerData::new(worker_rank, num_workers),
      context:      context.clone(),
      mpi:          Mpi,
      origin_buf:   origin_buf,
      target_buf:   target_buf,
      origin_buf_h: origin_buf_h,
      target_buf_h: target_buf_h,
      target_win_h: target_win_h,
      rng:          Xorshiftplus128Rng::new(&mut thread_rng()),
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
    if self.iter_counter % 1000 != 0 {
      return;
    }

    let data = data.as_view(ctx).data;
    data.raw_send(
        &self.target_buf.as_ref_range(offset, offset + data.len()),
    );
  }

  fn communicate(&mut self, ctx: &DeviceCtxRef) {
    if self.iter_counter % 1000 != 0 {
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
    let other_rank = self.ranks_perm[self_rank];

    if self_rank == other_rank {
      //self.origin_buf.sync_load(&self.target_buf_h, ctx);
      self.target_buf.raw_send(&self.origin_buf, ctx);
      return;
    }

    self.target_buf.sync_store(&mut self.target_buf_h, ctx);
    //ctx.sync();

    // FIXME(20160415): getting communication to work.
    self.mpi.barrier();
    /*match unsafe { self.target_win_h.rma_get(self.origin_buf_h.as_mut_ptr(), self.origin_buf_h.len(), other_rank, &self.mpi) } {
      Ok(_) => {}
      Err(e) => panic!("mpi dist comm worker: failed to call rma_get: {:?}", e),
    }
    self.mpi.barrier();*/

    // FIXME(20160415): load from the correct buffer.
    self.origin_buf.sync_load(&self.target_buf_h, ctx);
    //self.origin_buf.sync_load(&self.origin_buf_h, ctx);
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    if self.iter_counter % 1000 != 0 {
      return;
    }

    let mut data = data.as_view_mut(ctx).data;
    let data_len = data.len();
    data.raw_recv(
        &self.origin_buf.as_ref_range(offset, offset + data_len),
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
    let comm_worker = Rc::new(RefCell::new(MpiDistSyncGossipCommWorker::new(gossip_cfg, context.clone())));
    let worker_data = comm_worker.borrow().worker_data.clone();

    let input_op = config.input_op.unwrap().build_input_operator::<MpiDistSyncGossipCommWorker>(self.batch_size, context.clone());
    let mut hidden_ops: Vec<Box<Operator>> = vec![];
    let mut params_off = 0;
    for r in 0 .. config.hidden_ops.len() {
      let hidden_op = {
        let prev_op = match r {
          0 => input_op.downcast(),
          _ => &*hidden_ops[r-1],
        };
        // FIXME(20160412): used fixed MPI comm worker.
        config.hidden_ops[r].build_operator::<MpiDistSyncGossipCommWorker>(self.batch_size, self.capability, params_off, Some(prev_op), Some(comm_worker.clone()), context.clone())
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
      config.loss_op.unwrap().build_loss_operator::<MpiDistSyncGossipCommWorker>(self.batch_size, Some(prev_op), context.clone())
    };

    MpiDistPipelineOperatorWorker{
      worker_data:  worker_data,
      batch_size:   self.batch_size,
      config:       self.config,
      shared_seed:  self.shared_seed,
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
  comm_worker:  Rc<RefCell<MpiDistSyncGossipCommWorker>>,
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
