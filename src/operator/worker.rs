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
use operator::comm::{CommWorkerBuilder, CommWorker};
use operator::loss::{
  LossOperator,
  CategoricalLossConfig,
};

use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use rng::xorshift::{Xorshiftplus128Rng};
//use procgroup::{ProcGroup};
use threadpool::{ThreadPool};
use worker_::{WorkerData};

use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell};
use std::collections::{HashSet};
use std::iter::{FromIterator, repeat};
use std::marker::{PhantomData};
use std::rc::{Rc};
use std::sync::{Arc, Barrier};

pub trait OperatorWorkerBuilder<Comm>: Send + Clone where Comm: 'static + CommWorker {
  type Worker: OperatorWorker;

  fn into_worker(self, tid: usize, context: Rc<DeviceContext>, comm_worker: Rc<RefCell<Comm>>) -> Self::Worker;
}

pub trait OperatorWorker: Operator {
  fn num_workers(&self) -> usize;
  fn worker_rank(&self) -> usize;
  fn shared_seed(&self) -> [u64; 2];
  fn input_operator(&mut self) -> &mut InputOperator;
  fn loss_count(&self) -> usize;
  fn loss_operator(&mut self, rank: usize) -> &mut LossOperator;
  fn wait_barrier(&self) { unimplemented!(); }
  fn next(&mut self) {}
}

pub struct DeviceParallelOperatorServer<Comm, CBuilder, WBuilder>
where Comm: 'static + CommWorker,
      CBuilder: CommWorkerBuilder<Worker=Comm>,
      WBuilder: OperatorWorkerBuilder<Comm>,
{
  num_workers:  usize,
  worker_pool:  ThreadPool,
  comm_worker_builder:  CBuilder,
  op_worker_builder:    WBuilder,
  join_barrier: Arc<Barrier>,
  _marker:      PhantomData<(Comm)>,
}

impl<Comm, CBuilder, WBuilder> DeviceParallelOperatorServer<Comm, CBuilder, WBuilder>
where Comm: 'static + CommWorker + Sync,
      CBuilder: 'static + CommWorkerBuilder<Worker=Comm>,
      WBuilder: 'static + OperatorWorkerBuilder<Comm>,
{
  pub fn new(num_workers: usize, comm_worker_builder: CBuilder, op_worker_builder: WBuilder) -> DeviceParallelOperatorServer<Comm, CBuilder, WBuilder> {
    DeviceParallelOperatorServer{
      num_workers:  num_workers,
      worker_pool:  ThreadPool::new(num_workers),
      comm_worker_builder:    comm_worker_builder,
      op_worker_builder:      op_worker_builder,
      join_barrier: Arc::new(Barrier::new(num_workers + 1)),
      _marker:      PhantomData,
    }
  }

  pub fn fork<F>(&self, tid: usize, f: F) where F: FnOnce(WBuilder::Worker) + Send + 'static {
    let comm_worker_builder = self.comm_worker_builder.clone();
    let op_worker_builder = self.op_worker_builder.clone();
    let join_barrier = self.join_barrier.clone();
    self.worker_pool.execute(move || {
      let context = Rc::new(DeviceContext::new(tid));
      let comm_worker = Rc::new(RefCell::new(comm_worker_builder.into_worker(tid)));
      let mut op_worker = op_worker_builder.into_worker(tid, context, comm_worker);
      f(op_worker);
      join_barrier.wait();
    });
  }

  pub fn join(self) {
    self.join_barrier.wait();
  }
}

// FIXME(20160330): why can't I derive Clone here?
//#[derive(Clone)]
/*pub struct PipelineOperatorConfig<Comm> where Comm: 'static + CommWorker {
  pub input_op:     Option<OperatorConfig<Comm>>,
  pub hidden_ops:   Vec<OperatorConfig<Comm>>,
  pub loss_op:      Option<OperatorConfig<Comm>>,
}*/

#[derive(Clone)]
pub struct PipelineOperatorConfig {
  pub input_op:     Option<OperatorConfig>,
  pub hidden_ops:   Vec<OperatorConfig>,
  pub loss_op:      Option<OperatorConfig>,
}

/*impl<Comm> Clone for PipelineOperatorConfig<Comm> where Comm: 'static + CommWorker {
  fn clone(&self) -> PipelineOperatorConfig<Comm> {
    PipelineOperatorConfig{
      input_op:     self.input_op.clone(),
      hidden_ops:   self.hidden_ops.clone(),
      loss_op:      self.loss_op.clone(),
    }
  }
}*/

//impl<Comm> PipelineOperatorConfig<Comm> where Comm: 'static + CommWorker {
impl PipelineOperatorConfig {
  //pub fn new() -> PipelineOperatorConfig<Comm> {
  pub fn new() -> PipelineOperatorConfig {
    PipelineOperatorConfig{
      input_op:     None,
      hidden_ops:   vec![],
      loss_op:      None,
    }
  }

  pub fn params_len(&self) -> usize {
    let mut params_len = 0;
    for op in self.hidden_ops.iter() {
      params_len += op.params_len();
    }
    params_len
  }

  pub fn data3d(&mut self, cfg: Data3dOperatorConfig) -> &mut Self {
    self.input_op = Some(OperatorConfig::Data3d(cfg));
    self
  }

  pub fn affine(&mut self, cfg: AffineOperatorConfig) -> &mut Self  {
    self.hidden_ops.push(OperatorConfig::Affine(cfg));
    self
  }

  pub fn conv2d(&mut self, cfg: Conv2dOperatorConfig) -> &mut Self  {
    self.hidden_ops.push(OperatorConfig::Conv2d(cfg));
    self
  }

  pub fn pool2d(&mut self, cfg: Pool2dOperatorConfig) -> &mut Self  {
    self.hidden_ops.push(OperatorConfig::Pool2d(cfg));
    self
  }

  pub fn dropout(&mut self, cfg: DropoutOperatorConfig) -> &mut Self {
    self.hidden_ops.push(OperatorConfig::Dropout(cfg));
    self
  }

  pub fn softmax_kl_loss(&mut self, cfg: CategoricalLossConfig) -> &mut Self  {
    self.loss_op = Some(OperatorConfig::SoftmaxKLLoss(cfg));
    self
  }
}

pub struct PipelineOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
//pub struct PipelineOperatorWorkerBuilder {
  num_workers:  usize,
  batch_size:   usize,
  //config:       PipelineOperatorConfig<Comm>,
  config:       PipelineOperatorConfig,
  capability:   OpCapability,
  shared_seed:  [u64; 2],
  // XXX: Contravariance.
  _marker:      PhantomData<fn () -> Comm>,
}

impl<Comm> Clone for PipelineOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
  fn clone(&self) -> PipelineOperatorWorkerBuilder<Comm> {
    PipelineOperatorWorkerBuilder{
      num_workers:  self.num_workers,
      batch_size:   self.batch_size,
      config:       self.config.clone(),
      capability:   self.capability,
      shared_seed:  self.shared_seed,
      _marker:      PhantomData,
    }
  }
}

impl<Comm> PipelineOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
  //pub fn new(num_workers: usize, batch_size: usize, config: PipelineOperatorConfig<Comm>, capability: OpCapability) -> PipelineOperatorWorkerBuilder<Comm> {
  pub fn new(num_workers: usize, batch_size: usize, config: PipelineOperatorConfig, capability: OpCapability) -> PipelineOperatorWorkerBuilder<Comm> {
    PipelineOperatorWorkerBuilder{
      num_workers:  num_workers,
      batch_size:   batch_size,
      config:       config,
      capability:   capability,
      shared_seed:  [thread_rng().next_u64(), thread_rng().next_u64()],
      _marker:      PhantomData,
    }
  }
}

impl<Comm> OperatorWorkerBuilder<Comm> for PipelineOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
  type Worker = PipelineOperatorWorker<Comm>;

  fn into_worker(self, tid: usize, context: Rc<DeviceContext>, comm_worker: Rc<RefCell<Comm>>) -> PipelineOperatorWorker<Comm> {
    let config = self.config.clone();
    let total_params_len = config.params_len();
    let input_op = config.input_op.unwrap().build_input_operator::<Comm>(self.batch_size, context.clone());
    let mut hidden_ops: Vec<Box<Operator>> = vec![];
    let mut params_off = 0;
    for r in 0 .. config.hidden_ops.len() {
      let hidden_op = {
        let prev_op = match r {
          0 => input_op.downcast(),
          _ => &*hidden_ops[r-1],
        };
        config.hidden_ops[r].build_operator(self.batch_size, self.capability, params_off, Some(prev_op), Some(comm_worker.clone()), context.clone())
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
      config.loss_op.unwrap().build_loss_operator::<Comm>(self.batch_size, Some(prev_op), context.clone())
    };
    PipelineOperatorWorker{
      worker_data:  WorkerData::new(tid, self.num_workers),
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

pub struct PipelineOperatorWorker<Comm> where Comm: 'static + CommWorker {
  worker_data:  WorkerData,
  batch_size:   usize,
  //config:       PipelineOperatorConfig<Comm>,
  config:       PipelineOperatorConfig,
  shared_seed:  [u64; 2],

  context:      Rc<DeviceContext>,
  comm_worker:  Rc<RefCell<Comm>>,
  input_op:     Box<InputOperator>,
  hidden_ops:   Vec<Box<Operator>>,
  loss_op:      Box<LossOperator>,
}

impl<Comm> PipelineOperatorWorker<Comm> where Comm: 'static + CommWorker {
}

impl<Comm> OperatorWorker for PipelineOperatorWorker<Comm> where Comm: 'static + CommWorker {
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
}

impl<Comm> Operator for PipelineOperatorWorker<Comm> where Comm: 'static + CommWorker {
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

  /*fn reset_params(&mut self, momentum: f32, nesterov: bool) {
    for op in self.hidden_ops.iter_mut() {
      op.reset_params(momentum, nesterov);
    }
  }*/

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

//pub struct GraphOperatorConfig<Comm> {
pub struct GraphOperatorConfig {
  //pub ops:  Vec<(Vec<usize>, OperatorConfig<Comm>)>,
  pub ops:  Vec<(Vec<usize>, OperatorConfig)>,
}

//impl<Comm> GraphOperatorConfig<Comm> where Comm: CommWorker {
impl GraphOperatorConfig {
  pub fn data3d(&mut self, cfg: Data3dOperatorConfig) -> usize {
    let curr_id = self.ops.len();
    self.ops.push((vec![], OperatorConfig::Data3d(cfg)));
    curr_id
  }

  pub fn affine(&mut self, prev_ids: Vec<usize>, cfg: AffineOperatorConfig) -> usize {
    let curr_id = self.ops.len();
    self.ops.push((prev_ids, OperatorConfig::Affine(cfg)));
    curr_id
  }

  pub fn conv2d(&mut self, prev_ids: Vec<usize>, cfg: Conv2dOperatorConfig) -> usize {
    let curr_id = self.ops.len();
    self.ops.push((prev_ids, OperatorConfig::Conv2d(cfg)));
    curr_id
  }

  pub fn softmax_kl_loss(&mut self, prev_ids: Vec<usize>, cfg: CategoricalLossConfig) -> usize {
    let curr_id = self.ops.len();
    self.ops.push((prev_ids, OperatorConfig::SoftmaxKLLoss(cfg)));
    curr_id
  }
}

pub struct GraphOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
  //op_nodes:     Vec<OperatorConfig<Comm>>,
  op_nodes:     Vec<OperatorConfig>,
  in_node_ids:  Vec<Vec<usize>>,
  out_node_ids: Vec<Vec<usize>>,
  top_order:    Vec<usize>,
  _marker:      PhantomData<fn () -> Comm>,
}

impl<Comm> GraphOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
  //pub fn new(config: GraphOperatorConfig<Comm>) -> GraphOperatorWorkerBuilder<Comm> {
  pub fn new(config: GraphOperatorConfig) -> GraphOperatorWorkerBuilder<Comm> {
    //let mut op_nodes: Vec<OperatorConfig<Comm>> = vec![];
    let mut op_nodes: Vec<OperatorConfig> = vec![];
    let num_ops = config.ops.len();
    let mut in_node_ids = vec![];
    let mut out_node_ids = vec![];
    for _ in 0 .. num_ops {
      in_node_ids.push(vec![]);
      out_node_ids.push(vec![]);
    }
    for (curr_id, &(ref prev_ids, ref op_cfg)) in config.ops.iter().enumerate() {
      op_nodes.push(op_cfg.clone());
      for &prev_id in prev_ids.iter() {
        in_node_ids[curr_id].push(prev_id);
        out_node_ids[prev_id].push(curr_id);
      }
    }
    // Insert "structural" operators as needed.
    let mut new_id = num_ops;
    for curr_id in 0 .. num_ops {
      if in_node_ids[curr_id].len() <= 1 && out_node_ids[curr_id].len() <= 1 {
        // Do nothing.
      } else if in_node_ids[curr_id].len() <= 1 && out_node_ids[curr_id].len() > 1 {
        // FIXME(20160330): Insert a `SplitOperator`.
        //op_nodes.push(OperatorConfig::Split(op_nodes[curr_id].config().get_in_dims()));
        in_node_ids.push(vec![curr_id]);
        let curr_out_nodes = out_node_ids[curr_id].clone();
        out_node_ids.push(curr_out_nodes);
        out_node_ids[curr_id] = vec![new_id];
        new_id += 1;
      } else if in_node_ids[curr_id].len() > 1 && out_node_ids[curr_id].len() <= 1 {
        unimplemented!();
      } else if in_node_ids[curr_id].len() > 1 && out_node_ids[curr_id].len() > 1 {
        unimplemented!();
      }
    }
    // Sort the graph into topological order.
    let num_ops = op_nodes.len();
    let mut top_order = Vec::with_capacity(num_ops);
    let mut unmarked = HashSet::from_iter(0 .. num_ops);
    let mut ismarked: Vec<_> = repeat(false).take(num_ops).collect();
    fn dfs_top_sort(node_id: usize, unmarked: &mut HashSet<usize>, ismarked: &mut [bool], top_order: &mut Vec<usize>, out_node_ids: &Vec<Vec<usize>>) {
      if !ismarked[node_id] {
        for &out_id in out_node_ids[node_id].iter() {
          dfs_top_sort(out_id, unmarked, ismarked, top_order, out_node_ids);
        }
        unmarked.remove(&node_id);
        ismarked[node_id] = true;
        top_order.push(node_id);
      }
    }
    while !unmarked.is_empty() {
      dfs_top_sort(*unmarked.iter().next().unwrap(), &mut unmarked, &mut ismarked, &mut top_order, &out_node_ids);
    }
    top_order.reverse();
    GraphOperatorWorkerBuilder{
      op_nodes:     op_nodes,
      in_node_ids:  in_node_ids,
      out_node_ids: out_node_ids,
      top_order:    vec![],
      _marker:      PhantomData,
    }
  }

  pub fn into_worker(self, tid: usize, comm_worker: Comm) -> GraphOperatorWorker<Comm> {
    // FIXME(20160330)
    unimplemented!();
  }
}

pub struct GraphOperatorWorker<Comm> where Comm: CommWorker {
  worker_data:  WorkerData,
  comm_worker:  Comm,
}

/*#[derive(Clone)]
pub struct DistOperatorWorkerBuilder<B, Comm>
where B: OperatorWorkerBuilder<Comm>,
      Comm: 'static + CommWorker,
{
  inner_builder:    B,
  _marker:          PhantomData<Comm>,
}

impl<B, Comm> DistOperatorWorkerBuilder<B, Comm>
where B: OperatorWorkerBuilder<Comm>,
      Comm: 'static + CommWorker,
{
}

impl<Comm, B> OperatorWorkerBuilder<Comm> for DistOperatorWorkerBuilder<Comm, B>
where B: OperatorWorkerBuilder<Comm>,
      Comm: 'static + CommWorker,
{
  pub fn into_worker(self) -> DistOperatorWorker<B::Worker, Comm> {
    // FIXME(20160411)
    unimplemented!();
  }
}

pub struct DistOperatorWorker<W, Comm>
//where W: OperatorWorker, Pg: ProcGroup<Pid=usize> {
where W: OperatorWorker,
      Comm: 'static + CommWorker,
{
  //proc_group:   Pg,
  worker_data:  WorkerData,
  inner_worker: W,
  //_marker:      PhantomData<Pg>,
}

impl<W, Comm> DistOperatorWorker<W, Comm>
//where W: OperatorWorker, Pg: ProcGroup<Pid=usize> {
where W: OperatorWorker,
      Comm: 'static + CommWorker,
{
}

impl<W, Comm> OperatorWorker for DistOperatorWorker<W, Comm>
//where W: OperatorWorker, Pg: ProcGroup<Pid=usize> {
where W: OperatorWorker,
      Comm: 'static + CommWorker,
{
  fn num_workers(&self) -> usize {
    /*// FIXME(20160402): this is absolutely false, since different nodes may
    // have different numbers of workers.
    let num_local_workers = self.inner_worker.num_workers();
    let num_proc_peers = 0;
    unimplemented!()
    //num_local_workers * num_proc_peers*/
    self.worker_data.num_workers()
  }

  fn worker_rank(&self) -> usize {
    // FIXME(20160402)
    //unimplemented!();
    self.worker_data.tid()
  }

  fn shared_seed(&self) -> [u64; 2] {
    self.inner_worker.shared_seed()
  }

  fn input_operator(&mut self) -> &mut InputOperator {
    self.inner_worker.input_operator()
  }

  fn loss_count(&self) -> usize {
    self.inner_worker.loss_count()
  }

  fn loss_operator(&mut self, rank: usize) -> &mut LossOperator {
    self.inner_worker.loss_operator(rank)
  }

  fn wait_barrier(&self) {
  }
}

impl<W, Comm> Operator for DistOperatorWorker<W, Comm>
//where W: OperatorWorker, Pg: ProcGroup<Pid=usize> {
where W: OperatorWorker,
      Comm: 'static + CommWorker,
  fn batch_size(&self) -> usize {
    // FIXME(20160402)
    unimplemented!();
  }

  fn init_params(&mut self, shared_seed: [u64; 2]) {
    self.inner_worker.init_params(shared_seed);
  }

  fn read_params(&mut self, blob: &[u8]) -> usize {
    self.inner_worker.read_params(blob)
  }

  fn write_params(&mut self, blob: &mut Vec<u8>) {
    self.inner_worker.write_params(blob)
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    self.inner_worker.forward(batch_size, phase);
  }

  fn backward(&mut self, batch_size: usize) {
    self.inner_worker.backward(batch_size);
  }

  fn regularize(&mut self, reg: Regularization) {
    self.inner_worker.regularize(reg);
  }

  fn accumulate_grads(&mut self, scale: f32, momentum: f32) {
    self.inner_worker.accumulate_grads(scale, momentum);
  }

  fn update_params(&mut self, scale: f32) {
    self.inner_worker.update_params(scale);
  }

  /*fn reset_params(&mut self, momentum: f32, nesterov: bool) {
    self.inner_worker.reset_params(momentum, nesterov);
  }*/

  fn save_params(&mut self) {
    // FIXME(20160406)
    unimplemented!();
  }

  fn restore_params(&mut self) {
    // FIXME(20160406)
    unimplemented!();
  }

  fn set_grads_with_params_diff(&mut self) {
    // FIXME(20160406)
    unimplemented!();
  }

  fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    // XXX(20160412): First, implicitly sync the inner worker parameters.
    self.inner_worker.stage_params();
  }

  fn sync_params(&mut self) {
    self.inner_worker.sync_params();
    /*// FIXME(20160402): additionally sync across nodes, and wait on a local
    // barrier b/w device workers.
    if self.inner_worker.worker_rank() == 0 {
      unimplemented!();
    }
    self.inner_worker.wait_barrier();*/
  }

  fn reset_grads(&mut self, scale: f32) {
    self.inner_worker.reset_grads(scale);
  }

  fn reset(&mut self) {
    self.inner_worker.reset();
  }
}*/
