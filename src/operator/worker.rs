use operator::{
  Operator, InputOperator, LossOperator,
  OperatorNode, OperatorConfig,
  OpCapability, OpPhase,
  Data3dOperatorConfig,
  AffineOperatorConfig,
  Conv2dOperatorConfig,
  Pool2dOperatorConfig,
  CategoricalLossConfig,
};
use operator::comm::{CommWorker};

use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use rng::xorshift::{Xorshiftplus128Rng};
use worker::{WorkerData};

use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell};
use std::collections::{HashSet};
use std::iter::{FromIterator, repeat};
use std::rc::{Rc};

pub trait OperatorWorker: Operator {
  fn num_workers(&self) -> usize;
  fn tid(&self) -> usize;
  fn shared_seed(&self) -> [u64; 2];
  fn input_operator(&mut self) -> &mut InputOperator;
  fn loss_count(&self) -> usize;
  fn loss_operator(&mut self, rank: usize) -> &mut LossOperator;
}

// FIXME(20160330): why can't I derive Clone here?
//#[derive(Clone)]
pub struct PipelineOperatorWorkerConfig<Comm> where Comm: 'static + CommWorker {
  pub input_op:     Option<OperatorConfig<Comm>>,
  pub hidden_ops:   Vec<OperatorConfig<Comm>>,
  pub loss_op:      Option<OperatorConfig<Comm>>,
}

impl<Comm> Clone for PipelineOperatorWorkerConfig<Comm> where Comm: 'static + CommWorker {
  fn clone(&self) -> PipelineOperatorWorkerConfig<Comm> {
    PipelineOperatorWorkerConfig{
      input_op:     self.input_op.clone(),
      hidden_ops:   self.hidden_ops.clone(),
      loss_op:      self.loss_op.clone(),
    }
  }
}

impl<Comm> PipelineOperatorWorkerConfig<Comm> where Comm: 'static + CommWorker {
  pub fn new() -> PipelineOperatorWorkerConfig<Comm> {
    PipelineOperatorWorkerConfig{
      input_op:     None,
      hidden_ops:   vec![],
      loss_op:      None,
    }
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

  pub fn softmax_kl_loss(&mut self, cfg: CategoricalLossConfig) -> &mut Self  {
    self.loss_op = Some(OperatorConfig::SoftmaxKLLoss(cfg));
    self
  }
}

pub struct PipelineOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
  num_workers:  usize,
  batch_size:   usize,
  config:       PipelineOperatorWorkerConfig<Comm>,
  capability:   OpCapability,
  shared_seed:  [u64; 2],
}

impl<Comm> Clone for PipelineOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
  fn clone(&self) -> PipelineOperatorWorkerBuilder<Comm> {
    PipelineOperatorWorkerBuilder{
      num_workers:  self.num_workers,
      batch_size:   self.batch_size,
      config:       self.config.clone(),
      capability:   self.capability,
      shared_seed:  self.shared_seed,
    }
  }
}

impl<Comm> PipelineOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
  pub fn new(num_workers: usize, batch_size: usize, config: PipelineOperatorWorkerConfig<Comm>, capability: OpCapability) -> PipelineOperatorWorkerBuilder<Comm> {
    PipelineOperatorWorkerBuilder{
      num_workers:  num_workers,
      batch_size:   batch_size,
      config:       config,
      capability:   capability,
      shared_seed:  [thread_rng().next_u64(), thread_rng().next_u64()],
    }
  }

  pub fn into_worker(self, tid: usize, context: Rc<DeviceContext>, comm_worker: Rc<RefCell<Comm>>) -> PipelineOperatorWorker<Comm> {
    let config = self.config.clone();
    let input_op = config.input_op.unwrap().build_input_operator(self.batch_size, context.clone());
    let mut hidden_ops: Vec<Box<Operator>> = vec![];
    for r in 0 .. config.hidden_ops.len() {
      let hidden_op = {
        let prev_op = match r {
          0 => input_op.downcast(),
          _ => &*hidden_ops[r-1],
        };
        config.hidden_ops[r].build_operator(self.batch_size, self.capability, Some(prev_op), Some(comm_worker.clone()), context.clone())
      };
      hidden_ops.push(hidden_op);
    }
    let loss_op = {
      let num_hidden_ops = hidden_ops.len();
      let prev_op = match num_hidden_ops {
        0 => input_op.downcast(),
        _ => &*hidden_ops[num_hidden_ops-1],
      };
      config.loss_op.unwrap().build_loss_operator(self.batch_size, Some(prev_op), context.clone())
    };
    PipelineOperatorWorker{
      worker_data:  WorkerData::new(self.num_workers, tid),
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
  config:       PipelineOperatorWorkerConfig<Comm>,
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

  fn tid(&self) -> usize {
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
}

impl<Comm> Operator for PipelineOperatorWorker<Comm> where Comm: 'static + CommWorker {
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn init_params(&mut self, shared_seed: [u64; 2]) {
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    for r in 0 .. self.hidden_ops.len() {
      let op_seed = [rng.next_u64(), rng.next_u64()];
      self.hidden_ops[r].init_params(op_seed);
    }
  }

  fn load_params(&mut self, blob: &[u8]) -> usize {
    let mut offset = 0;
    for r in 0 .. self.hidden_ops.len() {
      offset += self.hidden_ops[r].load_params(&blob[offset .. ]);
    }
    offset
  }

  fn save_params(&mut self, blob: &mut Vec<u8>) {
    for r in 0 .. self.hidden_ops.len() {
      self.hidden_ops[r].save_params(blob);
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

  fn update_params(&mut self, step_size: f32, l2_reg_coef: f32) {
    for r in 0 .. self.hidden_ops.len() {
      self.hidden_ops[r].update_params(step_size, l2_reg_coef);
    }
  }

  fn sync_grads(&mut self) {
    for r in 0 .. self.hidden_ops.len() {
      self.hidden_ops[r].sync_grads();
    }
  }

  fn stage_params(&mut self) {
    for op in self.hidden_ops.iter_mut() {
      op.stage_params();
    }
  }

  fn sync_params(&mut self) {
    let ctx = &(*self.context).as_ref();
    self.comm_worker.borrow_mut().communicate(ctx);
    for op in self.hidden_ops.iter_mut() {
      op.sync_params();
    }
  }

  fn reset_grads(&mut self, scale: f32) {
    for r in 0 .. self.hidden_ops.len() {
      self.hidden_ops[r].reset_grads(scale);
    }
  }
}

pub struct GraphOperatorConfig<Comm> {
  pub ops:  Vec<(Vec<usize>, OperatorConfig<Comm>)>,
}

impl<Comm> GraphOperatorConfig<Comm> where Comm: CommWorker {
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

pub struct GraphOperatorWorkerBuilder<Comm> {
  op_nodes:     Vec<OperatorConfig<Comm>>,
  in_node_ids:  Vec<Vec<usize>>,
  out_node_ids: Vec<Vec<usize>>,
  top_order:    Vec<usize>,
}

impl<Comm> GraphOperatorWorkerBuilder<Comm> where Comm: 'static + CommWorker {
  pub fn new(config: GraphOperatorConfig<Comm>) -> GraphOperatorWorkerBuilder<Comm> {
    let mut op_nodes: Vec<OperatorConfig<Comm>> = vec![];
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
