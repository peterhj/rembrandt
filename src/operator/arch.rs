use comm::{CommWorker};
use operator::{
  Operator, InputOperator, LossOperator,
  OpNode, OperatorConfig,
  OpMode, OpPhase,
  Data3dOperatorConfig,
  AffineOperatorConfig,
  Conv2dOperatorConfig,
  CategoricalLossConfig,
};

use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use worker::{WorkerData};

use std::cell::{RefCell};
use std::collections::{HashSet};
use std::iter::{FromIterator, repeat};
use std::rc::{Rc};

pub struct PipelineOperatorConfig {
  pub input_op:     Option<OperatorConfig>,
  pub hidden_ops:   Vec<OperatorConfig>,
  pub loss_op:      Option<OperatorConfig>,
}

impl PipelineOperatorConfig {
  pub fn data3d(&mut self, cfg: Data3dOperatorConfig) {
  }

  pub fn affine(&mut self, cfg: AffineOperatorConfig) {
  }

  pub fn conv2d(&mut self, cfg: Conv2dOperatorConfig) {
  }

  pub fn softmax_kl_loss(&mut self, cfg: CategoricalLossConfig) {
  }
}

pub struct PipelineOperatorWorkerBuilder {
  config:   PipelineOperatorConfig,
  mode:     OpMode,
}

impl PipelineOperatorWorkerBuilder {
  pub fn into_worker<Comm>(self, context: Rc<DeviceContext>, comm_worker: Comm) -> PipelineOperatorWorker<Comm>
  where Comm: CommWorker {
    unimplemented!();
  }
}

pub struct PipelineOperatorWorker<Comm> where Comm: CommWorker {
  worker_data:  WorkerData,
  config:       PipelineOperatorConfig,
  shared_seed:  [u64; 2],

  context:      Rc<DeviceContext>,
  comm_worker:  Rc<RefCell<Comm>>,
  input_op:     Box<InputOperator>,
  hidden_ops:   Vec<Box<Operator>>,
  loss_op:      Box<LossOperator>,
}

impl<Comm> PipelineOperatorWorker<Comm> where Comm: CommWorker {
  pub fn get_shared_seed(&self) -> [u64; 2] {
    self.shared_seed
  }
}

impl<Comm> Operator for PipelineOperatorWorker<Comm> where Comm: CommWorker {
  fn init_params(&mut self, shared_seed: [u64; 2]) {
    for r in 0 .. self.hidden_ops.len() {
      self.hidden_ops[r].init_params(shared_seed);
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
      self.save_params(blob);
    }
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    self.input_op.forward(batch_size, phase);
    for r in 0 .. self.hidden_ops.len() {
      self.hidden_ops[r].forward(batch_size, phase);
    }
    self.loss_op.forward(batch_size, phase);
  }

  fn backward(&mut self, batch_size: usize) {
    self.loss_op.backward(batch_size);
    for r in (0 .. self.hidden_ops.len()).rev() {
      self.hidden_ops[r].backward(batch_size);
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

  fn sync_params(&mut self) {
    for r in 0 .. self.hidden_ops.len() {
      self.hidden_ops[r].sync_params();
    }
  }

  fn reset_grads(&mut self, scale: f32) {
    for r in 0 .. self.hidden_ops.len() {
      self.hidden_ops[r].reset_grads(scale);
    }
  }
}

pub struct GraphOperatorConfig {
  pub ops:  Vec<(Vec<usize>, OperatorConfig)>,
}

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

pub struct GraphOperatorWorkerBuilder {
  op_nodes:     Vec<OperatorConfig>,
  in_node_ids:  Vec<Vec<usize>>,
  out_node_ids: Vec<Vec<usize>>,
  top_order:    Vec<usize>,
}

impl GraphOperatorWorkerBuilder {
  pub fn new(config: GraphOperatorConfig) -> GraphOperatorWorkerBuilder {
    let mut op_nodes = vec![];
    let num_ops = config.ops.len();
    let mut in_node_ids = vec![];
    let mut out_node_ids = vec![];
    for _ in 0 .. num_ops {
      in_node_ids.push(vec![]);
      out_node_ids.push(vec![]);
    }
    for (curr_id, &(ref prev_ids, ref op_cfg)) in config.ops.iter().enumerate() {
      op_nodes.push(*op_cfg);
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
        // Insert a `SplitOperator`.
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
    // FIXME(20160329): Sort the graph into topological order.
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
}

pub struct GraphOperatorWorker<Comm> where Comm: CommWorker {
  worker_data:  WorkerData,
  comm_worker:  Comm,
}
