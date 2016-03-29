use comm::{CommWorker};
use operator::{
  Operator, InputOperator, LossOperator,
  OpMode, OpPhase,
  OperatorConfig,
  Data3dOperatorConfig,
  AffineOperatorConfig,
  Conv2dOperatorConfig,
  CategoricalLossConfig,
};

use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use worker::{WorkerData};

use std::cell::{RefCell};
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

  pub fn softmax_kl(&mut self, cfg: CategoricalLossConfig) {
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

  fn descend_params(&mut self, step_size: f32, l2_reg_coef: f32) {
    for r in 0 .. self.hidden_ops.len() {
      self.hidden_ops[r].descend_params(step_size, l2_reg_coef);
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
  pub fn data3d(&mut self, cfg: Data3dOperatorConfig) {
    let curr_id = self.ops.len();
  }

  pub fn affine(&mut self, prev_ids: Vec<usize>, cfg: AffineOperatorConfig) {
    let curr_id = self.ops.len();
  }

  pub fn conv2d(&mut self, prev_ids: Vec<usize>, cfg: Conv2dOperatorConfig) {
    let curr_id = self.ops.len();
  }

  pub fn softmax_kl(&mut self, prev_ids: Vec<usize>, cfg: CategoricalLossConfig) {
    let curr_id = self.ops.len();
  }
}

enum GraphOpNode {
  Hidden(Box<Operator>),
  Input(Box<InputOperator>),
  Loss(Box<LossOperator>),
  Struct(Box<Operator>),
}

pub struct GraphOperatorWorkerBuilder {
  op_nodes:     Vec<GraphOpNode>,
  top_order:    Vec<usize>,
}

impl GraphOperatorWorkerBuilder {
  pub fn new(config: GraphOperatorConfig) -> GraphOperatorWorkerBuilder {
    //let mut in_nodes = vec![];
    //let mut out_nodes = vec![];
    unimplemented!();
  }
}

pub struct GraphOperatorWorker<Comm> where Comm: CommWorker {
  worker_data:  WorkerData,
  comm_worker:  Comm,
}
