use operator::{
  Operator, InputOperator, LossOperator,
  OperatorNode, OperatorConfig,
  OpCapability, OpPhase,
  Regularization,
  AffineOperatorConfig,
  Conv2dOperatorConfig,
  Pool2dOperatorConfig,
  DropoutOperatorConfig,
};
use operator::conv::{
  BNormConv2dOperatorConfig,
  StackResConv2dOperatorConfig,
  ProjStackResConv2dOperatorConfig,
  BotResConv2dOperatorConfig,
  ProjBotResConv2dOperatorConfig,
};
use operator::input::{
  Data3dOperatorConfig,
  VarData3dOperatorConfig,
};
use operator::loss::{
  CategoricalLossConfig,
};

use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, SeedableRng, thread_rng};
use std::rc::{Rc};

pub struct SequentialOperatorConfig {
  pub input_op:     Option<OperatorConfig>,
  pub hidden_ops:   Vec<OperatorConfig>,
  pub loss_op:      Option<OperatorConfig>,
}

impl SequentialOperatorConfig {
  pub fn new() -> SequentialOperatorConfig {
    SequentialOperatorConfig{
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

  pub fn var_data3d(&mut self, cfg: VarData3dOperatorConfig) -> &mut Self {
    self.input_op = Some(OperatorConfig::VarData3d(cfg));
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

  pub fn bnorm_conv2d(&mut self, cfg: BNormConv2dOperatorConfig) -> &mut Self  {
    self.hidden_ops.push(OperatorConfig::BNormConv2d(cfg));
    self
  }

  pub fn stack_res_conv2d(&mut self, cfg: StackResConv2dOperatorConfig) -> &mut Self {
    self.hidden_ops.push(OperatorConfig::StackResConv2d(cfg));
    self
  }

  pub fn proj_stack_res_conv2d(&mut self, cfg: ProjStackResConv2dOperatorConfig) -> &mut Self {
    self.hidden_ops.push(OperatorConfig::ProjStackResConv2d(cfg));
    self
  }

  pub fn bot_res_conv2d(&mut self, cfg: BotResConv2dOperatorConfig) -> &mut Self {
    // FIXME(20160418)
    unimplemented!();
  }

  pub fn proj_bot_res_conv2d(&mut self, cfg: ProjBotResConv2dOperatorConfig) -> &mut Self {
    // FIXME(20160418)
    unimplemented!();
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

pub struct SequentialOperator {
  batch_size:   usize,
  config:       SequentialOperatorConfig,
  capability:   OpCapability,
  shared_seed:  [u64; 2],

  local_rng:    Xorshiftplus128Rng,
  context:      Rc<DeviceContext>,
  input_op:     Box<InputOperator>,
  hidden_ops:   Vec<Box<Operator>>,
  loss_op:      Box<LossOperator>,
}

impl SequentialOperator {
  pub fn new(batch_size: usize, config: SequentialOperatorConfig, capability: OpCapability, shared_seed: [u64; 2], context: Rc<DeviceContext>) -> SequentialOperator {
    let total_params_len = config.params_len();
    let input_op = config.input_op.as_ref().unwrap().build_input_operator(batch_size, context.clone());
    let mut hidden_ops: Vec<Box<Operator>> = vec![];
    let mut params_off = 0;
    for r in 0 .. config.hidden_ops.len() {
      let hidden_op = {
        let prev_op = match r {
          0 => input_op.downcast(),
          _ => &*hidden_ops[r-1],
        };
        config.hidden_ops[r].build_operator(batch_size, capability, params_off, Some(prev_op), /*Some(comm_worker.clone()),*/ context.clone())
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
      config.loss_op.as_ref().unwrap().build_loss_operator(batch_size, Some(prev_op), context.clone())
    };

    SequentialOperator{
      batch_size:   batch_size,
      config:       config,
      capability:   capability,
      shared_seed:  shared_seed,
      local_rng:    Xorshiftplus128Rng::new(&mut thread_rng()),
      context:      context,
      input_op:     input_op,
      hidden_ops:   hidden_ops,
      loss_op:      loss_op,
    }
  }
}

impl Operator for SequentialOperator {
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

  fn decode_params(&mut self, blob: &[u8]) -> usize {
    let mut offset = 0;
    for op in self.hidden_ops.iter_mut() {
      offset += op.decode_params(&blob[offset .. ]);
    }
    offset
  }

  fn encode_params(&mut self, blob: &mut Vec<u8>) {
    for op in self.hidden_ops.iter_mut() {
      op.encode_params(blob);
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

  fn update_params2(&mut self, grad_scale: f32, update_scale: f32) {
    for op in self.hidden_ops.iter_mut() {
      op.update_params2(grad_scale, update_scale);
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

  fn reset(&mut self) {
    for op in self.hidden_ops.iter_mut() {
      op.reset();
    }
  }
}
