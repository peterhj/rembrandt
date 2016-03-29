use comm::{CommWorker};

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceCtxRef};
use array_cuda::device::memory::{DeviceBuffer};

use std::cell::{RefCell};
use std::rc::{Rc};

pub trait Operator {
  fn forward(&mut self, batch_size: usize, phase: OpPhase, ctx: &DeviceCtxRef);

  // Requires `Backward` mode.
  fn backward(&mut self, batch_size: usize, ctx: &DeviceCtxRef);
  fn descend_params(&mut self, step_size: f32, ctx: &DeviceCtxRef);
  fn sync_params(&mut self, comm_worker: &mut CommWorker, ctx: &DeviceCtxRef);

  // Requires `HvBackward` mode.
  fn hv_reset_direction(&mut self, _init: HvDirectionInit, _ctx: &DeviceCtxRef) { unimplemented!(); }
  fn hv_forward(&mut self, _batch_size: usize, _ctx: &DeviceCtxRef) { unimplemented!(); }
  fn hv_backward(&mut self, _batch_size: usize, _ctx: &DeviceCtxRef) { unimplemented!(); }
  fn hv_descend_params(&mut self, step_size: f32, ctx: &DeviceCtxRef);
}

pub trait InputOperator: Operator {
}

pub trait LossOperator: Operator {
}

pub enum OperatorConfig {
  Data3d(Data3dOperatorConfig),
  Affine(AffineOperatorConfig),
  Conv2d(Conv2dOperatorConfig),
  SoftmaxKLLoss(CategoricalLossConfig),
}

pub enum OpMode {
  Forward,
  Backward,
  HvBackward,
}

pub enum OpPhase {
  Inference,
  Training,
}

pub enum HvDirectionInit {
  Gradient,
}

pub enum ActivationFunction {
  Identity,
  Rect,
  Logistic,
  Tanh,
}

pub enum ParamsInitialization {
  Disabled,
  Normal{std: f32},
  Uniform{half_range: f32},
}

pub type SharedDeviceBuf = Rc<RefCell<DeviceBuffer<f32>>>;

pub struct Data3dOperatorConfig {
  pub dims: (usize, usize, usize),
  pub normalize:    bool,
}

pub struct Data3dOperator;

pub enum AffineBackend {
  CublasGemm,
}

pub struct AffineOperatorConfig {
  pub in_channels:  usize,
  pub out_channels: usize,
  pub act_func:     ActivationFunction,
  pub weights_init: ParamsInitialization,
  pub backend:      CublasGemm,
}

pub struct AffineOperator {
  batch_cap:    usize,
  config:       AffineOperatorConfig,

  in_act:       SharedDeviceBuf,
  in_delta:     SharedDeviceBuf,
  out_act:      SharedDeviceBuf,
  out_delta:    SharedDeviceBuf,

  weights:      DeviceArray2d<f32>,
  bias:         DeviceArray2d<f32>,

  backward:     Option<AffineBwdOperator>,
  hv_backward:  Option<AffineHvBwdOperator>,
}

struct AffineBwdOperator {
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,
}

struct AffineHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
  dir_bias:     DeviceArray2d<f32>,
}

pub enum Conv2dBackend {
  CudnnImplicitPrecompGemm,
  CudnnFftTiling,
}

pub struct Conv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub out_channels: usize,
  pub conv_size:    usize,
  pub conv_stride:  usize,
  pub conv_pad:     usize,
  pub act_func:     ActivationFunction,
  pub weights_init: ParamsInitialization,
  pub backend:      Conv2dBackend,
}

pub struct Conv2dOperator {
  batch_cap:    usize,
  config:       Conv2dOperatorConfig,

  in_act:       SharedDeviceBuf,
  in_delta:     SharedDeviceBuf,
  out_act:      SharedDeviceBuf,
  out_delta:    SharedDeviceBuf,

  weights:      DeviceArray2d<f32>,
  bias:         DeviceArray2d<f32>,

  backward:     Option<Conv2dBwdOperator>,
  hv_backward:  Option<Conv2dHvBwdOperator>,
}

struct Conv2dBwdOperator {
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,
}

struct Conv2dHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
  dir_bias:     DeviceArray2d<f32>,
}

pub struct CategoricalLossConfig {
  pub category_count:   usize,
}

pub struct SoftmaxKLLossOperator {
  batch_cap:    usize,
  loss_config:  CategoricalLossConfig,

  in_act:       SharedDeviceBuf,
  in_delta:     SharedDeviceBuf,

  out_values:   DeviceBuffer<f32>,
}
