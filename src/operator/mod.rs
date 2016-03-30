use data_new::{SampleLabel};
use operator::comm::{CommWorker};

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::ext::{DeviceCastBytesExt, DeviceNumExt};
use array_cuda::device::linalg::{BlasMatrixExt, BlasVectorExt, Transpose};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer};
use array_new::{Shape, Array, AsyncArray, ArrayView, ArrayViewMut, ArrayZeroExt, Array2d};
use cuda_dnn::v4::{
  CudnnConvFwdOp, CudnnConvBwdFilterOp, CudnnConvBwdDataOp,
  CudnnAddOp, CudnnActKind, CudnnActOp, CudnnSoftmaxOp,
  CudnnTensorDesc, CudnnFilterDesc, CudnnConvDesc,
};
use cuda_dnn::v4::ffi::{cudnnConvolutionFwdAlgo_t};
use rembrandt_kernels::ffi::*;

use std::cell::{RefCell};
use std::cmp::{max};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::rc::{Rc};

pub mod arch;
pub mod comm;

pub trait Operator {
  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> { None }
  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> { None }
  fn init_params(&mut self, _shared_seed: [u64; 2]) {}
  fn load_params(&mut self, _blob: &[u8]) -> usize { 0 }
  fn save_params(&mut self, _blob: &mut Vec<u8>) {}
  fn forward(&mut self, batch_size: usize, phase: OpPhase);

  // Requires `Backward` capability.
  fn backward(&mut self, batch_size: usize);
  fn update_params(&mut self, step_size: f32, l2_reg_coef: f32);
  fn sync_grads(&mut self);
  fn sync_params(&mut self);
  fn reset_grads(&mut self, scale: f32);

  // Requires `HvBackward` capability.
  fn hv_reset_direction(&mut self, _init: HvDirectionInit) { unimplemented!(); }
  fn hv_solve_direction(&mut self, _solver: HvDirectionSolver) { unimplemented!(); }
  fn hv_forward(&mut self, _batch_size: usize) { unimplemented!(); }
  fn hv_backward(&mut self, _batch_size: usize) { unimplemented!(); }
  fn hv_update_params(&mut self, _step_size: f32) { unimplemented!(); }
}

pub trait InputOperator: Operator {
  fn downcast(&self) -> &Operator;
  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8];
  fn load_frames(&mut self, batch_size: usize);
}

pub trait LossOperator: Operator {
  fn downcast(&self) -> &Operator;
  fn stage_label(&mut self, batch_idx: usize, label: &SampleLabel);
  fn load_labels(&mut self, batch_size: usize);
  fn stage_weight(&mut self, batch_idx: usize, weight: f32);
  fn load_weights(&mut self, batch_size: usize);
  fn store_output_values(&mut self, batch_size: usize);
  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32>;
  fn store_output_categories(&mut self, batch_size: usize);
  fn get_output_categories(&self, batch_size: usize) -> &Array2d<i32>;
  //fn reset_loss(&mut self);
}

pub enum OperatorNode {
  Hidden(Box<Operator>),
  Input(Box<InputOperator>),
  Loss(Box<LossOperator>),
  Split(Box<Operator>),
  //Join(Box<Operator>),
}

pub enum OperatorConfig<Comm> {
  Data3d(Data3dOperatorConfig),
  Affine(AffineOperatorConfig),
  Conv2d(Conv2dOperatorConfig),
  Pool2d(Pool2dOperatorConfig),
  SoftmaxKLLoss(CategoricalLossConfig),
  Split((usize, usize, usize)),
  //Join((usize, usize, usize)),
  _Dummy(PhantomData<Comm>),
}

impl<Comm> Clone for OperatorConfig<Comm> where Comm: 'static + CommWorker {
  fn clone(&self) -> OperatorConfig<Comm> {
    match self {
      &OperatorConfig::Data3d(cfg)        => OperatorConfig::Data3d(cfg),
      &OperatorConfig::Affine(cfg)        => OperatorConfig::Affine(cfg),
      &OperatorConfig::Conv2d(cfg)        => OperatorConfig::Conv2d(cfg),
      &OperatorConfig::Pool2d(cfg)        => OperatorConfig::Pool2d(cfg),
      &OperatorConfig::SoftmaxKLLoss(cfg) => OperatorConfig::SoftmaxKLLoss(cfg),
      &OperatorConfig::Split(cfg)         => OperatorConfig::Split(cfg),
      &OperatorConfig::_Dummy(marker)     => OperatorConfig::_Dummy(PhantomData),
    }
  }
}

impl<Comm> OperatorConfig<Comm> where Comm: 'static + CommWorker {
  pub fn build_node(&self, batch_size: usize, capability: OpCapability, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> OperatorNode {
    match self {
      &OperatorConfig::Affine(ref cfg) => {
        OperatorNode::Hidden(Box::new(AffineOperator::new(batch_size, capability, *cfg, prev_op, comm_worker, context)))
      }
      &OperatorConfig::Conv2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(Conv2dOperator::new(batch_size, capability, *cfg, prev_op, comm_worker, context)))
      }
      &OperatorConfig::Data3d(ref cfg) => {
        OperatorNode::Input(Box::new(Data3dOperator::new(batch_size, *cfg, context)))
      }
      &OperatorConfig::SoftmaxKLLoss(ref cfg) => {
        OperatorNode::Loss(Box::new(SoftmaxKLLossOperator::new(batch_size, *cfg, prev_op, context)))
      }
      &OperatorConfig::Split(dims) => {
        unimplemented!();
      }
      /*&OperatorConfig::Join(dims) => {
        unimplemented!();
      }*/
      _ => unreachable!(),
    }
  }

  pub fn build_operator(&self, batch_size: usize, capability: OpCapability, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> Box<Operator> {
    match self.build_node(batch_size, capability, prev_op, comm_worker, context) {
      OperatorNode::Hidden(op) => op,
      _ => unimplemented!(),
    }
  }

  pub fn build_input_operator(&self, batch_size: usize, context: Rc<DeviceContext>) -> Box<InputOperator> {
    match self.build_node(batch_size, OpCapability::Forward, None, None, context) {
      OperatorNode::Input(op) => op,
      _ => unimplemented!(),
    }
  }

  pub fn build_loss_operator(&self, batch_size: usize, prev_op: Option<&Operator>, context: Rc<DeviceContext>) -> Box<LossOperator> {
    // FIXME(20160330): set proper `OpCapability`.
    match self.build_node(batch_size, OpCapability::Backward, prev_op, None, context) {
      OperatorNode::Loss(op) => op,
      _ => unimplemented!(),
    }
  }
}

#[derive(Clone, Copy)]
pub enum OpCapability {
  Forward,
  Backward,
  HvBackward,
}

impl OpCapability {
  pub fn backward_enabled(&self) -> bool {
    match *self {
      OpCapability::Forward     => false,
      OpCapability::Backward    => true,
      OpCapability::HvBackward  => true,
    }
  }

  pub fn hv_backward_enabled(&self) -> bool {
    match *self {
      OpCapability::Forward     => false,
      OpCapability::Backward    => false,
      OpCapability::HvBackward  => true,
    }
  }
}

#[derive(Clone, Copy)]
pub enum OpPhase {
  Inference,
  Training,
}

#[derive(Clone, Copy)]
pub enum HvDirectionInit {
  Gradient,
}

#[derive(Clone, Copy)]
pub enum HvDirectionSolver {
  PrecondConjGrad,
}

#[derive(Clone, Copy)]
pub enum ActivationFunction {
  Identity,
  Rect,
  Logistic,
  Tanh,
}

#[derive(Clone, Copy)]
pub enum ParamsInit {
  Disabled,
  Normal{std: f32},
  Uniform{half_range: f32},
}

#[derive(Clone, Copy)]
pub enum PoolOperation {
  Max,
  Average,
}

pub type SharedDeviceBuf<T> = Rc<RefCell<DeviceBuffer<T>>>;

pub struct SplitOperator {
  batch_cap:    usize,
  in_dims:      (usize, usize, usize),

  context:      Rc<DeviceContext>,

  shared_act:   SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_deltas:   Vec<SharedDeviceBuf<f32>>,
}

impl SplitOperator {
  pub fn new(batch_size: usize, in_dims: (usize, usize, usize), prev_op: Option<&Operator>, context: Rc<DeviceContext>) -> SplitOperator {
    SplitOperator{
      batch_cap:    batch_size,
      in_dims:      in_dims,
      context:      context,
      shared_act:   match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("SplitOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_deltas:   vec![],
    }
  }

  pub fn try_push_output_deltas(&mut self) {
    if self.in_delta.is_some() {
      let ctx = &(*self.context).as_ref();
      self.out_deltas.push(
          Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(self.in_dims.len() * self.batch_cap, ctx)))
      );
    }
  }
}

impl Operator for SplitOperator {
  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.shared_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    if self.in_delta.is_some() {
      assert!(self.out_deltas.len() >= 1);
      Some(self.out_deltas[self.out_deltas.len()-1].clone())
    } else {
      None
    }
  }

  fn forward(&mut self, _batch_size: usize, _phase: OpPhase) {
    // Do nothing.
  }

  fn backward(&mut self, batch_size: usize) {
    if let Some(in_delta) = self.in_delta.as_ref() {
      let ctx = &(*self.context).as_ref();
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);
      self.out_deltas[0].borrow_mut().as_ref(ctx)
        .send(&mut in_delta);
      for r in 1 .. self.out_deltas.len() {
        in_delta.row_vector_sum(1.0, &self.out_deltas[r].borrow_mut().as_ref(ctx));
      }
    }
  }

  fn update_params(&mut self, _step_size: f32, _l2_reg_coef: f32) {
    // Do nothing.
  }

  fn sync_grads(&mut self) {
    // Do nothing.
  }

  fn sync_params(&mut self) {
    // Do nothing.
  }

  fn reset_grads(&mut self, _scale: f32) {
    // Do nothing.
  }
}

pub struct JoinOperator;

#[derive(Clone, Copy)]
pub struct Data3dOperatorConfig {
  pub dims: (usize, usize, usize),
  pub normalize:    bool,
}

pub struct Data3dOperator {
  batch_cap:    usize,
  config:       Data3dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_buf_h:     Vec<u8>,
  in_buf:       DeviceBuffer<u8>,
  out_buf:      SharedDeviceBuf<f32>,
}

impl Data3dOperator {
  pub fn new(batch_size: usize, config: Data3dOperatorConfig, context: Rc<DeviceContext>) -> Data3dOperator {
    let ctx = &(*context).as_ref();
    let frame_len = config.dims.len();
    Data3dOperator{
      batch_cap:    batch_size,
      config:       config,
      context:      context.clone(),
      in_buf_h:     repeat(0).take(batch_size * frame_len).collect(),
      in_buf:       DeviceBuffer::zeros(batch_size * frame_len, ctx),
      out_buf:      Rc::new(RefCell::new(DeviceBuffer::zeros(batch_size * frame_len, ctx))),
    }
  }
}

impl Operator for Data3dOperator {
  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_buf.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    //let length = self.config.dims.len();
    let in_buf = self.in_buf.as_ref(ctx);
    let mut out_buf = self.out_buf.borrow_mut().as_ref_mut(ctx);
    if self.config.normalize {
      in_buf.cast_bytes_normalized(&mut out_buf);
    } else {
      in_buf.cast_bytes(&mut out_buf);
    }
  }

  fn backward(&mut self, _batch_size: usize) {
    // Do nothing.
  }

  fn update_params(&mut self, _step_size: f32, _l2_reg_coef: f32) {
    // Do nothing.
  }

  fn sync_grads(&mut self) {
    // Do nothing.
  }

  fn sync_params(&mut self) {
    // Do nothing.
  }

  fn reset_grads(&mut self, _scale: f32) {
    // Do nothing.
  }
}

impl InputOperator for Data3dOperator {
  fn downcast(&self) -> &Operator {
    self
  }

  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    assert!(batch_idx < self.batch_cap);
    let frame_len = self.config.dims.len();
    &mut self.in_buf_h[batch_idx * frame_len .. (batch_idx + 1) * frame_len]
  }

  fn load_frames(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    {
      // FIXME(20160329): does not use `batch_size` at all!
      let in_buf_h = &self.in_buf_h;
      let mut in_buf = self.in_buf.as_ref_mut(ctx);
      in_buf.sync_load(in_buf_h);
    }
  }
}

#[derive(Clone, Copy)]
pub enum AffineBackend {
  CublasGemm,
}

#[derive(Clone, Copy)]
pub struct AffineOperatorConfig {
  pub in_channels:  usize,
  pub out_channels: usize,
  pub act_func:     ActivationFunction,
  pub weights_init: ParamsInit,
  pub backend:      AffineBackend,
}

impl AffineOperatorConfig {
}

pub struct AffineOperator<Comm> {
  batch_cap:    usize,
  _capability:  OpCapability,
  config:       AffineOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights:      DeviceArray2d<f32>,
  bias:         DeviceArray2d<f32>,

  add_bias:     CudnnAddOp,

  backward:     Option<AffineBwdOperator<Comm>>,
  hv_backward:  Option<AffineHvBwdOperator>,
}

struct AffineBwdOperator<Comm> {
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,

  unit_bias:    DeviceArray2d<f32>,

  comm_worker:  Rc<RefCell<Comm>>,
}

struct AffineHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
  dir_bias:     DeviceArray2d<f32>,
}

impl<Comm> AffineOperator<Comm> where Comm: CommWorker {
  pub fn new(batch_size: usize, capability: OpCapability, config: AffineOperatorConfig, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> AffineOperator<Comm> {
    let in_channels = config.in_channels;
    let out_channels = config.out_channels;

    let ctx = &(*context).as_ref();

    let backward = if capability.backward_enabled() {
      let mut unit_bias = DeviceArray2d::<f32>::zeros((1, batch_size), ctx);
      unit_bias.as_view_mut(ctx).set_constant(1.0);
      Some(AffineBwdOperator{
        grad_weights: DeviceArray2d::<f32>::zeros((in_channels, out_channels), ctx),
        grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        unit_bias:    unit_bias,
        comm_worker:  comm_worker.unwrap(),
      })
    } else {
      None
    };

    let add_bias = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, batch_size).unwrap(),
    );

    AffineOperator{
      batch_cap:    batch_size,
      _capability:  capability,
      config:       config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("AffineOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_channels * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_channels * batch_size, ctx))),
      weights:      DeviceArray2d::<f32>::zeros((in_channels, out_channels), ctx),
      bias:         DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      add_bias:     add_bias,
      backward:     backward,
      hv_backward:  None,
    }
  }
}

impl<Comm> Operator for AffineOperator<Comm> where Comm: CommWorker {
  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn init_params(&mut self, _shared_seed: [u64; 2]) {
    // FIXME(20160330)
    unimplemented!();
  }

  fn load_params(&mut self, _blob: &[u8]) -> usize {
    // FIXME(20160330)
    unimplemented!();
  }

  fn save_params(&mut self, _blob: &mut Vec<u8>) {
    // FIXME(20160330)
    unimplemented!();
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let in_channels = self.config.in_channels;
    let out_channels = self.config.out_channels;

    let &mut AffineOperator{
      ref context,
      ref mut in_act, ref mut out_act,
      ref mut weights, ref mut bias,
      .. } = self;

    let ctx = &(**context).as_ref();
    let weights = weights.as_view(ctx);
    let bias = bias.as_view(ctx);
    let in_act = in_act.borrow_mut().as_ref(ctx)
      .into_2d_view((in_channels, batch_size));
    let mut out_act = out_act.borrow_mut().as_ref_mut(ctx)
      .into_2d_view_mut((out_channels, batch_size));

    out_act.matrix_prod(1.0, &weights, Transpose::T, &in_act, Transpose::N, 0.0);

    self.add_bias.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias.forward(
        bias.as_ptr(),
        out_act.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_inplace(
            out_act.as_mut_ptr(),
            out_channels as i32,
            batch_size as i32,
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }
  }

  fn backward(&mut self, batch_size: usize) {
    assert!(self.backward.is_some());
    assert!(batch_size <= self.batch_cap);
    let in_channels = self.config.in_channels;
    let out_channels = self.config.out_channels;

    let &mut AffineOperator{
      ref context,
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights, ref mut bias,
      ref mut backward,
      .. } = self;
    let mut backward = backward.as_mut().unwrap();
    let &mut AffineBwdOperator{
      ref mut grad_weights, ref mut grad_bias,
      ref mut unit_bias,
      .. } = backward;

    let ctx = &(**context).as_ref();
    let weights = weights.as_view(ctx);
    let mut grad_weights = grad_weights.as_view_mut(ctx);
    let mut grad_bias = grad_bias.as_view_mut(ctx);
    let in_act = in_act.borrow_mut().as_ref(ctx)
      .into_2d_view((in_channels, batch_size));
    let out_act = out_act.borrow_mut().as_ref(ctx)
      .into_2d_view((out_channels, batch_size));
    let unit_bias = unit_bias.as_view(ctx);

    {
      let mut out_delta = out_delta.borrow_mut().as_ref_mut(ctx)
        .into_2d_view_mut((out_channels, batch_size));
      match self.config.act_func {
        ActivationFunction::Identity => {}
        ActivationFunction::Rect => {
          unsafe { rembrandt_kernel_batch_map_rect_backprop_inplace(
              out_act.as_ptr(),
              out_channels as i32,
              batch_size as i32,
              out_delta.as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
        _ => unimplemented!(),
      }
    }

    let out_delta = out_delta.borrow_mut().as_ref(ctx)
      .into_2d_view((out_channels, batch_size));
    grad_weights.matrix_prod(1.0, &in_act, Transpose::N, &out_delta, Transpose::T, 1.0);
    grad_bias.matrix_prod(1.0, &unit_bias, Transpose::N, &out_delta, Transpose::T, 1.0);

    if let &mut Some(ref mut in_delta) = in_delta {
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx)
        .into_2d_view_mut((in_channels, batch_size));
      in_delta.matrix_prod(1.0, &weights, Transpose::N, &out_delta, Transpose::N, 0.0);
    }
  }

  fn update_params(&mut self, step_size: f32, l2_reg_coef: f32) {
    assert!(self.backward.is_some());
    assert!(l2_reg_coef >= 0.0);
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    if l2_reg_coef > 0.0 {
      backward.grad_weights.as_view_mut(ctx)
        .matrix_sum(l2_reg_coef, &self.weights.as_view(ctx));
      backward.grad_bias.as_view_mut(ctx)
        .row_vector_sum(l2_reg_coef, &self.bias.as_view(ctx));
    }
    {
      self.weights.as_view_mut(ctx)
        .matrix_sum(-step_size, &backward.grad_weights.as_view(ctx));
      self.bias.as_view_mut(ctx)
        .row_vector_sum(-step_size, &backward.grad_bias.as_view(ctx));
    }
  }

  fn sync_grads(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_mut().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.communicate(&mut backward.grad_weights, ctx);
    comm_worker.communicate(&mut backward.grad_bias, ctx);
  }

  fn sync_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.communicate(&mut self.weights, ctx);
    comm_worker.communicate(&mut self.bias, ctx);
  }

  fn reset_grads(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.grad_weights.as_view_mut(ctx)
      .matrix_scale(scale);
    backward.grad_bias.as_view_mut(ctx)
      .row_vector_scale(scale);
  }
}

#[derive(Clone, Copy)]
pub enum Conv2dFwdBackend {
  CudnnFastest,
  CudnnImplicitPrecompGemm,
  CudnnFftTiling,
}

#[derive(Clone, Copy)]
pub enum Conv2dBwdBackend {
  CudnnFastest,
  CudnnDeterministic,
}

#[derive(Clone, Copy)]
pub struct Conv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub conv_size:    usize,
  pub conv_stride:  usize,
  pub conv_pad:     usize,
  pub out_channels: usize,
  pub act_func:     ActivationFunction,
  pub weights_init: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}

impl Conv2dOperatorConfig {
  fn get_out_dims(&self) -> (usize, usize, usize) {
    let (in_width, in_height, _) = self.in_dims;
    let out_width = max(0, (in_width + 2 * self.conv_pad - self.conv_size + self.conv_stride) as isize) as usize / self.conv_stride;
    let out_height = max(0, (in_height + 2 * self.conv_pad - self.conv_size + self.conv_stride) as isize) as usize / self.conv_stride;
    (out_width, out_height, self.out_channels)
  }

  fn params_len(&self) -> usize {
    let (_, _, in_channels) = self.in_dims;
    let weights_len = self.conv_size * self.conv_size * in_channels * self.out_channels;
    let bias_len = self.out_channels;
    weights_len + bias_len
  }
}

pub struct Conv2dOperator<Comm> {
  batch_cap:    usize,
  _capability:  OpCapability,
  config:       Conv2dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights:      DeviceArray2d<f32>,
  bias:         DeviceArray2d<f32>,

  workspace:    DeviceBuffer<u8>,
  conv_fwd:     CudnnConvFwdOp,
  add_bias:     CudnnAddOp,

  backward:     Option<Conv2dBwdOperator<Comm>>,
  hv_backward:  Option<Conv2dHvBwdOperator>,
}

struct Conv2dBwdOperator<Comm> {
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,

  conv_bwd_w:   CudnnConvBwdFilterOp,
  conv_bwd_d:   CudnnConvBwdDataOp,

  comm_worker:  Rc<RefCell<Comm>>,
}

struct Conv2dHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
  dir_bias:     DeviceArray2d<f32>,
}

impl<Comm> Conv2dOperator<Comm> where Comm: CommWorker {
  pub fn new(batch_size: usize, capability: OpCapability, config: Conv2dOperatorConfig, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> Conv2dOperator<Comm> {
    let Conv2dOperatorConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    let ctx = &(*context).as_ref();

    let mut workspace_size = 0;
    let fwd_algo = match config.fwd_backend {
      Conv2dFwdBackend::CudnnImplicitPrecompGemm => cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
      Conv2dFwdBackend::CudnnFftTiling           => cudnnConvolutionFwdAlgo_t::FftTiling,
      _ => unimplemented!(),
    };
    let conv_fwd = CudnnConvFwdOp::create_algo(
        fwd_algo,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv_fwd.work_size);

    let backward = if capability.backward_enabled() {
      let conv_bwd_w = CudnnConvBwdFilterOp::create_fastest(
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
          CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv_bwd_w.work_size);
      let conv_bwd_d = CudnnConvBwdDataOp::create_fastest(
          CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv_bwd_d.work_size);
      Some(Conv2dBwdOperator{
        grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        conv_bwd_w:   conv_bwd_w,
        conv_bwd_d:   conv_bwd_d,
        comm_worker:  comm_worker.unwrap(),
      })
    } else {
      None
    };

    let add_bias = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
    );

    Conv2dOperator{
      batch_cap:    batch_size,
      _capability:  capability,
      config:       config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("Conv2dOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      weights:      DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
      bias:         DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      workspace:    DeviceBuffer::<u8>::zeros(workspace_size, ctx),
      conv_fwd:     conv_fwd,
      add_bias:     add_bias,
      backward:     backward,
      hv_backward:  None,
    }
  }
}

impl<Comm> Operator for Conv2dOperator<Comm> where Comm: CommWorker {
  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn init_params(&mut self, _shared_seed: [u64; 2]) {
    // FIXME(20160330)
    unimplemented!();
  }

  fn load_params(&mut self, _blob: &[u8]) -> usize {
    // FIXME(20160330)
    unimplemented!();
  }

  fn save_params(&mut self, _blob: &mut Vec<u8>) {
    // FIXME(20160330)
    unimplemented!();
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    /*let Conv2dOperatorConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = self.config;*/
    //let (in_width, in_height, in_channels) = in_dims;
    //let in_length = in_dims.len();
    let out_dims = self.config.get_out_dims();
    //let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    let &mut Conv2dOperator{
      ref context,
      ref mut in_act, ref mut out_act,
      ref mut weights, ref mut bias,
      ref mut workspace,
      .. } = self;

    let ctx = &(**context).as_ref();
    let mut out_act = out_act.borrow_mut().as_ref_mut(ctx);

    self.conv_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv_fwd.forward(
        in_act.borrow_mut().as_ref(ctx).as_ptr(),
        weights.as_view(ctx).as_ptr(),
        out_act.as_mut_ptr(),
        workspace.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("conv2d forward failed: {:?}", e); }
    }
    self.add_bias.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias.forward(
        bias.as_view(ctx).as_ptr(),
        out_act.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_inplace(
            out_act.as_mut_ptr(),
            out_length as i32,
            batch_size as i32,
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }
  }

  fn backward(&mut self, batch_size: usize) {
    assert!(self.backward.is_some());
    assert!(batch_size <= self.batch_cap);
    /*let Conv2dOperatorConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = self.config;*/
    //let (in_width, in_height, in_channels) = in_dims;
    //let in_length = in_dims.len();
    let out_dims = self.config.get_out_dims();
    //let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    let &mut Conv2dOperator{
      ref context,
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights, //ref mut bias,
      ref mut workspace,
      ref mut backward,
      .. } = self;
    let mut backward = backward.as_mut().unwrap();
    let &mut Conv2dBwdOperator{
      ref mut grad_weights, ref mut grad_bias,
      .. } = backward;

    let ctx = &(**context).as_ref();
    let in_act = in_act.borrow_mut().as_ref(ctx);
    let out_act = out_act.borrow_mut().as_ref(ctx);
    let mut out_delta = out_delta.borrow_mut().as_ref_mut(ctx);
    let mut workspace = workspace.as_ref_mut(ctx);

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_backprop_inplace(
            out_act.as_ptr(),
            out_length as i32,
            batch_size as i32,
            out_delta.as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }

    backward.conv_bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { backward.conv_bwd_w.backward_filter(
        1.0,
        in_act.as_ptr(),
        out_delta.as_ptr(),
        grad_weights.as_view_mut(ctx).as_mut_ptr(),
        workspace.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    unsafe { backward.conv_bwd_w.backward_bias(
        1.0,
        out_delta.as_ptr(),
        grad_bias.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    if let &mut Some(ref mut in_delta) = in_delta {
      backward.conv_bwd_d.set_batch_size(batch_size).unwrap();
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);
      unsafe { backward.conv_bwd_d.backward_data(
          weights.as_view(ctx).as_ptr(),
          out_delta.as_ptr(),
          in_delta.as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
    }
  }

  fn update_params(&mut self, step_size: f32, l2_reg_coef: f32) {
    assert!(self.backward.is_some());
    assert!(l2_reg_coef >= 0.0);
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    if l2_reg_coef > 0.0 {
      backward.grad_weights.as_view_mut(ctx)
        .matrix_sum(l2_reg_coef, &self.weights.as_view(ctx));
      backward.grad_bias.as_view_mut(ctx)
        .row_vector_sum(l2_reg_coef, &self.bias.as_view(ctx));
    }
    {
      self.weights.as_view_mut(ctx)
        .matrix_sum(-step_size, &backward.grad_weights.as_view(ctx));
      self.bias.as_view_mut(ctx)
        .row_vector_sum(-step_size, &backward.grad_bias.as_view(ctx));
    }
  }

  fn sync_grads(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.communicate(&mut backward.grad_weights, ctx);
    comm_worker.communicate(&mut backward.grad_bias, ctx);
  }

  fn sync_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.communicate(&mut self.weights, ctx);
    comm_worker.communicate(&mut self.bias, ctx);
  }

  fn reset_grads(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.grad_weights.as_view_mut(ctx)
      .matrix_scale(scale);
    backward.grad_bias.as_view_mut(ctx)
      .row_vector_scale(scale);
  }
}

#[derive(Clone, Copy)]
pub struct Pool2dOperatorConfig {
  pub in_dims:  (usize, usize, usize),
  pub pool_size:    usize,
  pub pool_stride:  usize,
  pub pool_pad:     usize,
  pub pool_op:      PoolOperation,
  pub act_func:     ActivationFunction,
}

impl Pool2dOperatorConfig {
  pub fn get_out_dims(&self) -> (usize, usize, usize) {
    let (in_width, in_height, in_channels) = self.in_dims;
    let out_width = max(0, (in_width + 2 * self.pool_pad - self.pool_size + self.pool_stride) as isize) as usize / self.pool_stride;
    let out_height = max(0, (in_height + 2 * self.pool_pad - self.pool_size + self.pool_stride) as isize) as usize / self.pool_stride;
    (out_width, out_height, in_channels)
  }
}

pub struct Pool2dOperator {
  batch_cap:    usize,
  config:       Pool2dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  pool_mask:    DeviceBuffer<f32>,
}

impl Pool2dOperator {
  pub fn new(batch_size: usize, config: Pool2dOperatorConfig, prev_op: Option<&Operator>, context: Rc<DeviceContext>) -> Pool2dOperator {
    let in_dims = config.in_dims;
    let in_len = in_dims.len();
    let out_dims = config.get_out_dims();
    let out_len = out_dims.len();
    let ctx = &(*context).as_ref();
    Pool2dOperator{
      batch_cap:    batch_size,
      config:       config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("Pool2dOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_len * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_len * batch_size, ctx))),
      pool_mask:    DeviceBuffer::<f32>::zeros(in_len * batch_size, ctx),
    }
  }
}

impl Operator for Pool2dOperator {
  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    // FIXME(20160330)
    unimplemented!();
  }

  fn backward(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    // FIXME(20160330)
    unimplemented!();
  }

  fn update_params(&mut self, _step_size: f32, _l2_reg_coef: f32) {
    // Do nothing.
  }

  fn sync_grads(&mut self) {
    // Do nothing.
  }

  fn sync_params(&mut self) {
    // Do nothing.
  }

  fn reset_grads(&mut self, _scale: f32) {
    // Do nothing.
  }
}

#[derive(Clone, Copy)]
pub struct CategoricalLossConfig {
  pub category_count:   usize,
}

pub struct SoftmaxKLLossOperator {
  batch_cap:    usize,
  loss_config:  CategoricalLossConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     SharedDeviceBuf<f32>,

  label_cats:   DeviceArray2d<i32>,
  label_cats_h: Array2d<i32>,
  weights:      DeviceArray2d<f32>,
  weights_h:    Array2d<f32>,

  out_values:   DeviceArray2d<f32>,
  out_values_h: Array2d<f32>,
  max_value:    DeviceArray2d<f32>,
  out_cats:     DeviceArray2d<i32>,
  out_cats_h:   Array2d<i32>,
  out_loss1:    DeviceArray2d<f32>,
  out_loss:     DeviceBuffer<f32>,
  out_loss_h:   Vec<f32>,

  softmax:      CudnnSoftmaxOp,
}

impl SoftmaxKLLossOperator {
  pub fn new(batch_size: usize, loss_config: CategoricalLossConfig, prev_op: Option<&Operator>, context: Rc<DeviceContext>) -> SoftmaxKLLossOperator {
    let softmax = CudnnSoftmaxOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, loss_config.category_count, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, loss_config.category_count, batch_size).unwrap(),
    );
    let ctx = &(*context).as_ref();
    SoftmaxKLLossOperator{
      batch_cap:    batch_size,
      loss_config:  loss_config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("SoftmaxKLLossOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas().unwrap(),
      label_cats:   DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      label_cats_h: Array2d::<i32>::zeros((1, batch_size)),
      weights:      DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      weights_h:    Array2d::<f32>::zeros((1, batch_size)),
      out_values:   DeviceArray2d::<f32>::zeros((loss_config.category_count, batch_size), ctx),
      out_values_h: Array2d::zeros((loss_config.category_count, batch_size)),
      max_value:    DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      out_cats:     DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      out_cats_h:   Array2d::<i32>::zeros((1, batch_size)),
      out_loss1:    DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      out_loss:     DeviceBuffer::<f32>::zeros(1, ctx),
      out_loss_h:   vec![0.0],
      softmax:      softmax,
    }
  }
}

impl Operator for SoftmaxKLLossOperator {
  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    self.softmax.set_batch_size(batch_size).unwrap();
    unsafe { self.softmax.forward(
        self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
        self.out_values.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) }.unwrap();
  }

  fn backward(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    unsafe { rembrandt_kernel_batch_map_softmax_kl_backward(
        self.out_values.as_view(ctx).as_ptr(),
        self.loss_config.category_count as i32,
        batch_size as i32,
        self.label_cats.as_view(ctx).as_ptr(),
        self.weights.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }

  fn update_params(&mut self, _step_size: f32, _l2_reg_coef: f32) {
    // Do nothing.
  }

  fn sync_grads(&mut self) {
    // Do nothing.
  }

  fn sync_params(&mut self) {
    // Do nothing.
  }

  fn reset_grads(&mut self, _scale: f32) {
    // Do nothing.
  }
}

impl LossOperator for SoftmaxKLLossOperator {
  fn downcast(&self) -> &Operator {
    self as &Operator
  }

  fn stage_label(&mut self, batch_idx: usize, label: &SampleLabel) {
    match label {
      &SampleLabel::Category{category} => {
        self.label_cats_h.as_mut_slice()[batch_idx] = category;
      }
      _ => unimplemented!(),
    }
  }

  fn load_labels(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    self.label_cats.as_view_mut(ctx)
      .sync_load(&self.label_cats_h.as_view());
  }

  fn stage_weight(&mut self, batch_idx: usize, weight: f32) {
    self.weights_h.as_mut_slice()[batch_idx] = weight;
  }

  fn load_weights(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    self.weights.as_view_mut(ctx)
      .sync_load(&self.weights_h.as_view());
  }

  fn store_output_values(&mut self, batch_size: usize) {
  }

  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32> {
    &self.out_values_h
  }

  fn store_output_categories(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    assert!(self.loss_config.category_count <= 1024);
    let ctx = &(*self.context).as_ref();
    unsafe { rembrandt_kernel_batch_blockreduce_argmax(
        self.out_values.as_view(ctx).as_ptr(),
        self.loss_config.category_count as i32,
        batch_size as i32,
        self.max_value.as_view_mut(ctx).as_mut_ptr(),
        self.out_cats.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.out_cats.as_view(ctx)
      .sync_store(&mut self.out_cats_h.as_view_mut());
  }

  fn get_output_categories(&self, batch_size: usize) -> &Array2d<i32> {
    &self.out_cats_h
  }
}
