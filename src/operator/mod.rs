use data_new::{SampleLabel};
use operator::comm::{CommWorker};
use operator::input::{
  Data3dOperatorConfig,
  Data3dOperator,
  VarData3dOperatorConfig,
  VarData3dOperator,
};
use operator::loss::{
  //LossOperator,
  CategoricalLossConfig,
  SoftmaxKLLossOperator,
};
use operator::affine::{
  AffineOperatorConfig,
  AffineOperator,
};
use operator::conv::{
  Conv2dOperatorConfig,
  Conv2dOperator,
  BNormConv2dOperatorConfig,
  BNormConv2dOperator,
  StackResConv2dOperatorConfig,
  StackResConv2dOperator,
  ProjStackResConv2dOperatorConfig,
  ProjStackResConv2dOperator,
};

use array::{
  Array, AsyncArray, ArrayView, ArrayViewMut, ArrayZeroExt, NdArraySerialize,
  Shape, Array2d, Array3d,
};
use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::ext::{DeviceCastBytesExt, DeviceNumExt};
use array_cuda::device::linalg::{BlasMatrixExt, BlasVectorExt, Transpose};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut};
use array_cuda::device::random::{RandomSampleExt, UniformDist, GaussianDist};
use cuda_dnn::v4::{
  CudnnConvFwdOp, CudnnConvBwdFilterOp, CudnnConvBwdDataOp,
  CudnnAddOp, CudnnActKind, CudnnActOp, CudnnSoftmaxOp, CudnnPoolingOp, CudnnTransformOp,
  CudnnTensorDesc, CudnnFilterDesc, CudnnConvDesc,
};
use cuda_dnn::v4::ffi::{cudnnConvolutionFwdAlgo_t, cudnnPoolingMode_t};
use rembrandt_kernels::*;
use rembrandt_kernels::ffi::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, SeedableRng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{max};
use std::fs::{File};
use std::io::{Cursor};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::path::{PathBuf};
use std::rc::{Rc};

pub mod affine;
pub mod comm;
pub mod conv;
pub mod graph;
pub mod input;
pub mod loss;
pub mod pool;
pub mod seq;
pub mod worker;

pub trait OpRead {
  fn read<'ctx>(&mut self, offset: usize, dst: &mut DeviceBufferRefMut<'ctx, f32>) -> usize;
}

pub trait OpWrite {
  fn write<'ctx>(&mut self, offset: usize, src: &DeviceBufferRef<'ctx, f32>) -> usize;
}

pub struct OpCursor<T> {
  inner:    T,
}

impl<T> OpCursor<T> {
  pub fn new(inner: T) -> OpCursor<T> {
    OpCursor{inner: inner}
  }
}

impl OpRead for OpCursor<DeviceBuffer<f32>> {
  fn read<'ctx>(&mut self, offset: usize, dst: &mut DeviceBufferRefMut<'ctx, f32>) -> usize {
    let buf_len = dst.len();
    dst.copy(&mut self.inner.as_ref_range(offset, offset + buf_len, dst.ctx));
    buf_len
  }
}

impl OpWrite for OpCursor<DeviceBuffer<f32>> {
  fn write<'ctx>(&mut self, offset: usize, src: &DeviceBufferRef<'ctx, f32>) -> usize {
    let buf_len = src.len();
    self.inner.as_ref_mut_range(offset, offset + buf_len, src.ctx).copy(src);
    buf_len
  }
}

pub trait Operator {
  fn upcast_input(&mut self) -> &mut InputOperator { unimplemented!(); }
  fn upcast_loss(&mut self) -> &mut LossOperator { unimplemented!(); }

  fn batch_size(&self) -> usize;
  fn params_len(&self) -> usize { unimplemented!(); }
  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> { None }
  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> { None }
  fn get_output_act(&self, arm: usize) -> Option<SharedDeviceBuf<f32>> { None }
  fn get_output_delta(&self, arm: usize) -> Option<SharedDeviceBuf<f32>> { None }
  fn get_output_r_act(&self, arm: usize) -> Option<SharedDeviceBuf<f32>> { None }
  fn get_output_r_delta(&self, arm: usize) -> Option<SharedDeviceBuf<f32>> { None }
  fn init_params(&mut self, _shared_seed: [u64; 2]) {}
  fn decode_params(&mut self, _blob: &[u8]) -> usize { 0 }
  fn encode_params(&mut self, _blob: &mut Vec<u8>) {}
  fn decode_state(&mut self, _blob: &[u8]) -> usize { 0 }
  fn encode_state(&mut self, _blob: &mut Vec<u8>) {}
  fn read_params(&mut self, _offset: usize, _reader: &mut OpRead) -> usize { 0 }
  fn write_params(&mut self, _offset: usize, _writer: &mut OpWrite) -> usize { 0 }
  fn forward(&mut self, batch_size: usize, phase: OpPhase);

  // Requires `Backward` capability.
  fn read_grads(&mut self, _offset: usize, _reader: &mut OpRead) -> usize { 0 }
  fn write_grads(&mut self, _offset: usize, _writer: &mut OpWrite) -> usize { 0 }
  fn backward(&mut self, batch_size: usize);
  fn regularize(&mut self, _reg: Regularization) {}
  fn accumulate_grads(&mut self, _scale: f32, _momentum: f32) {}
  fn update_params(&mut self, _scale: f32) {}
  fn update_params2(&mut self, _grad_scale: f32, _update_scale: f32) {}
  fn save_params(&mut self) {}
  fn restore_params(&mut self) {}
  //fn set_grads_with_params_diff(&mut self) {}
  fn stage_grads(&mut self, _offset: usize, _comm_worker: &mut CommWorker) -> usize { 0 }
  fn merge_grads(&mut self, _offset: usize, _comm_worker: &mut CommWorker) -> usize { 0 }
  fn stage_params(&mut self, _offset: usize, _comm_worker: &mut CommWorker) -> usize { 0 }
  fn merge_params(&mut self, _offset: usize, _comm_worker: &mut CommWorker) -> usize { 0 }
  fn reset(&mut self) {}

  // Requires `RForward` capability.
  fn read_direction(&mut self, _offset: usize, _reader: &mut OpRead) -> usize { 0 }
  fn write_direction(&mut self, _offset: usize, _writer: &mut OpWrite) -> usize { 0 }
  fn r_forward(&mut self, _batch_size: usize) { unimplemented!(); }

  // Requires `RBackward` capability.
  fn r_backward(&mut self, _batch_size: usize) { unimplemented!(); }

  fn hv_reset_direction(&mut self, _init: HvDirectionInit) { unimplemented!(); }
  fn hv_solve_direction(&mut self, _solver: HvDirectionSolver) { unimplemented!(); }
  fn hv_forward(&mut self, _batch_size: usize) { unimplemented!(); }
  fn hv_backward(&mut self, _batch_size: usize) { unimplemented!(); }
  fn hv_update_params(&mut self, _scale: f32) { unimplemented!(); }
}

pub trait InputOperator: Operator {
  fn downcast(&self) -> &Operator;
  fn stage_shape(&mut self, batch_idx: usize, shape: (usize, usize, usize));
  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8];
  fn load_frames(&mut self, batch_size: usize);
  fn preload_frame(&mut self, _batch_idx: usize) { unimplemented!(); }
  fn wait_preload_frames(&mut self, _batch_size: usize) { unimplemented!(); }
}

pub trait LossOperator: Operator {
  fn downcast(&self) -> &Operator;

  fn stage_label(&mut self, batch_idx: usize, label: &SampleLabel);
  fn load_labels(&mut self, batch_size: usize);
  fn stage_weight(&mut self, batch_idx: usize, weight: f32);
  fn load_weights(&mut self, batch_size: usize);
  fn stage_category_weight(&mut self, _batch_idx: usize, _category: i32, _cat_weight: f32) {}
  fn load_category_weights(&mut self, _batch_size: usize) {}

  fn set_target_factors_with_rfwd_targets(&mut self, _batch_size: usize) { unimplemented!(); }
  fn reset_target_factors(&mut self, _batch_size: usize) { unimplemented!(); }

  fn store_loss(&mut self, batch_size: usize) -> f32;
  fn store_output_values(&mut self, batch_size: usize);
  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32>;
  fn store_output_categories(&mut self, batch_size: usize);
  fn get_output_categories(&self, batch_size: usize) -> &Array2d<i32>;
  fn accuracy_count(&self, batch_size: usize) -> usize;
  //fn reset_loss(&mut self);

  /*fn forward_loss(&mut self, batch_size: usize);
  fn backward_loss(&mut self, batch_size: usize);
  fn r_forward_loss(&mut self, batch_size: usize);
  fn r_backward_loss(&mut self, batch_size: usize);*/

  // Requires `HVBackward` capability.
  fn hv_stage_hessian_weight(&mut self, _batch_idx: usize, _h_weight: f32) { unimplemented!(); }
  fn hv_load_hessian_weights(&mut self, _batch_size: usize) { unimplemented!(); }
}

pub trait FullOperator: Operator + InputOperator + LossOperator {
}

pub enum OperatorNode {
  Hidden(Box<Operator>),
  Input(Box<InputOperator>),
  Loss(Box<LossOperator>),
  Split(Box<Operator>),
  //Join(Box<Operator>),
}

#[derive(Clone, Debug)]
//pub enum OperatorConfig<Comm> {
pub enum OperatorConfig {
  Data3d(Data3dOperatorConfig),
  VarData3d(VarData3dOperatorConfig),
  Affine(AffineOperatorConfig),
  Conv2d(Conv2dOperatorConfig),
  BNormConv2d(BNormConv2dOperatorConfig),
  StackResConv2d(StackResConv2dOperatorConfig),
  ProjStackResConv2d(ProjStackResConv2dOperatorConfig),
  Pool2d(Pool2dOperatorConfig),
  Dropout(DropoutOperatorConfig),
  SoftmaxKLLoss(CategoricalLossConfig),
  Split((usize, usize, usize)),
  //Join((usize, usize, usize)),
  //_Dummy(PhantomData<Comm>),
}

// FIXME(20160331): this is needed because of the hacky <Comm> generic.
/*unsafe impl<Comm> Send for OperatorConfig<Comm> where Comm: 'static + CommWorker {
}*/

/*impl<Comm> Clone for OperatorConfig<Comm> where Comm: 'static + CommWorker {
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
}*/

//impl<Comm2> OperatorConfig<Comm2> where Comm2: 'static + CommWorker {
impl OperatorConfig {
  pub fn params_len(&self) -> usize {
    match self {
      &OperatorConfig::Affine(ref cfg) => cfg.params_len(),
      &OperatorConfig::Conv2d(ref cfg) => cfg.params_len(),
      &OperatorConfig::BNormConv2d(ref cfg) => cfg.params_len(),
      &OperatorConfig::StackResConv2d(ref cfg) => cfg.params_len(),
      &OperatorConfig::ProjStackResConv2d(ref cfg) => cfg.params_len(),
      _ => 0,
    }
  }

  pub fn build_node(&self, batch_size: usize, capability: OpCapability, params_offset: Option<usize>, prev_op: Option<&Operator>, /*comm_worker: Option<Rc<RefCell<Comm>>>,*/ context: Rc<DeviceContext>) -> OperatorNode {
    match self {
      &OperatorConfig::Affine(ref cfg) => {
        OperatorNode::Hidden(Box::new(AffineOperator::new(batch_size, capability, params_offset.unwrap(), *cfg, prev_op, /*comm_worker,*/ context)))
      }
      &OperatorConfig::Conv2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(Conv2dOperator::new(batch_size, capability, params_offset.unwrap(), *cfg, prev_op, /*comm_worker,*/ context)))
      }
      &OperatorConfig::BNormConv2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(BNormConv2dOperator::new(batch_size, capability, params_offset.unwrap(), *cfg, prev_op, /*comm_worker,*/ context)))
      }
      &OperatorConfig::StackResConv2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(StackResConv2dOperator::new(batch_size, capability, params_offset.unwrap(), *cfg, prev_op, /*comm_worker,*/ context)))
      }
      &OperatorConfig::ProjStackResConv2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(ProjStackResConv2dOperator::new(batch_size, capability, params_offset.unwrap(), *cfg, prev_op, /*comm_worker,*/ context)))
      }
      &OperatorConfig::Pool2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(Pool2dOperator::new(batch_size, *cfg, prev_op, context)))
      }
      &OperatorConfig::Dropout(ref cfg) => {
        OperatorNode::Hidden(Box::new(DropoutOperator::new(batch_size, *cfg, prev_op, context)))
      }
      &OperatorConfig::Data3d(ref cfg) => {
        OperatorNode::Input(Box::new(Data3dOperator::new(batch_size, cfg.clone(), context)))
      }
      &OperatorConfig::VarData3d(ref cfg) => {
        OperatorNode::Input(Box::new(VarData3dOperator::new(batch_size, cfg.clone(), context)))
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
      //_ => unreachable!(),
    }
  }

  //pub fn build_operator<Comm: 'static + CommWorker>(&self, batch_size: usize, capability: OpCapability, params_offset: usize, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> Box<Operator> {
  pub fn build_operator(&self, batch_size: usize, capability: OpCapability, params_offset: usize, prev_op: Option<&Operator>, /*comm_worker: Option<Rc<RefCell<Comm>>>,*/ context: Rc<DeviceContext>) -> Box<Operator> {
    match self.build_node(batch_size, capability, Some(params_offset), prev_op, /*comm_worker,*/ context) {
      OperatorNode::Hidden(op) => op,
      _ => unimplemented!(),
    }
  }

  pub fn build_input_operator(&self, batch_size: usize, context: Rc<DeviceContext>) -> Box<InputOperator> {
    match self.build_node(batch_size, OpCapability::Forward, None, None, /*None,*/ context) {
      OperatorNode::Input(op) => op,
      _ => unimplemented!(),
    }
  }

  pub fn build_loss_operator(&self, batch_size: usize, prev_op: Option<&Operator>, context: Rc<DeviceContext>) -> Box<LossOperator> {
    // FIXME(20160330): set proper `OpCapability`.
    match self.build_node(batch_size, OpCapability::Backward, None, prev_op, /*None,*/ context) {
      OperatorNode::Loss(op) => op,
      _ => unimplemented!(),
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum OpCapability {
  Forward,
  Backward,
  RForward,
  RBackward,
}

impl OpCapability {
  pub fn backward_enabled(&self) -> bool {
    match *self {
      OpCapability::Forward     => false,
      OpCapability::Backward    => true,
      OpCapability::RForward    => true,
      OpCapability::RBackward   => true,
    }
  }

  pub fn r_forward_enabled(&self) -> bool {
    match *self {
      OpCapability::Forward     => false,
      OpCapability::Backward    => false,
      OpCapability::RForward    => true,
      OpCapability::RBackward   => true,
    }
  }

  pub fn r_backward_enabled(&self) -> bool {
    match *self {
      OpCapability::Forward     => false,
      OpCapability::Backward    => false,
      OpCapability::RForward    => false,
      OpCapability::RBackward   => true,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum OpPhase {
  Inference,
  Training{t: usize},
}

#[derive(Clone, Copy, Debug)]
pub enum Regularization {
  L2{l2_reg_coef: f32},
}

#[derive(Clone, Copy, Debug)]
pub enum HvDirectionInit {
  Gradient,
}

#[derive(Clone, Copy, Debug)]
pub enum HvDirectionSolver {
  PrecondConjGrad,
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationFunction {
  Identity,
  Rect,
  Logistic,
  Tanh,
}

#[derive(Clone, Copy, Debug)]
pub enum ParamsInit {
  Disabled,
  Uniform{half_range: f32},
  Normal{std: f32},
  Xavier,
  KaimingFwd,
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
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

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
}

pub struct JoinOperatorConfig {
  pub out_dims:  (usize, usize, usize),
}

pub struct AddJoinOperator {
  batch_cap:    usize,
  num_in_arms:  usize,
  config:       JoinOperatorConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

}

impl AddJoinOperator {
  pub fn new(batch_size: usize, num_in_arms: usize, config: JoinOperatorConfig, prev_ops: Vec<&Operator>, context: Rc<DeviceContext>) -> AddJoinOperator {
    unimplemented!();
  }
}

pub struct CatJoinOperator;

#[derive(Clone, Copy, Debug)]
pub enum PoolOperation {
  Max,
  Average,
}

#[derive(Clone, Copy, Debug)]
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

  pooling:      CudnnPoolingOp,
}

impl Pool2dOperator {
  pub fn new(batch_size: usize, config: Pool2dOperatorConfig, prev_op: Option<&Operator>, context: Rc<DeviceContext>) -> Pool2dOperator {
    let in_dims = config.in_dims;
    let (in_width, in_height, in_channels) = in_dims;
    let in_len = in_dims.len();
    let out_dims = config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let out_len = out_dims.len();
    let ctx = &(*context).as_ref();
    let pooling = match CudnnPoolingOp::create_2d_symmetric(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        config.pool_size,
        config.pool_stride,
        config.pool_pad,
        match config.pool_op {
          PoolOperation::Max      => cudnnPoolingMode_t::Max,
          PoolOperation::Average  => cudnnPoolingMode_t::AverageCountIncludingPadding,
          //PoolOperation::Average  => cudnnPoolingMode_t::AverageCountExcludingPadding,
        },
    ) {
      Ok(pooling) => pooling,
      Err(e) => panic!("Pool2dOperator failed to create CudnnPoolingOp: {:?}", e),
    };
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
      pooling:      pooling,
    }
  }
}

impl Operator for Pool2dOperator {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    self.pooling.set_batch_size(batch_size).unwrap();
    unsafe { self.pooling.forward(
        self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
        self.out_act.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) }.unwrap();
  }

  fn backward(&mut self, batch_size: usize) {
    if let Some(ref mut in_delta) = self.in_delta {
      assert!(batch_size <= self.batch_cap);
      let ctx = &(*self.context).as_ref();
      self.pooling.set_batch_size(batch_size).unwrap();
      unsafe { self.pooling.backward(
          self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
          self.out_act.borrow_mut().as_ref(ctx).as_ptr(),
          self.out_delta.borrow_mut().as_ref(ctx).as_ptr(),
          in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
          &*ctx.get_dnn(),
      ) }.unwrap();
    }
  }

  /*fn sync_grads(&mut self) {
    // Do nothing.
  }

  fn sync_params(&mut self) {
    // Do nothing.
  }*/
}

#[derive(Clone, Copy, Debug)]
pub struct DropoutOperatorConfig {
  pub channels:     usize,
  pub drop_ratio:   f32,
}

pub struct DropoutOperator {
  batch_cap:    usize,
  config:       DropoutOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  uniform_dist: UniformDist,
  rand_samples: DeviceBuffer<f32>,
  drop_mask:    DeviceBuffer<i32>,

  //state:        DeviceBuffer<u8>,
  //dropout:      CudnnDropoutOp,
}

impl DropoutOperator {
  pub fn new(batch_size: usize, config: DropoutOperatorConfig, prev_op: Option<&Operator>, context: Rc<DeviceContext>) -> DropoutOperator {
    let channels = config.channels;
    let ctx = &(*context).as_ref();
    /*let pooling = match CudnnPoolingOp::create_2d_symmetric(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        config.pool_size,
        config.pool_stride,
        config.pool_pad,
        match config.pool_op {
          PoolOperation::Max      => cudnnPoolingMode_t::Max,
          PoolOperation::Average  => cudnnPoolingMode_t::AverageCountExcludingPadding,
        },
    ) {
      Ok(pooling) => pooling,
      Err(e) => panic!("Pool2dOperator failed to create CudnnPoolingOp: {:?}", e),
    };*/
    DropoutOperator{
      batch_cap:    batch_size,
      config:       config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("DropoutOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(channels * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(channels * batch_size, ctx))),
      uniform_dist: UniformDist,
      rand_samples: DeviceBuffer::zeros(channels * batch_size, ctx),
      drop_mask:    DeviceBuffer::zeros(channels * batch_size, ctx),
    }
  }
}

impl Operator for DropoutOperator {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    match phase {
      OpPhase::Inference => {
        self.in_act.borrow_mut().as_ref(ctx)
          .send(&mut self.out_act.borrow_mut().as_ref_mut(ctx));
      }
      OpPhase::Training{..} => {
        self.rand_samples.as_ref_mut(ctx).sample(&self.uniform_dist);
        unsafe { rembrandt_kernel_map_dropout(
            self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
            (self.config.channels * batch_size) as i32,
            self.config.drop_ratio, 1.0,
            self.rand_samples.as_ref(ctx).as_ptr(),
            self.out_act.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
            self.drop_mask.as_ref_mut(ctx).as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
    }
  }

  fn backward(&mut self, batch_size: usize) {
    if let Some(ref mut in_delta) = self.in_delta {
      assert!(batch_size <= self.batch_cap);
      let ctx = &(*self.context).as_ref();
      unsafe { rembrandt_kernel_map_dropout_backprop(
          self.out_delta.borrow_mut().as_ref(ctx).as_ptr(),
          (self.config.channels * batch_size) as i32,
          self.config.drop_ratio, 1.0,
          self.drop_mask.as_ref(ctx).as_ptr(),
          in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
          ctx.stream.ptr,
      ) };
    }
  }

  /*fn sync_grads(&mut self) {
    // Do nothing.
  }

  fn sync_params(&mut self) {
    // Do nothing.
  }*/
}
