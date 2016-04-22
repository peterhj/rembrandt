use data_new::{SampleLabel};
use operator::{
  Operator,
  ActivationFunction,
  ParamsInit,
  Regularization,
  OpCapability,
  OpPhase,
  Conv2dFwdBackend,
  Conv2dBwdBackend,
  SharedDeviceBuf,
  Conv2dOperatorConfig,
};
use operator::comm::{CommWorker};
use operator::loss::{
  LossOperator,
  CategoricalLossConfig,
  SoftmaxKLLossOperator,
};

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::ext::{DeviceCastBytesExt, DeviceNumExt};
use array_cuda::device::linalg::{BlasMatrixExt, BlasVectorExt, Transpose};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer};
use array_cuda::device::random::{RandomSampleExt, UniformDist};
use array_new::{
  Array, AsyncArray, ArrayView, ArrayViewMut, ArrayZeroExt, NdArraySerialize,
  Shape, Array2d,
};
use cuda_dnn::v4::{
  CudnnTensorDesc, CudnnFilterDesc, CudnnConvDesc,
  CudnnConvFwdOp, CudnnConvBwdFilterOp, CudnnConvBwdDataOp,
  CudnnAddOp, CudnnActKind, CudnnActOp, CudnnSoftmaxOp, CudnnPoolingOp, CudnnTransformOp, CudnnBatchNormOp,
};
use cuda_dnn::v4::ffi::{cudnnConvolutionFwdAlgo_t, cudnnPoolingMode_t, cudnnBatchNormMode_t};
use rembrandt_kernels::*;
use rembrandt_kernels::ffi::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, SeedableRng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{max};
use std::io::{Cursor};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub enum BnormMovingAverage {
  Cumulative,
  Exponential{ema_factor: f64},
}

impl BnormMovingAverage {
  pub fn at_iter(&self, t: usize) -> f64 {
    match self {
      &BnormMovingAverage::Cumulative => {
        1.0 / (1.0 + t as f64)
      }
      &BnormMovingAverage::Exponential{ema_factor} => {
        if t == 0 {
          1.0
        } else {
          ema_factor
        }
      }
    }
  }
}

pub struct BnormConv2dOperator<Comm> {
  batch_cap:    usize,
  _capability:  OpCapability,
  params_off:   usize,
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

  backward:     Option<BnormConv2dBwdOperator<Comm>>,
  hv_backward:  Option<BnormConv2dHvBwdOperator>,
}

struct BnormConv2dBwdOperator<Comm> {
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,
  acc_grad_weights: DeviceArray2d<f32>,
  acc_grad_bias:    DeviceArray2d<f32>,
  save_weights: DeviceArray2d<f32>,
  save_bias:    DeviceArray2d<f32>,

  conv_bwd_w:   CudnnConvBwdFilterOp,
  conv_bwd_d:   CudnnConvBwdDataOp,

  comm_worker:  Rc<RefCell<Comm>>,
}

struct BnormConv2dHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
  dir_bias:     DeviceArray2d<f32>,
}

impl<Comm> BnormConv2dOperator<Comm> where Comm: CommWorker {
  pub fn new(batch_size: usize, capability: OpCapability, params_offset: usize, config: Conv2dOperatorConfig, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> BnormConv2dOperator<Comm> {
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
      Some(BnormConv2dBwdOperator{
        grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        acc_grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        acc_grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        save_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        save_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
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

    BnormConv2dOperator{
      batch_cap:    batch_size,
      _capability:  capability,
      params_off:   params_offset,
      config:       config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("BnormConv2dOperator missing required prev operator output vars"),
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

impl<Comm> Operator for BnormConv2dOperator<Comm> where Comm: CommWorker {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn init_params(&mut self, shared_seed: [u64; 2]) {
    let Conv2dOperatorConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    let mut init_weights = Array2d::zeros((conv_size * conv_size * in_channels, out_channels));
    match self.config.init_weights {
      ParamsInit::Disabled => {
        panic!("BnormConv2dOperator: params init explicitly disabled");
      }
      ParamsInit::Uniform{half_range} => {
        let dist = Range::new(-half_range as f64, half_range as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInit::Normal{std} => {
        let dist = Normal::new(0.0, std as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInit::Xavier => {
        // FIXME(20160420)
        unimplemented!();
      }
      ParamsInit::KaimingFwd => {
        let in_conns = self.config.conv_size * self.config.conv_size * self.config.in_dims.2;
        let std = (2.0 / in_conns as f64).sqrt();
        let dist = Normal::new(0.0, std);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
    }
    let init_bias = Array2d::zeros((1, out_channels));
    self.weights.as_view_mut(ctx).sync_load(&init_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&init_bias.as_view());
  }

  fn read_params(&mut self, blob: &[u8]) -> usize {
    let Conv2dOperatorConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("BnormConv2dOperator failed to deserialize weights!");
    let load_bias = Array2d::deserialize(&mut reader)
      .ok().expect("BnormConv2dOperator failed to deserialize bias!");
    assert_eq!((conv_size * conv_size * in_channels, out_channels), load_weights.as_view().bound());
    assert_eq!((1, out_channels), load_bias.as_view().bound());
    self.weights.as_view_mut(ctx).sync_load(&load_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&load_bias.as_view());
    let progress = reader.position() as usize;
    progress
  }

  fn write_params(&mut self, blob: &mut Vec<u8>) {
    let ctx = &(*self.context).as_ref();
    let weights = self.weights.as_view(ctx);
    let bias = self.bias.as_view(ctx);
    let mut save_weights = Array2d::zeros(weights.bound());
    let mut save_bias = Array2d::zeros(bias.bound());
    weights.sync_store(&mut save_weights.as_view_mut());
    bias.sync_store(&mut save_bias.as_view_mut());
    save_weights.serialize(blob).unwrap();
    save_bias.serialize(blob).unwrap();
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let out_dims = self.config.get_out_dims();
    let out_length = out_dims.len();

    let &mut BnormConv2dOperator{
      ref context,
      ref mut in_act, ref mut out_act,
      ref mut weights, ref mut bias,
      ref mut workspace,
      .. } = self;

    let ctx = &(**context).as_ref();
    let mut out_act = out_act.borrow_mut().as_ref_mut(ctx);

    self.conv_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv_fwd.forward(
        1.0,
        in_act.borrow_mut().as_ref(ctx).as_ptr(),
        weights.as_view(ctx).as_ptr(),
        0.0,
        out_act.as_mut_ptr(),
        workspace.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("conv2d forward failed: {:?}", e); }
    }
    self.add_bias.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias.forward(
        1.0,
        bias.as_view(ctx).as_ptr(),
        1.0,
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
    let out_dims = self.config.get_out_dims();
    let out_length = out_dims.len();

    let &mut BnormConv2dOperator{
      ref context,
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights, //ref mut bias,
      ref mut workspace,
      ref mut backward,
      .. } = self;
    let mut backward = backward.as_mut().unwrap();
    let &mut BnormConv2dBwdOperator{
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
        1.0,
        grad_weights.as_view_mut(ctx).as_mut_ptr(),
        workspace.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    unsafe { backward.conv_bwd_w.backward_bias(
        1.0,
        out_delta.as_ptr(),
        1.0,
        grad_bias.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    if let &mut Some(ref mut in_delta) = in_delta {
      backward.conv_bwd_d.set_batch_size(batch_size).unwrap();
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);
      unsafe { backward.conv_bwd_d.backward_data(
          1.0,
          weights.as_view(ctx).as_ptr(),
          out_delta.as_ptr(),
          0.0,
          in_delta.as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
    }
  }

  fn regularize(&mut self, reg: Regularization) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    match reg {
      Regularization::L2{l2_reg_coef} => {
        assert!(l2_reg_coef >= 0.0);
        if l2_reg_coef > 0.0 {
          backward.grad_weights.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights.as_view(ctx));
          backward.grad_bias.as_view_mut(ctx)
            .row_vector_sum(l2_reg_coef, &self.bias.as_view(ctx));
        }
      }
    }
  }

  fn accumulate_grads(&mut self, scale: f32, momentum: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.acc_grad_weights.as_view_mut(ctx)
      .matrix_scale(momentum);
    backward.acc_grad_bias.as_view_mut(ctx)
      .row_vector_scale(momentum);
    backward.acc_grad_weights.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights.as_view(ctx));
    backward.acc_grad_bias.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.grad_bias.as_view(ctx));
  }

  fn update_params(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    self.weights.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights.as_view(ctx));
    self.bias.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.acc_grad_bias.as_view(ctx));
  }

  /*fn reset_params(&mut self, momentum: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    assert!(momentum >= 0.0);
    self.weights.as_view_mut(ctx)
      .matrix_sum(momentum, &backward.acc_grad_weights.as_view(ctx));
    self.bias.as_view_mut(ctx)
      .row_vector_sum(momentum, &backward.acc_grad_bias.as_view(ctx));
  }*/

  /*fn update_params(&mut self, step_size: f32, l2_reg_coef: f32) {
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
    self.weights.as_view_mut(ctx)
      .matrix_sum(-step_size, &backward.grad_weights.as_view(ctx));
    self.bias.as_view_mut(ctx)
      .row_vector_sum(-step_size, &backward.grad_bias.as_view(ctx));
  }*/

  fn save_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    self.weights.as_view(ctx)
      .send(&mut backward.save_weights.as_view_mut(ctx));
    self.bias.as_view(ctx)
      .send(&mut backward.save_bias.as_view_mut(ctx));
  }

  fn restore_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.save_weights.as_view(ctx)
      .send(&mut self.weights.as_view_mut(ctx));
    backward.save_bias.as_view(ctx)
      .send(&mut self.bias.as_view_mut(ctx));
  }

  fn set_grads_with_params_diff(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    self.weights.as_view(ctx)
      .send(&mut backward.acc_grad_weights.as_view_mut(ctx));
    self.bias.as_view(ctx)
      .send(&mut backward.acc_grad_bias.as_view_mut(ctx));
    backward.acc_grad_weights.as_view_mut(ctx)
      .matrix_sum(-1.0, &backward.save_weights.as_view(ctx));
    backward.acc_grad_bias.as_view_mut(ctx)
      .row_vector_sum(-1.0, &backward.save_bias.as_view(ctx));
  }

  fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.load(self.params_off, &mut self.weights, ctx);
    comm_worker.load(self.params_off, &mut self.bias, ctx);
  }

  fn sync_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights, ctx);
    comm_worker.store(self.params_off, &mut self.bias, ctx);
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

  fn reset(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.grad_weights.as_view_mut(ctx)
      .matrix_scale(0.0);
    backward.grad_bias.as_view_mut(ctx)
      .row_vector_scale(0.0);
  }
}

#[derive(Clone, Copy, Debug)]
pub struct StackResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  //pub out_dims:     (usize, usize, usize),
  pub bnorm_mov_avg:    BnormMovingAverage,
  pub bnorm_epsilon:    f64,
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}

impl StackResConv2dOperatorConfig {
  pub fn params_len(&self) -> usize {
    let (_, _, in_channels) = self.in_dims;
    let weights1_len = 3 * 3 * in_channels * in_channels;
    let bias1_len = in_channels;
    let weights2_len = 3 * 3 * in_channels * in_channels;
    let bias2_len = in_channels;
    weights1_len + bias1_len +
        weights2_len + bias2_len
  }
}

pub struct StackResConv2dOperator<Comm> {
  batch_cap:    usize,
  _capability:  OpCapability,
  params_off:   usize,
  config:       StackResConv2dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights1:     DeviceArray2d<f32>,
  bias1:        DeviceArray2d<f32>,
  weights2:     DeviceArray2d<f32>,
  bias2:        DeviceArray2d<f32>,

  tmp1_pre_act:     DeviceBuffer<f32>,
  tmp1_pre_delta:   DeviceBuffer<f32>,
  tmp1_post_act:    DeviceBuffer<f32>,
  tmp1_post_delta:  DeviceBuffer<f32>,
  tmp2_pre_act:     DeviceBuffer<f32>,
  tmp2_pre_delta:   DeviceBuffer<f32>,

  bn_scale1:        DeviceArray2d<f32>,
  bn_scale1_grad:   DeviceArray2d<f32>,
  acc_bn_scale1_grad:   DeviceArray2d<f32>,
  bn_bias1:         DeviceArray2d<f32>,
  bn_bias1_grad:    DeviceArray2d<f32>,
  acc_bn_bias1_grad:    DeviceArray2d<f32>,
  bn_running_mean1: DeviceArray2d<f32>,
  bn_running_ivar1: DeviceArray2d<f32>,
  bn_cached_mean1:  DeviceArray2d<f32>,
  bn_cached_ivar1:  DeviceArray2d<f32>,
  batchnorm1:   CudnnBatchNormOp,

  bn_scale2:        DeviceArray2d<f32>,
  bn_scale2_grad:   DeviceArray2d<f32>,
  acc_bn_scale2_grad:   DeviceArray2d<f32>,
  bn_bias2:         DeviceArray2d<f32>,
  bn_bias2_grad:    DeviceArray2d<f32>,
  acc_bn_bias2_grad:    DeviceArray2d<f32>,
  bn_running_mean2: DeviceArray2d<f32>,
  bn_running_ivar2: DeviceArray2d<f32>,
  bn_cached_mean2:  DeviceArray2d<f32>,
  bn_cached_ivar2:  DeviceArray2d<f32>,
  batchnorm2:   CudnnBatchNormOp,

  workspace:    DeviceBuffer<u8>,
  conv1_fwd:    CudnnConvFwdOp,
  add_bias1:    CudnnAddOp,
  conv2_fwd:    CudnnConvFwdOp,
  add_bias2:    CudnnAddOp,
  add_input:    CudnnAddOp,

  backward:     Option<StackResConv2dBwdOperator<Comm>>,
  //hv_backward:  Option<BotResConv2dHvBwdOperator>,
}

struct StackResConv2dBwdOperator<Comm> {
  grad_weights1:      DeviceArray2d<f32>,
  grad_bias1:      DeviceArray2d<f32>,
  acc_grad_weights1:  DeviceArray2d<f32>,
  acc_grad_bias1:  DeviceArray2d<f32>,
  conv1_bwd_w:  CudnnConvBwdFilterOp,
  conv1_bwd_d:  CudnnConvBwdDataOp,

  first_batch1: bool,

  grad_weights2:      DeviceArray2d<f32>,
  grad_bias2:      DeviceArray2d<f32>,
  acc_grad_weights2:  DeviceArray2d<f32>,
  acc_grad_bias2:  DeviceArray2d<f32>,
  conv2_bwd_w:  CudnnConvBwdFilterOp,
  conv2_bwd_d:  CudnnConvBwdDataOp,

  first_batch2: bool,

  comm_worker:  Rc<RefCell<Comm>>,
}

impl<Comm> StackResConv2dOperator<Comm> where Comm: CommWorker {
  pub fn new(batch_size: usize, capability: OpCapability, params_offset: usize, config: StackResConv2dOperatorConfig, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> StackResConv2dOperator<Comm> {
    let StackResConv2dOperatorConfig{
      in_dims,
      .. } = config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_length = in_dims.len();

    let ctx = &(*context).as_ref();

    let mut workspace_size = 0;

    // FIXME(20160420): support batch norm (the "spatial" variant).

    let conv1_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels, in_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv1_fwd.work_size);

    let conv2_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels, in_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv2_fwd.work_size);

    let batchnorm1 = CudnnBatchNormOp::new(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels, 1).unwrap(),
        cudnnBatchNormMode_t::Spatial,
    );

    let batchnorm2 = CudnnBatchNormOp::new(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels, 1).unwrap(),
        cudnnBatchNormMode_t::Spatial,
    );

    let backward = if capability.backward_enabled() {
      let conv1_bwd_w = CudnnConvBwdFilterOp::create_fastest(
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
          CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels, in_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels, 1).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv1_bwd_w.work_size);

      let conv1_bwd_d = CudnnConvBwdDataOp::create_fastest(
          CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels, in_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv1_bwd_d.work_size);

      let conv2_bwd_w = CudnnConvBwdFilterOp::create_fastest(
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
          CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels, in_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels, 1).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv2_bwd_w.work_size);

      let conv2_bwd_d = CudnnConvBwdDataOp::create_fastest(
          CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels, in_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv2_bwd_d.work_size);

      Some(StackResConv2dBwdOperator{
        grad_weights1:      DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),
        grad_bias1:         DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
        acc_grad_weights1:  DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),
        acc_grad_bias1:     DeviceArray2d::<f32>::zeros((1, in_channels), ctx),

        conv1_bwd_w:  conv1_bwd_w,
        conv1_bwd_d:  conv1_bwd_d,

        first_batch1: true,

        grad_weights2:      DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),
        grad_bias2:         DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
        acc_grad_weights2:  DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),
        acc_grad_bias2:     DeviceArray2d::<f32>::zeros((1, in_channels), ctx),

        conv2_bwd_w:  conv2_bwd_w,
        conv2_bwd_d:  conv2_bwd_d,

        first_batch2: true,

        comm_worker:  comm_worker.unwrap(),
      })
    } else {
      None
    };

    let add_bias1 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
    );
    let add_bias2 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
    );
    let add_input = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
    );

    // XXX(20160421): Initialize gammas to all ones.
    let mut bn_scale1 = DeviceArray2d::<f32>::zeros((1, in_channels), ctx);
    bn_scale1.as_view_mut(ctx).set_constant(1.0);
    let mut bn_scale2 = DeviceArray2d::<f32>::zeros((1, in_channels), ctx);
    bn_scale2.as_view_mut(ctx).set_constant(1.0);

    StackResConv2dOperator{
      batch_cap:    batch_size,
      _capability:  capability,
      params_off:   params_offset,
      config:       config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("BotResConv2dOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),

      weights1:     DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),
      bias1:        DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      weights2:     DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),
      bias2:        DeviceArray2d::<f32>::zeros((1, in_channels), ctx),

      tmp1_pre_act:     DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp1_pre_delta:   DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp1_post_act:    DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp1_post_delta:  DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp2_pre_act:     DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp2_pre_delta:   DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),

      bn_scale1:        bn_scale1,
      bn_scale1_grad:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      acc_bn_scale1_grad:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_bias1:         DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_bias1_grad:   DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      acc_bn_bias1_grad:   DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_running_mean1: DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_running_ivar1: DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_cached_mean1:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_cached_ivar1:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      batchnorm1: batchnorm1,

      bn_scale2:        bn_scale2,
      bn_scale2_grad:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      acc_bn_scale2_grad:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_bias2:         DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_bias2_grad:   DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      acc_bn_bias2_grad:   DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_running_mean2: DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_running_ivar2: DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_cached_mean2:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_cached_ivar2:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      batchnorm2: batchnorm2,

      workspace:    DeviceBuffer::<u8>::zeros(workspace_size, ctx),
      conv1_fwd:    conv1_fwd,
      add_bias1:    add_bias1,
      conv2_fwd:    conv2_fwd,
      add_bias2:    add_bias2,
      add_input:    add_input,

      backward:     backward,
      //hv_backward:  None,
    }
  }
}

impl<Comm> Operator for StackResConv2dOperator<Comm> where Comm: CommWorker {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn init_params(&mut self, shared_seed: [u64; 2]) {
    let StackResConv2dOperatorConfig{in_dims, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    let mut init_weights1 = Array2d::zeros((3 * 3 * in_channels, in_channels));
    let mut init_weights2 = Array2d::zeros((3 * 3 * in_channels, in_channels));
    match self.config.init_weights {
      ParamsInit::Disabled => {
        panic!("StackResConv2dOperator: params init explicitly disabled");
      }
      ParamsInit::Uniform{half_range} => {
        let dist = Range::new(-half_range as f64, half_range as f64);
        for w in init_weights1.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
        for w in init_weights2.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInit::Normal{std} => {
        let dist = Normal::new(0.0, std as f64);
        for w in init_weights1.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
        for w in init_weights2.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInit::Xavier => {
        // FIXME(20160420)
        unimplemented!();
      }
      ParamsInit::KaimingFwd => {
        let in_conns = 3 * 3 * self.config.in_dims.2;
        let std = (2.0 / in_conns as f64).sqrt();
        let dist = Normal::new(0.0, std as f64);
        for w in init_weights1.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
        for w in init_weights2.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
    }
    let init_bias = Array2d::zeros((1, in_channels));
    self.weights1.as_view_mut(ctx).sync_load(&init_weights1.as_view());
    self.bias1.as_view_mut(ctx).sync_load(&init_bias.as_view());
    self.weights2.as_view_mut(ctx).sync_load(&init_weights2.as_view());
    self.bias2.as_view_mut(ctx).sync_load(&init_bias.as_view());
  }

  fn read_params(&mut self, blob: &[u8]) -> usize {
    let StackResConv2dOperatorConfig{in_dims, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut reader = Cursor::new(blob);
    let load_weights1 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize weights!");
    let load_bias1 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize bias!");
    let load_weights2 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize weights!");
    let load_bias2 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize bias!");
    assert_eq!((3 * 3 * in_channels, in_channels), load_weights1.as_view().bound());
    assert_eq!((1, in_channels), load_bias1.as_view().bound());
    assert_eq!((3 * 3 * in_channels, in_channels), load_weights2.as_view().bound());
    assert_eq!((1, in_channels), load_bias2.as_view().bound());
    self.weights1.as_view_mut(ctx).sync_load(&load_weights1.as_view());
    self.bias1.as_view_mut(ctx).sync_load(&load_bias1.as_view());
    self.weights2.as_view_mut(ctx).sync_load(&load_weights2.as_view());
    self.bias2.as_view_mut(ctx).sync_load(&load_bias2.as_view());
    let progress = reader.position() as usize;
    progress
  }

  fn write_params(&mut self, blob: &mut Vec<u8>) {
    let ctx = &(*self.context).as_ref();

    let weights1 = self.weights1.as_view(ctx);
    let bias1 = self.bias1.as_view(ctx);
    let mut save_weights1 = Array2d::zeros(weights1.bound());
    let mut save_bias1 = Array2d::zeros(bias1.bound());
    weights1.sync_store(&mut save_weights1.as_view_mut());
    bias1.sync_store(&mut save_bias1.as_view_mut());
    save_weights1.serialize(blob).unwrap();
    save_bias1.serialize(blob).unwrap();

    let weights2 = self.weights2.as_view(ctx);
    let bias2 = self.bias2.as_view(ctx);
    let mut save_weights2 = Array2d::zeros(weights2.bound());
    let mut save_bias2 = Array2d::zeros(bias2.bound());
    weights2.sync_store(&mut save_weights2.as_view_mut());
    bias2.sync_store(&mut save_bias2.as_view_mut());
    save_weights2.serialize(blob).unwrap();
    save_bias2.serialize(blob).unwrap();
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    //let out_dims = self.config.get_out_dims();
    let out_dims = self.config.in_dims;
    let out_length = out_dims.len();

    // FIXME(20160413): the bottleneck residual consists of 3 convolutions.

    let &mut StackResConv2dOperator{
      ref context,
      ref mut in_act, ref mut out_act,
      ref mut tmp1_pre_act,
      ref mut tmp1_pre_delta,
      ref mut tmp1_post_act,
      ref mut tmp1_post_delta,
      ref mut tmp2_pre_act,
      ref mut tmp2_pre_delta,
      ref mut weights1, ref mut bias1,
      ref mut weights2, ref mut bias2,
      ref mut workspace,
      .. } = self;

    let ctx = &(**context).as_ref();
    let mut in_act = in_act.borrow_mut();
    let mut out_act = out_act.borrow_mut();

    in_act.as_ref(ctx).send(&mut out_act.as_ref_mut(ctx));

    self.conv1_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv1_fwd.forward(
        1.0,
        in_act.as_ref(ctx).as_ptr(),
        weights1.as_view(ctx).as_ptr(),
        0.0,
        tmp1_pre_act.as_ref_mut(ctx).as_mut_ptr(),
        workspace.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("conv2d forward failed: {:?}", e); }
    }
    /*self.add_bias1.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias1.forward(
        bias1.as_view(ctx).as_ptr(),
        tmp1_pre_act.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };*/

    match phase {
      OpPhase::Inference => {
        self.batchnorm1.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm1.forward_inference(
            1.0,
            tmp1_pre_act.as_ref(ctx).as_ptr(),
            0.0,
            tmp1_post_act.as_ref_mut(ctx).as_mut_ptr(),
            self.config.bnorm_epsilon,
            self.bn_scale1.as_view(ctx).as_ptr(),
            self.bn_bias1.as_view(ctx).as_ptr(),
            self.bn_running_mean1.as_view(ctx).as_ptr(),
            self.bn_running_ivar1.as_view(ctx).as_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
      OpPhase::Training => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        let ema_factor = match backward.first_batch1 {
          true  => {
            backward.first_batch1 = false;
            1.0
          }
          false => 0.01,
        };
        self.batchnorm1.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm1.forward_training(
            1.0,
            tmp1_pre_act.as_ref(ctx).as_ptr(),
            0.0,
            tmp1_post_act.as_ref_mut(ctx).as_mut_ptr(),
            ema_factor,
            self.config.bnorm_epsilon,
            self.bn_scale1.as_view(ctx).as_ptr(),
            self.bn_bias1.as_view(ctx).as_ptr(),
            self.bn_running_mean1.as_view_mut(ctx).as_mut_ptr(),
            self.bn_running_ivar1.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_mean1.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_ivar1.as_view_mut(ctx).as_mut_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
    }

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_inplace(
            tmp1_post_act.as_ref_mut(ctx).as_mut_ptr(),
            out_length as i32,
            batch_size as i32,
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }

    self.conv2_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv2_fwd.forward(
        1.0,
        tmp1_post_act.as_ref(ctx).as_ptr(),
        weights2.as_view(ctx).as_ptr(),
        0.0,
        tmp2_pre_act.as_ref_mut(ctx).as_mut_ptr(),
        workspace.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("conv2d forward failed: {:?}", e); }
    }
    /*self.add_bias2.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias2.forward(
        bias2.as_view(ctx).as_ptr(),
        tmp2_pre_act.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };*/

    match phase {
      OpPhase::Inference => {
        self.batchnorm2.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm2.forward_inference(
            1.0,
            tmp2_pre_act.as_ref(ctx).as_ptr(),
            1.0,
            out_act.as_ref_mut(ctx).as_mut_ptr(),
            self.config.bnorm_epsilon,
            self.bn_scale2.as_view(ctx).as_ptr(),
            self.bn_bias2.as_view(ctx).as_ptr(),
            self.bn_running_mean2.as_view(ctx).as_ptr(),
            self.bn_running_ivar2.as_view(ctx).as_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
      OpPhase::Training => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        let ema_factor = match backward.first_batch2 {
          true  => {
            backward.first_batch2 = false;
            1.0
          }
          false => 0.01,
        };
        self.batchnorm2.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm2.forward_training(
            1.0,
            tmp2_pre_act.as_ref(ctx).as_ptr(),
            1.0,
            out_act.as_ref_mut(ctx).as_mut_ptr(),
            ema_factor,
            self.config.bnorm_epsilon,
            self.bn_scale2.as_view(ctx).as_ptr(),
            self.bn_bias2.as_view(ctx).as_ptr(),
            self.bn_running_mean2.as_view_mut(ctx).as_mut_ptr(),
            self.bn_running_ivar2.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_mean2.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_ivar2.as_view_mut(ctx).as_mut_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
    }

    /*self.add_input.set_batch_size(batch_size).unwrap();
    unsafe { self.add_input.forward(
        in_act.borrow_mut().as_ref(ctx).as_ptr(),
        out_act.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };*/

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_inplace(
            out_act.as_ref_mut(ctx).as_mut_ptr(),
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
    /*let BotResConv2dOperatorConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = self.config;*/
    //let (in_width, in_height, in_channels) = in_dims;
    //let in_length = in_dims.len();
    //let out_dims = self.config.get_out_dims();
    let out_dims = self.config.in_dims;
    //let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    // FIXME(20160413): the bottleneck residual consists of 3 convolutions;
    // the output delta is copied to the input delta before applying the conv
    // backward pass.

    let &mut StackResConv2dOperator{
      ref context,
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights1, //ref mut bias,
      ref mut weights2, //ref mut bias,
      ref mut tmp1_pre_act, ref mut tmp1_pre_delta,
      ref mut tmp1_post_act, ref mut tmp1_post_delta,
      ref mut tmp2_pre_act, ref mut tmp2_pre_delta,
      ref mut workspace,
      ref mut backward,
      .. } = self;
    let mut backward = backward.as_mut().unwrap();
    let &mut StackResConv2dBwdOperator{
      ref mut grad_weights1, ref mut grad_bias1,
      ref mut grad_weights2, ref mut grad_bias2,
      .. } = backward;

    let ctx = &(**context).as_ref();
    let in_act = in_act.borrow_mut().as_ref(ctx);
    //let tmp_act = tmp_act.as_ref(ctx);
    let out_act = out_act.borrow_mut().as_ref(ctx);
    let mut workspace = workspace.as_ref_mut(ctx);

    {
      let mut out_delta = out_delta.borrow_mut().as_ref_mut(ctx);
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
    }

    let out_delta = out_delta.borrow_mut().as_ref(ctx);

    // FIXME(20160420): backward pass of residuals.

    {
      self.batchnorm2.set_batch_size(batch_size).unwrap();
      unsafe { self.batchnorm2.backward(
          1.0, 0.0,
          tmp2_pre_act.as_ref(ctx).as_ptr(),
          out_delta.as_ptr(),
          tmp2_pre_delta.as_ref_mut(ctx).as_mut_ptr(),
          1.0, 1.0,
          self.config.bnorm_epsilon,
          self.bn_scale2.as_view(ctx).as_ptr(),
          self.bn_scale2_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_bias2_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_cached_mean2.as_view(ctx).as_ptr(),
          self.bn_cached_ivar2.as_view(ctx).as_ptr(),
          &*ctx.get_dnn(),
      ) }.unwrap();
    }

    {
      let tmp2_pre_delta = tmp2_pre_delta.as_ref(ctx);
      backward.conv2_bwd_w.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv2_bwd_w.backward_filter(
          1.0,
          tmp1_post_act.as_ref(ctx).as_ptr(),
          tmp2_pre_delta.as_ptr(),
          1.0,
          grad_weights2.as_view_mut(ctx).as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
      /*unsafe { backward.conv2_bwd_w.backward_bias(
          1.0,
          tmp2_pre_delta.as_ptr(),
          1.0,
          grad_bias2.as_view_mut(ctx).as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };*/
    }

    {
      let mut tmp1_post_delta = tmp1_post_delta.as_ref_mut(ctx);

      backward.conv2_bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv2_bwd_d.backward_data(
          1.0,
          weights2.as_view(ctx).as_ptr(),
          tmp2_pre_delta.as_ref(ctx).as_ptr(),
          0.0,
          tmp1_post_delta.as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };

      match self.config.act_func {
        ActivationFunction::Identity => {}
        ActivationFunction::Rect => {
          unsafe { rembrandt_kernel_batch_map_rect_backprop_inplace(
              tmp1_post_act.as_ref(ctx).as_ptr(),
              out_length as i32,
              batch_size as i32,
              tmp1_post_delta.as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
        _ => unimplemented!(),
      }
    }

    {
      let epsilon = 1.0e-4;
      self.batchnorm1.set_batch_size(batch_size).unwrap();
      unsafe { self.batchnorm1.backward(
          1.0, 0.0,
          tmp1_pre_act.as_ref(ctx).as_ptr(),
          tmp1_post_delta.as_ref(ctx).as_ptr(),
          tmp1_pre_delta.as_ref_mut(ctx).as_mut_ptr(),
          1.0, 1.0,
          self.config.bnorm_epsilon,
          self.bn_scale1.as_view(ctx).as_ptr(),
          self.bn_scale1_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_bias1_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_cached_mean1.as_view(ctx).as_ptr(),
          self.bn_cached_ivar1.as_view(ctx).as_ptr(),
          &*ctx.get_dnn(),
      ) }.unwrap();
    }

    {
      let tmp1_pre_delta = tmp1_pre_delta.as_ref(ctx);
      backward.conv1_bwd_w.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv1_bwd_w.backward_filter(
          1.0,
          in_act.as_ptr(),
          tmp1_pre_delta.as_ptr(),
          1.0,
          grad_weights1.as_view_mut(ctx).as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
      /*unsafe { backward.conv1_bwd_w.backward_bias(
          1.0,
          tmp1_pre_delta.as_ptr(),
          1.0,
          grad_bias1.as_view_mut(ctx).as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };*/
    }

    if let &mut Some(ref mut in_delta) = in_delta {
      let tmp1_pre_delta = tmp1_pre_delta.as_ref(ctx);
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);

      out_delta.send(&mut in_delta);

      backward.conv1_bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv1_bwd_d.backward_data(
          1.0,
          weights1.as_view(ctx).as_ptr(),
          tmp1_pre_delta.as_ptr(),
          1.0,
          in_delta.as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
    }
  }

  fn regularize(&mut self, reg: Regularization) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    match reg {
      Regularization::L2{l2_reg_coef} => {
        assert!(l2_reg_coef >= 0.0);
        if l2_reg_coef > 0.0 {
          backward.grad_weights1.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights1.as_view(ctx));
          backward.grad_bias1.as_view_mut(ctx)
            .row_vector_sum(l2_reg_coef, &self.bias1.as_view(ctx));
          backward.grad_weights2.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights2.as_view(ctx));
          backward.grad_bias2.as_view_mut(ctx)
            .row_vector_sum(l2_reg_coef, &self.bias2.as_view(ctx));
          // XXX(20160420): Don't regularize the batch-normalization params.
        }
      }
    }
  }

  fn accumulate_grads(&mut self, scale: f32, momentum: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.acc_grad_weights1.as_view_mut(ctx)
      .matrix_scale(momentum);
    backward.acc_grad_bias1.as_view_mut(ctx)
      .row_vector_scale(momentum);
    backward.acc_grad_weights1.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights1.as_view(ctx));
    backward.acc_grad_bias1.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.grad_bias1.as_view(ctx));

    self.acc_bn_scale1_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_scale1_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_scale1_grad.as_view(ctx));
    self.acc_bn_bias1_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_bias1_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_bias1_grad.as_view(ctx));

    backward.acc_grad_weights2.as_view_mut(ctx)
      .matrix_scale(momentum);
    backward.acc_grad_bias2.as_view_mut(ctx)
      .row_vector_scale(momentum);
    backward.acc_grad_weights2.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights2.as_view(ctx));
    backward.acc_grad_bias2.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.grad_bias2.as_view(ctx));

    self.acc_bn_scale2_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_scale2_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_scale2_grad.as_view(ctx));
    self.acc_bn_bias2_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_bias2_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_bias2_grad.as_view(ctx));
  }

  fn update_params(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    self.weights1.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights1.as_view(ctx));
    self.bias1.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.acc_grad_bias1.as_view(ctx));

    self.bn_scale1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale1_grad.as_view(ctx));
    self.bn_bias1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias1_grad.as_view(ctx));

    self.weights2.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights2.as_view(ctx));
    self.bias2.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.acc_grad_bias2.as_view(ctx));

    self.bn_scale2.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale2_grad.as_view(ctx));
    self.bn_bias2.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias2_grad.as_view(ctx));
  }

  fn save_params(&mut self) {
    unimplemented!();
  }

  fn restore_params(&mut self) {
    unimplemented!();
  }

  fn set_grads_with_params_diff(&mut self) {
    unimplemented!();
  }

  fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.load(self.params_off, &mut self.weights1, ctx);
    comm_worker.load(self.params_off, &mut self.bias1, ctx);
    comm_worker.load(self.params_off, &mut self.weights2, ctx);
    comm_worker.load(self.params_off, &mut self.bias2, ctx);
    // FIXME(20160420): batch norm params.
  }

  fn sync_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights1, ctx);
    comm_worker.store(self.params_off, &mut self.bias1, ctx);
    comm_worker.store(self.params_off, &mut self.weights2, ctx);
    comm_worker.store(self.params_off, &mut self.bias2, ctx);
    // FIXME(20160420): batch norm params.
  }

  fn reset_grads(&mut self, scale: f32) {
    unimplemented!();
  }

  fn reset(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.grad_weights1.as_view_mut(ctx)
      .matrix_scale(0.0);
    backward.grad_bias1.as_view_mut(ctx)
      .row_vector_scale(0.0);

    self.bn_scale1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
    self.bn_bias1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);

    backward.grad_weights2.as_view_mut(ctx)
      .matrix_scale(0.0);
    backward.grad_bias2.as_view_mut(ctx)
      .row_vector_scale(0.0);

    self.bn_scale2_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
    self.bn_bias2_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
  }
}

#[derive(Clone, Copy, Debug)]
pub struct ProjStackResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub out_dims:     (usize, usize, usize),
  pub bnorm_mov_avg:    BnormMovingAverage,
  pub bnorm_epsilon:    f64,
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}

impl ProjStackResConv2dOperatorConfig {
  pub fn params_len(&self) -> usize {
    let (_, _, in_channels) = self.in_dims;
    let (_, _, out_channels) = self.out_dims;
    let weights1_len = 3 * 3 * in_channels * out_channels;
    let bias1_len = out_channels;
    let weights2_len = 3 * 3 * out_channels * out_channels;
    let bias2_len = out_channels;
    let weights3_len = 1 * 1 * in_channels * out_channels;
    let bias3_len = out_channels;
    weights1_len + bias1_len +
        weights2_len + bias2_len +
        weights3_len + bias3_len
  }
}

pub struct ProjStackResConv2dOperator<Comm> {
  batch_cap:    usize,
  _capability:  OpCapability,
  params_off:   usize,
  config:       ProjStackResConv2dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights1:     DeviceArray2d<f32>,
  bias1:        DeviceArray2d<f32>,
  weights2:     DeviceArray2d<f32>,
  bias2:        DeviceArray2d<f32>,
  weights3:     DeviceArray2d<f32>,
  bias3:        DeviceArray2d<f32>,

  /*tmp_act:      DeviceBuffer<f32>,
  tmp_delta:    DeviceBuffer<f32>,*/
  tmp1_pre_act:     DeviceBuffer<f32>,
  tmp1_pre_delta:   DeviceBuffer<f32>,
  tmp1_post_act:    DeviceBuffer<f32>,
  tmp1_post_delta:  DeviceBuffer<f32>,
  tmp2_pre_act:     DeviceBuffer<f32>,
  tmp2_pre_delta:   DeviceBuffer<f32>,
  tmp3_pre_act:     DeviceBuffer<f32>,
  tmp3_pre_delta:   DeviceBuffer<f32>,

  workspace:    DeviceBuffer<u8>,
  conv1_fwd:    CudnnConvFwdOp,
  add_bias1:    CudnnAddOp,
  conv2_fwd:    CudnnConvFwdOp,
  add_bias2:    CudnnAddOp,
  conv3_fwd:    CudnnConvFwdOp,
  add_bias3:    CudnnAddOp,
  //add_input:    CudnnAddOp,

  bn_scale1:            DeviceArray2d<f32>,
  bn_scale1_grad:       DeviceArray2d<f32>,
  acc_bn_scale1_grad:   DeviceArray2d<f32>,
  bn_bias1:             DeviceArray2d<f32>,
  bn_bias1_grad:        DeviceArray2d<f32>,
  acc_bn_bias1_grad:    DeviceArray2d<f32>,
  bn_running_mean1:     DeviceArray2d<f32>,
  bn_running_ivar1:     DeviceArray2d<f32>,
  bn_cached_mean1:      DeviceArray2d<f32>,
  bn_cached_ivar1:      DeviceArray2d<f32>,
  batchnorm1:           CudnnBatchNormOp,

  bn_scale2:            DeviceArray2d<f32>,
  bn_scale2_grad:       DeviceArray2d<f32>,
  acc_bn_scale2_grad:   DeviceArray2d<f32>,
  bn_bias2:             DeviceArray2d<f32>,
  bn_bias2_grad:        DeviceArray2d<f32>,
  acc_bn_bias2_grad:    DeviceArray2d<f32>,
  bn_running_mean2:     DeviceArray2d<f32>,
  bn_running_ivar2:     DeviceArray2d<f32>,
  bn_cached_mean2:      DeviceArray2d<f32>,
  bn_cached_ivar2:      DeviceArray2d<f32>,
  batchnorm2:           CudnnBatchNormOp,

  bn_scale3:            DeviceArray2d<f32>,
  bn_scale3_grad:       DeviceArray2d<f32>,
  acc_bn_scale3_grad:   DeviceArray2d<f32>,
  bn_bias3:             DeviceArray2d<f32>,
  bn_bias3_grad:        DeviceArray2d<f32>,
  acc_bn_bias3_grad:    DeviceArray2d<f32>,
  bn_running_mean3:     DeviceArray2d<f32>,
  bn_running_ivar3:     DeviceArray2d<f32>,
  bn_cached_mean3:      DeviceArray2d<f32>,
  bn_cached_ivar3:      DeviceArray2d<f32>,
  batchnorm3:           CudnnBatchNormOp,

  backward:     Option<ProjStackResConv2dBwdOperator<Comm>>,
  //hv_backward:  Option<BotResConv2dHvBwdOperator>,
}

struct ProjStackResConv2dBwdOperator<Comm> {
  grad_weights1:      DeviceArray2d<f32>,
  grad_bias1:      DeviceArray2d<f32>,
  acc_grad_weights1:  DeviceArray2d<f32>,
  acc_grad_bias1:  DeviceArray2d<f32>,
  conv1_bwd_w:  CudnnConvBwdFilterOp,
  conv1_bwd_d:  CudnnConvBwdDataOp,

  first_batch1:    bool,

  grad_weights2:      DeviceArray2d<f32>,
  grad_bias2:      DeviceArray2d<f32>,
  acc_grad_weights2:  DeviceArray2d<f32>,
  acc_grad_bias2:  DeviceArray2d<f32>,
  conv2_bwd_w:  CudnnConvBwdFilterOp,
  conv2_bwd_d:  CudnnConvBwdDataOp,

  first_batch2:    bool,

  grad_weights3:      DeviceArray2d<f32>,
  grad_bias3:      DeviceArray2d<f32>,
  acc_grad_weights3:  DeviceArray2d<f32>,
  acc_grad_bias3:  DeviceArray2d<f32>,
  conv3_bwd_w:  CudnnConvBwdFilterOp,
  conv3_bwd_d:  CudnnConvBwdDataOp,

  first_batch3:    bool,

  comm_worker:  Rc<RefCell<Comm>>,
}

impl<Comm> ProjStackResConv2dOperator<Comm> where Comm: CommWorker {
  pub fn new(batch_size: usize, capability: OpCapability, params_offset: usize, config: ProjStackResConv2dOperatorConfig, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> ProjStackResConv2dOperator<Comm> {
    let ProjStackResConv2dOperatorConfig{
      in_dims, out_dims,
      .. } = config;
    let (in_width, in_height, in_channels) = in_dims;
    let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    // FIXME(20160420): a hack, but should always be satisfied in usual archs.
    let conv1_stride = if in_width == out_width && in_height == out_height {
      1
    } else if in_width == 2 * out_width && in_height == 2 * out_height {
      2
    } else {
      unimplemented!();
    };

    let ctx = &(*context).as_ref();

    let mut workspace_size = 0;

    // FIXME(20160420): support batch norm (the "spatial" variant).

    let conv1_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels, out_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv1_stride, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv1_fwd.work_size);

    let conv2_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(3, 3, out_channels, out_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv2_fwd.work_size);

    let conv3_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(1, 1, in_channels, out_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv1_stride, 0).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv3_fwd.work_size);

    let batchnorm1 = CudnnBatchNormOp::new(
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        cudnnBatchNormMode_t::Spatial,
    );

    let batchnorm2 = CudnnBatchNormOp::new(
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        cudnnBatchNormMode_t::Spatial,
    );

    let batchnorm3 = CudnnBatchNormOp::new(
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        cudnnBatchNormMode_t::Spatial,
    );

    let backward = if capability.backward_enabled() {
      let conv1_bwd_w = CudnnConvBwdFilterOp::create_fastest(
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(conv1_stride, 1).unwrap(),
          CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv1_bwd_w.work_size);

      let conv1_bwd_d = CudnnConvBwdDataOp::create_fastest(
          CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(conv1_stride, 1).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv1_bwd_d.work_size);

      let conv2_bwd_w = CudnnConvBwdFilterOp::create_fastest(
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
          CudnnFilterDesc::<f32>::create_4d(3, 3, out_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv2_bwd_w.work_size);

      let conv2_bwd_d = CudnnConvBwdDataOp::create_fastest(
          CudnnFilterDesc::<f32>::create_4d(3, 3, out_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv2_bwd_d.work_size);

      let conv3_bwd_w = CudnnConvBwdFilterOp::create_fastest(
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(conv1_stride, 0).unwrap(),
          CudnnFilterDesc::<f32>::create_4d(1, 1, in_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv3_bwd_w.work_size);

      let conv3_bwd_d = CudnnConvBwdDataOp::create_fastest(
          CudnnFilterDesc::<f32>::create_4d(1, 1, in_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(conv1_stride, 0).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv3_bwd_d.work_size);

      Some(ProjStackResConv2dBwdOperator{
        grad_weights1:      DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, out_channels), ctx),
        grad_bias1:         DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        acc_grad_weights1:  DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, out_channels), ctx),
        acc_grad_bias1:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),

        grad_weights2:      DeviceArray2d::<f32>::zeros((3 * 3 * out_channels, out_channels), ctx),
        grad_bias2:         DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        acc_grad_weights2:  DeviceArray2d::<f32>::zeros((3 * 3 * out_channels, out_channels), ctx),
        acc_grad_bias2:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),

        grad_weights3:      DeviceArray2d::<f32>::zeros((1 * 1 * in_channels, out_channels), ctx),
        grad_bias3:         DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        acc_grad_weights3:  DeviceArray2d::<f32>::zeros((1 * 1 * in_channels, out_channels), ctx),
        acc_grad_bias3:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),

        conv1_bwd_w:   conv1_bwd_w,
        conv1_bwd_d:   conv1_bwd_d,
        conv2_bwd_w:   conv2_bwd_w,
        conv2_bwd_d:   conv2_bwd_d,
        conv3_bwd_w:   conv3_bwd_w,
        conv3_bwd_d:   conv3_bwd_d,

        first_batch1:  true,
        first_batch2:  true,
        first_batch3:  true,

        comm_worker:  comm_worker.unwrap(),
      })
    } else {
      None
    };

    let add_bias1 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
    );
    let add_bias2 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
    );
    let add_bias3 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
    );
    /*let add_input = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
    );*/

    // XXX(20160421): Initialize gammas to all ones.
    let mut bn_scale1 = DeviceArray2d::<f32>::zeros((1, out_channels), ctx);
    bn_scale1.as_view_mut(ctx).set_constant(1.0);
    let mut bn_scale2 = DeviceArray2d::<f32>::zeros((1, out_channels), ctx);
    bn_scale2.as_view_mut(ctx).set_constant(1.0);
    let mut bn_scale3 = DeviceArray2d::<f32>::zeros((1, out_channels), ctx);
    bn_scale3.as_view_mut(ctx).set_constant(1.0);

    ProjStackResConv2dOperator{
      batch_cap:    batch_size,
      _capability:  capability,
      params_off:   params_offset,
      config:       config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("ProjStackResConv2dOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      weights1:     DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, out_channels), ctx),
      bias1:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      weights2:     DeviceArray2d::<f32>::zeros((3 * 3 * out_channels, out_channels), ctx),
      bias2:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      weights3:     DeviceArray2d::<f32>::zeros((1 * 1 * in_channels, out_channels), ctx),
      bias3:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),

      /*tmp_act:      DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp_delta:    DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),*/
      tmp1_pre_act:     DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp1_pre_delta:   DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp1_post_act:    DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp1_post_delta:  DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp2_pre_act:     DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp2_pre_delta:   DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp3_pre_act:     DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp3_pre_delta:   DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),

      workspace:    DeviceBuffer::<u8>::zeros(workspace_size, ctx),
      conv1_fwd:    conv1_fwd,
      add_bias1:    add_bias1,
      conv2_fwd:    conv2_fwd,
      add_bias2:    add_bias2,
      conv3_fwd:    conv3_fwd,
      add_bias3:    add_bias3,
      //add_input:    add_input,

      //bn_scale1:            DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_scale1:            bn_scale1,
      bn_scale1_grad:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_scale1_grad:   DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias1:             DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias1_grad:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_bias1_grad:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_mean1:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_ivar1:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_mean1:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_ivar1:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      batchnorm1:           batchnorm1,

      //bn_scale2:            DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_scale2:            bn_scale2,
      bn_scale2_grad:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_scale2_grad:   DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias2:             DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias2_grad:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_bias2_grad:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_mean2:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_ivar2:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_mean2:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_ivar2:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      batchnorm2:           batchnorm2,

      //bn_scale3:            DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_scale3:            bn_scale3,
      bn_scale3_grad:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_scale3_grad:   DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias3:             DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias3_grad:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_bias3_grad:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_mean3:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_ivar3:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_mean3:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_ivar3:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      batchnorm3:           batchnorm3,

      backward:     backward,
      //hv_backward:  None,
    }
  }
}

impl<Comm> Operator for ProjStackResConv2dOperator<Comm> where Comm: CommWorker {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn init_params(&mut self, shared_seed: [u64; 2]) {
    let ProjStackResConv2dOperatorConfig{in_dims, out_dims, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let (_, _, out_channels) = out_dims;
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    let mut init_weights1 = Array2d::zeros((3 * 3 * in_channels, out_channels));
    let mut init_weights2 = Array2d::zeros((3 * 3 * out_channels, out_channels));
    let mut init_weights3 = Array2d::zeros((1 * 1 * in_channels, out_channels));
    match self.config.init_weights {
      ParamsInit::Disabled => {
        panic!("StackResConv2dOperator: params init explicitly disabled");
      }
      ParamsInit::Uniform{half_range} => {
        let dist = Range::new(-half_range as f64, half_range as f64);
        for w in init_weights1.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
        for w in init_weights2.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
        for w in init_weights3.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInit::Normal{std} => {
        let dist = Normal::new(0.0, std as f64);
        for w in init_weights1.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
        for w in init_weights2.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
        for w in init_weights3.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInit::Xavier => {
        // FIXME(20160420)
        unimplemented!();
      }
      ParamsInit::KaimingFwd => {
        let in_conns = 3 * 3 * self.config.in_dims.2;
        let std = (2.0 / in_conns as f64).sqrt();
        let dist = Normal::new(0.0, std as f64);
        for w in init_weights1.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
        let in_conns = 3 * 3 * self.config.out_dims.2;
        let std = (2.0 / in_conns as f64).sqrt();
        let dist = Normal::new(0.0, std as f64);
        for w in init_weights2.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
        let in_conns = 1 * 1 * self.config.in_dims.2;
        let std = (2.0 / in_conns as f64).sqrt();
        let dist = Normal::new(0.0, std as f64);
        for w in init_weights3.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
    }
    let init_bias = Array2d::zeros((1, out_channels));
    self.weights1.as_view_mut(ctx).sync_load(&init_weights1.as_view());
    self.bias1.as_view_mut(ctx).sync_load(&init_bias.as_view());
    self.weights2.as_view_mut(ctx).sync_load(&init_weights2.as_view());
    self.bias2.as_view_mut(ctx).sync_load(&init_bias.as_view());
    self.weights3.as_view_mut(ctx).sync_load(&init_weights3.as_view());
    self.bias3.as_view_mut(ctx).sync_load(&init_bias.as_view());
  }

  fn read_params(&mut self, blob: &[u8]) -> usize {
    let ProjStackResConv2dOperatorConfig{in_dims, out_dims, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let (_, _, out_channels) = out_dims;
    let mut reader = Cursor::new(blob);
    let load_weights1 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize weights!");
    let load_bias1 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize bias!");
    let load_weights2 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize weights!");
    let load_bias2 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize bias!");
    let load_weights3 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize weights!");
    let load_bias3 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize bias!");
    assert_eq!((3 * 3 * in_channels, out_channels), load_weights1.as_view().bound());
    assert_eq!((1, out_channels), load_bias1.as_view().bound());
    assert_eq!((3 * 3 * out_channels, out_channels), load_weights2.as_view().bound());
    assert_eq!((1, out_channels), load_bias2.as_view().bound());
    assert_eq!((1 * 1 * in_channels, out_channels), load_weights3.as_view().bound());
    assert_eq!((1, out_channels), load_bias3.as_view().bound());
    self.weights1.as_view_mut(ctx).sync_load(&load_weights1.as_view());
    self.bias1.as_view_mut(ctx).sync_load(&load_bias1.as_view());
    self.weights2.as_view_mut(ctx).sync_load(&load_weights2.as_view());
    self.bias2.as_view_mut(ctx).sync_load(&load_bias2.as_view());
    self.weights3.as_view_mut(ctx).sync_load(&load_weights3.as_view());
    self.bias3.as_view_mut(ctx).sync_load(&load_bias3.as_view());
    let progress = reader.position() as usize;
    progress
  }

  fn write_params(&mut self, blob: &mut Vec<u8>) {
    let ctx = &(*self.context).as_ref();

    let weights1 = self.weights1.as_view(ctx);
    let bias1 = self.bias1.as_view(ctx);
    let mut save_weights1 = Array2d::zeros(weights1.bound());
    let mut save_bias1 = Array2d::zeros(bias1.bound());
    weights1.sync_store(&mut save_weights1.as_view_mut());
    bias1.sync_store(&mut save_bias1.as_view_mut());
    save_weights1.serialize(blob).unwrap();
    save_bias1.serialize(blob).unwrap();

    let weights2 = self.weights2.as_view(ctx);
    let bias2 = self.bias2.as_view(ctx);
    let mut save_weights2 = Array2d::zeros(weights2.bound());
    let mut save_bias2 = Array2d::zeros(bias2.bound());
    weights2.sync_store(&mut save_weights2.as_view_mut());
    bias2.sync_store(&mut save_bias2.as_view_mut());
    save_weights2.serialize(blob).unwrap();
    save_bias2.serialize(blob).unwrap();

    let weights3 = self.weights3.as_view(ctx);
    let bias3 = self.bias3.as_view(ctx);
    let mut save_weights3 = Array2d::zeros(weights3.bound());
    let mut save_bias3 = Array2d::zeros(bias3.bound());
    weights3.sync_store(&mut save_weights3.as_view_mut());
    bias3.sync_store(&mut save_bias3.as_view_mut());
    save_weights3.serialize(blob).unwrap();
    save_bias3.serialize(blob).unwrap();
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let out_dims = self.config.out_dims;
    //let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    // FIXME(20160413): the bottleneck residual consists of 3 convolutions.

    let &mut ProjStackResConv2dOperator{
      ref context,
      ref mut in_act, ref mut out_act,
      ref mut weights1, ref mut bias1,
      ref mut weights2, ref mut bias2,
      ref mut weights3, ref mut bias3,
      //ref mut tmp_act,
      ref mut tmp1_pre_act,
      ref mut tmp1_post_act,
      ref mut tmp2_pre_act,
      ref mut tmp3_pre_act,
      ref mut workspace,
      .. } = self;

    let ctx = &(**context).as_ref();
    let mut out_act = out_act.borrow_mut();

    self.conv1_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv1_fwd.forward(
        1.0,
        in_act.borrow_mut().as_ref(ctx).as_ptr(),
        weights1.as_view(ctx).as_ptr(),
        0.0,
        tmp1_pre_act.as_ref_mut(ctx).as_mut_ptr(),
        workspace.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("conv2d forward failed: {:?}", e); }
    }

    /*self.add_bias1.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias1.forward(
        1.0,
        bias1.as_view(ctx).as_ptr(),
        1.0,
        tmp_act.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };*/

    match phase {
      OpPhase::Inference => {
        self.batchnorm1.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm1.forward_inference(
            1.0,
            tmp1_pre_act.as_ref(ctx).as_ptr(),
            0.0,
            tmp1_post_act.as_ref_mut(ctx).as_mut_ptr(),
            self.config.bnorm_epsilon,
            self.bn_scale1.as_view(ctx).as_ptr(),
            self.bn_bias1.as_view(ctx).as_ptr(),
            self.bn_running_mean1.as_view(ctx).as_ptr(),
            self.bn_running_ivar1.as_view(ctx).as_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
      OpPhase::Training => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        let ema_factor = match backward.first_batch1 {
          true  => {
            backward.first_batch1 = false;
            1.0
          }
          false => 0.01,
        };
        self.batchnorm1.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm1.forward_training(
            1.0,
            tmp1_pre_act.as_ref(ctx).as_ptr(),
            0.0,
            tmp1_post_act.as_ref_mut(ctx).as_mut_ptr(),
            ema_factor,
            self.config.bnorm_epsilon,
            self.bn_scale1.as_view(ctx).as_ptr(),
            self.bn_bias1.as_view(ctx).as_ptr(),
            self.bn_running_mean1.as_view_mut(ctx).as_mut_ptr(),
            self.bn_running_ivar1.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_mean1.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_ivar1.as_view_mut(ctx).as_mut_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
    }

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_inplace(
            tmp1_post_act.as_ref_mut(ctx).as_mut_ptr(),
            out_length as i32,
            batch_size as i32,
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }

    self.conv2_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv2_fwd.forward(
        1.0,
        tmp1_post_act.as_ref(ctx).as_ptr(),
        weights2.as_view(ctx).as_ptr(),
        0.0,
        tmp2_pre_act.as_ref_mut(ctx).as_mut_ptr(),
        workspace.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("conv2d forward failed: {:?}", e); }
    }

    /*self.add_bias2.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias2.forward(
        1.0,
        bias2.as_view(ctx).as_ptr(),
        1.0,
        out_act.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };*/

    match phase {
      OpPhase::Inference => {
        self.batchnorm2.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm2.forward_inference(
            1.0,
            tmp2_pre_act.as_ref(ctx).as_ptr(),
            0.0,
            out_act.as_ref_mut(ctx).as_mut_ptr(),
            self.config.bnorm_epsilon,
            self.bn_scale2.as_view(ctx).as_ptr(),
            self.bn_bias2.as_view(ctx).as_ptr(),
            self.bn_running_mean2.as_view(ctx).as_ptr(),
            self.bn_running_ivar2.as_view(ctx).as_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
      OpPhase::Training => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        let ema_factor = match backward.first_batch2 {
          true  => {
            backward.first_batch2 = false;
            1.0
          }
          false => 0.01,
        };
        self.batchnorm2.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm2.forward_training(
            1.0,
            tmp2_pre_act.as_ref(ctx).as_ptr(),
            0.0,
            out_act.as_ref_mut(ctx).as_mut_ptr(),
            ema_factor,
            self.config.bnorm_epsilon,
            self.bn_scale2.as_view(ctx).as_ptr(),
            self.bn_bias2.as_view(ctx).as_ptr(),
            self.bn_running_mean2.as_view_mut(ctx).as_mut_ptr(),
            self.bn_running_ivar2.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_mean2.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_ivar2.as_view_mut(ctx).as_mut_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
    }

    self.conv3_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv3_fwd.forward(
        1.0,
        in_act.borrow_mut().as_ref(ctx).as_ptr(),
        weights3.as_view(ctx).as_ptr(),
        0.0,
        tmp3_pre_act.as_ref_mut(ctx).as_mut_ptr(),
        workspace.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("conv2d forward failed: {:?}", e); }
    }

    /*self.add_bias3.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias3.forward(
        1.0,
        bias3.as_view(ctx).as_ptr(),
        1.0,
        out_act.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };*/

    match phase {
      OpPhase::Inference => {
        self.batchnorm3.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm3.forward_inference(
            1.0,
            tmp3_pre_act.as_ref(ctx).as_ptr(),
            1.0,
            out_act.as_ref_mut(ctx).as_mut_ptr(),
            self.config.bnorm_epsilon,
            self.bn_scale3.as_view(ctx).as_ptr(),
            self.bn_bias3.as_view(ctx).as_ptr(),
            self.bn_running_mean3.as_view(ctx).as_ptr(),
            self.bn_running_ivar3.as_view(ctx).as_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
      OpPhase::Training => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        let ema_factor = match backward.first_batch3 {
          true  => {
            backward.first_batch3 = false;
            1.0
          }
          false => 0.01,
        };
        self.batchnorm3.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm3.forward_training(
            1.0,
            tmp3_pre_act.as_ref(ctx).as_ptr(),
            1.0,
            out_act.as_ref_mut(ctx).as_mut_ptr(),
            ema_factor,
            self.config.bnorm_epsilon,
            self.bn_scale3.as_view(ctx).as_ptr(),
            self.bn_bias3.as_view(ctx).as_ptr(),
            self.bn_running_mean3.as_view_mut(ctx).as_mut_ptr(),
            self.bn_running_ivar3.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_mean3.as_view_mut(ctx).as_mut_ptr(),
            self.bn_cached_ivar3.as_view_mut(ctx).as_mut_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
    }

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_inplace(
            out_act.as_ref_mut(ctx).as_mut_ptr(),
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
    let out_dims = self.config.out_dims;
    //let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    // FIXME(20160413): the bottleneck residual consists of 3 convolutions;
    // the output delta is copied to the input delta before applying the conv
    // backward pass.

    let &mut ProjStackResConv2dOperator{
      ref context,
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights1, //ref mut bias,
      ref mut weights2, //ref mut bias,
      ref mut weights3, //ref mut bias,
      ref mut tmp1_pre_act,
      ref mut tmp1_pre_delta,
      ref mut tmp1_post_act,
      ref mut tmp1_post_delta,
      ref mut tmp2_pre_act,
      ref mut tmp2_pre_delta,
      ref mut tmp3_pre_act,
      ref mut tmp3_pre_delta,
      ref mut workspace,
      ref mut backward,
      .. } = self;
    let mut backward = backward.as_mut().unwrap();
    let &mut ProjStackResConv2dBwdOperator{
      ref mut grad_weights1, ref mut grad_bias1,
      ref mut grad_weights2, ref mut grad_bias2,
      ref mut grad_weights3, ref mut grad_bias3,
      .. } = backward;

    let ctx = &(**context).as_ref();
    let in_act = in_act.borrow_mut().as_ref(ctx);
    //let tmp_act = tmp_act.as_ref(ctx);
    let out_act = out_act.borrow_mut().as_ref(ctx);
    let mut workspace = workspace.as_ref_mut(ctx);

    {
      let mut out_delta = out_delta.borrow_mut().as_ref_mut(ctx);
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
    }

    let out_delta = out_delta.borrow_mut().as_ref(ctx);

    // FIXME(20160420): backward pass of residuals.

    {
      self.batchnorm2.set_batch_size(batch_size).unwrap();
      unsafe { self.batchnorm2.backward(
          1.0, 0.0,
          tmp2_pre_act.as_ref(ctx).as_ptr(),
          out_delta.as_ptr(),
          tmp2_pre_delta.as_ref_mut(ctx).as_mut_ptr(),
          1.0, 1.0,
          self.config.bnorm_epsilon,
          self.bn_scale2.as_view(ctx).as_ptr(),
          self.bn_scale2_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_bias2_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_cached_mean2.as_view(ctx).as_ptr(),
          self.bn_cached_ivar2.as_view(ctx).as_ptr(),
          &*ctx.get_dnn(),
      ) }.unwrap();
    }

    {
      backward.conv2_bwd_w.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv2_bwd_w.backward_filter(
          1.0,
          tmp1_post_act.as_ref(ctx).as_ptr(),
          tmp2_pre_delta.as_ref(ctx).as_ptr(),
          1.0,
          grad_weights2.as_view_mut(ctx).as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
      /*unsafe { backward.conv2_bwd_w.backward_bias(
          1.0,
          out_delta.as_ptr(),
          1.0,
          grad_bias2.as_view_mut(ctx).as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };*/
    }

    {
      let mut tmp1_post_delta = tmp1_post_delta.as_ref_mut(ctx);

      backward.conv2_bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv2_bwd_d.backward_data(
          1.0,
          weights2.as_view(ctx).as_ptr(),
          tmp2_pre_delta.as_ref(ctx).as_ptr(),
          0.0,
          tmp1_post_delta.as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };

      match self.config.act_func {
        ActivationFunction::Identity => {}
        ActivationFunction::Rect => {
          unsafe { rembrandt_kernel_batch_map_rect_backprop_inplace(
              tmp1_post_act.as_ref(ctx).as_ptr(),
              out_length as i32,
              batch_size as i32,
              tmp1_post_delta.as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
        _ => unimplemented!(),
      }
    }

    {
      self.batchnorm1.set_batch_size(batch_size).unwrap();
      unsafe { self.batchnorm1.backward(
          1.0, 0.0,
          tmp1_pre_act.as_ref(ctx).as_ptr(),
          tmp1_post_delta.as_ref(ctx).as_ptr(),
          tmp1_pre_delta.as_ref_mut(ctx).as_mut_ptr(),
          1.0, 1.0,
          self.config.bnorm_epsilon,
          self.bn_scale1.as_view(ctx).as_ptr(),
          self.bn_scale1_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_bias1_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_cached_mean1.as_view(ctx).as_ptr(),
          self.bn_cached_ivar1.as_view(ctx).as_ptr(),
          &*ctx.get_dnn(),
      ) }.unwrap();
    }

    {
      //let tmp_delta = tmp_delta.as_ref(ctx);
      backward.conv1_bwd_w.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv1_bwd_w.backward_filter(
          1.0,
          in_act.as_ptr(),
          tmp1_pre_delta.as_ref(ctx).as_ptr(),
          1.0,
          grad_weights1.as_view_mut(ctx).as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
      /*unsafe { backward.conv1_bwd_w.backward_bias(
          1.0,
          tmp_delta.as_ptr(),
          1.0,
          grad_bias1.as_view_mut(ctx).as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };*/
    }

    {
      self.batchnorm3.set_batch_size(batch_size).unwrap();
      unsafe { self.batchnorm3.backward(
          1.0, 0.0,
          tmp3_pre_act.as_ref(ctx).as_ptr(),
          out_delta.as_ptr(),
          tmp3_pre_delta.as_ref_mut(ctx).as_mut_ptr(),
          1.0, 1.0,
          self.config.bnorm_epsilon,
          self.bn_scale3.as_view(ctx).as_ptr(),
          self.bn_scale3_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_bias3_grad.as_view_mut(ctx).as_mut_ptr(),
          self.bn_cached_mean3.as_view(ctx).as_ptr(),
          self.bn_cached_ivar3.as_view(ctx).as_ptr(),
          &*ctx.get_dnn(),
      ) }.unwrap();
    }

    {
      backward.conv3_bwd_w.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv3_bwd_w.backward_filter(
          1.0,
          in_act.as_ptr(),
          tmp3_pre_delta.as_ref(ctx).as_ptr(),
          1.0,
          grad_weights3.as_view_mut(ctx).as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
      /*unsafe { backward.conv3_bwd_w.backward_bias(
          1.0,
          out_delta.as_ptr(),
          1.0,
          grad_bias3.as_view_mut(ctx).as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };*/
    }

    if let &mut Some(ref mut in_delta) = in_delta {
      //let tmp_delta = tmp_delta.as_ref(ctx);
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);

      backward.conv1_bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv1_bwd_d.backward_data(
          1.0,
          weights1.as_view(ctx).as_ptr(),
          tmp1_pre_delta.as_ref(ctx).as_ptr(),
          0.0,
          in_delta.as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };

      backward.conv3_bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv3_bwd_d.backward_data(
          1.0,
          weights3.as_view(ctx).as_ptr(),
          tmp3_pre_delta.as_ref(ctx).as_ptr(),
          1.0,
          in_delta.as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
    }
  }

  fn regularize(&mut self, reg: Regularization) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    match reg {
      Regularization::L2{l2_reg_coef} => {
        assert!(l2_reg_coef >= 0.0);
        if l2_reg_coef > 0.0 {
          backward.grad_weights1.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights1.as_view(ctx));
          backward.grad_bias1.as_view_mut(ctx)
            .row_vector_sum(l2_reg_coef, &self.bias1.as_view(ctx));
          backward.grad_weights2.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights2.as_view(ctx));
          backward.grad_bias2.as_view_mut(ctx)
            .row_vector_sum(l2_reg_coef, &self.bias2.as_view(ctx));
          backward.grad_weights3.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights3.as_view(ctx));
          backward.grad_bias3.as_view_mut(ctx)
            .row_vector_sum(l2_reg_coef, &self.bias3.as_view(ctx));
          // XXX(20160421): Do not regularize the batch norm params!
        }
      }
    }
  }

  fn accumulate_grads(&mut self, scale: f32, momentum: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.acc_grad_weights1.as_view_mut(ctx)
      .matrix_scale(momentum);
    backward.acc_grad_bias1.as_view_mut(ctx)
      .row_vector_scale(momentum);
    backward.acc_grad_weights1.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights1.as_view(ctx));
    backward.acc_grad_bias1.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.grad_bias1.as_view(ctx));

    self.acc_bn_scale1_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_scale1_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_scale1_grad.as_view(ctx));
    self.acc_bn_bias1_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_bias1_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_bias1_grad.as_view(ctx));

    backward.acc_grad_weights2.as_view_mut(ctx)
      .matrix_scale(momentum);
    backward.acc_grad_bias2.as_view_mut(ctx)
      .row_vector_scale(momentum);
    backward.acc_grad_weights2.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights2.as_view(ctx));
    backward.acc_grad_bias2.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.grad_bias2.as_view(ctx));

    self.acc_bn_scale2_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_scale2_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_scale2_grad.as_view(ctx));
    self.acc_bn_bias2_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_bias2_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_bias2_grad.as_view(ctx));

    backward.acc_grad_weights3.as_view_mut(ctx)
      .matrix_scale(momentum);
    backward.acc_grad_bias3.as_view_mut(ctx)
      .row_vector_scale(momentum);
    backward.acc_grad_weights3.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights3.as_view(ctx));
    backward.acc_grad_bias3.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.grad_bias3.as_view(ctx));

    self.acc_bn_scale3_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_scale3_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_scale3_grad.as_view(ctx));
    self.acc_bn_bias3_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_bias3_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_bias3_grad.as_view(ctx));
  }

  fn update_params(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    self.weights1.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights1.as_view(ctx));
    self.bias1.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.acc_grad_bias1.as_view(ctx));

    self.bn_scale1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale1_grad.as_view(ctx));
    self.bn_bias1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias1_grad.as_view(ctx));

    self.weights2.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights2.as_view(ctx));
    self.bias2.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.acc_grad_bias2.as_view(ctx));

    self.bn_scale2.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale2_grad.as_view(ctx));
    self.bn_bias2.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias2_grad.as_view(ctx));

    self.weights3.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights3.as_view(ctx));
    self.bias3.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.acc_grad_bias3.as_view(ctx));

    self.bn_scale3.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale3_grad.as_view(ctx));
    self.bn_bias3.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias3_grad.as_view(ctx));
  }

  fn save_params(&mut self) {
    unimplemented!();
  }

  fn restore_params(&mut self) {
    unimplemented!();
  }

  fn set_grads_with_params_diff(&mut self) {
    unimplemented!();
  }

  fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.load(self.params_off, &mut self.weights1, ctx);
    comm_worker.load(self.params_off, &mut self.bias1, ctx);
    comm_worker.load(self.params_off, &mut self.weights2, ctx);
    comm_worker.load(self.params_off, &mut self.bias2, ctx);
    comm_worker.load(self.params_off, &mut self.weights3, ctx);
    comm_worker.load(self.params_off, &mut self.bias3, ctx);
    // FIXME(20160421): batch norm params.
    unimplemented!();
  }

  fn sync_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights1, ctx);
    comm_worker.store(self.params_off, &mut self.bias1, ctx);
    comm_worker.store(self.params_off, &mut self.weights2, ctx);
    comm_worker.store(self.params_off, &mut self.bias2, ctx);
    comm_worker.store(self.params_off, &mut self.weights3, ctx);
    comm_worker.store(self.params_off, &mut self.bias3, ctx);
    // FIXME(20160421): batch norm params.
    unimplemented!();
  }

  fn reset_grads(&mut self, scale: f32) {
    unimplemented!();
  }

  fn reset(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.grad_weights1.as_view_mut(ctx)
      .matrix_scale(0.0);
    backward.grad_bias1.as_view_mut(ctx)
      .row_vector_scale(0.0);

    self.bn_scale1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
    self.bn_bias1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);

    backward.grad_weights2.as_view_mut(ctx)
      .matrix_scale(0.0);
    backward.grad_bias2.as_view_mut(ctx)
      .row_vector_scale(0.0);

    self.bn_scale2_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
    self.bn_bias2_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);

    backward.grad_weights3.as_view_mut(ctx)
      .matrix_scale(0.0);
    backward.grad_bias3.as_view_mut(ctx)
      .row_vector_scale(0.0);

    self.bn_scale3_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
    self.bn_bias3_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
  }
}

#[derive(Clone, Copy)]
pub struct BotResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}

impl BotResConv2dOperatorConfig {
  fn params_len(&self) -> usize {
    let (_, _, in_channels) = self.in_dims;
    let weights1_len = 1 * 1 * in_channels * in_channels / 4;
    let bias1_len = in_channels / 4;
    let weights2_len = 3 * 3 * in_channels / 4 * in_channels / 4;
    let bias2_len = in_channels / 4;
    let weights3_len = 1 * 1 * in_channels / 4 * in_channels;
    let bias3_len = in_channels;
    weights1_len + bias1_len +
        weights2_len + bias2_len +
        weights3_len + bias3_len
  }
}

/*
pub struct BotResConv2dOperator<Comm> {
  batch_cap:    usize,
  _capability:  OpCapability,
  params_off:   usize,
  config:       BotResConv2dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights1:     DeviceArray2d<f32>,
  bias1:        DeviceArray2d<f32>,
  weights2:     DeviceArray2d<f32>,
  bias2:        DeviceArray2d<f32>,
  weights3:     DeviceArray2d<f32>,
  bias3:        DeviceArray2d<f32>,

  workspace:    DeviceBuffer<u8>,
  conv1_fwd:    CudnnConvFwdOp,
  add_bias1:    CudnnAddOp,
  conv2_fwd:    CudnnConvFwdOp,
  add_bias2:    CudnnAddOp,
  conv3_fwd:    CudnnConvFwdOp,
  add_bias3:    CudnnAddOp,

  backward:     Option<BotResConv2dBwdOperator<Comm>>,
  //hv_backward:  Option<BotResConv2dHvBwdOperator>,
}

struct BotResConv2dBwdOperator<Comm> {
  grad_w1:      DeviceArray2d<f32>,
  grad_b1:      DeviceArray2d<f32>,
  acc_grad_w1:  DeviceArray2d<f32>,
  acc_grad_b1:  DeviceArray2d<f32>,
  conv1_bwd_w:  CudnnConvBwdFilterOp,
  conv1_bwd_d:  CudnnConvBwdDataOp,

  grad_w2:      DeviceArray2d<f32>,
  grad_b2:      DeviceArray2d<f32>,
  acc_grad_w2:  DeviceArray2d<f32>,
  acc_grad_b2:  DeviceArray2d<f32>,
  conv2_bwd_w:  CudnnConvBwdFilterOp,
  conv2_bwd_d:  CudnnConvBwdDataOp,

  grad_w3:      DeviceArray2d<f32>,
  grad_b3:      DeviceArray2d<f32>,
  acc_grad_w3:  DeviceArray2d<f32>,
  acc_grad_b3:  DeviceArray2d<f32>,
  conv3_bwd_w:  CudnnConvBwdFilterOp,
  conv3_bwd_d:  CudnnConvBwdDataOp,

  comm_worker:  Rc<RefCell<Comm>>,
}

/*struct BotResConv2dHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
  dir_bias:     DeviceArray2d<f32>,
}*/

impl<Comm> BotResConv2dOperator<Comm> where Comm: CommWorker {
  pub fn new(batch_size: usize, capability: OpCapability, params_offset: usize, config: BotResConv2dOperatorConfig, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> BotResConv2dOperator<Comm> {
    let BotResConv2dOperatorConfig{
      in_dims,
      .. } = config;
    let (in_width, in_height, in_channels) = in_dims;

    let ctx = &(*context).as_ref();

    let mut workspace_size = 0;

    /*let fwd1_algo = match config.fwd1_backend {
      BotResConv2dFwdBackend::CudnnImplicitPrecompGemm => cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
      BotResConv2dFwdBackend::CudnnFftTiling           => cudnnConvolutionFwdAlgo_t::FftTiling,
      _ => unimplemented!(),
    };*/
    let conv1_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(1, 1, in_channels, in_channels / 4).unwrap(),
        CudnnConvDesc::create_2d_symmetric(1, 0).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv1_fwd.work_size);

    let conv2_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels / 4, in_channels / 4).unwrap(),
        CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv2_fwd.work_size);

    let conv3_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(1, 1, in_channels / 4, in_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(1, 0).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv3_fwd.work_size);

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

      Some(BotResConv2dBwdOperator{
        grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        acc_grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        acc_grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        conv_bwd_w:   conv_bwd_w,
        conv_bwd_d:   conv_bwd_d,
        comm_worker:  comm_worker.unwrap(),
      })
    } else {
      None
    };

    let add_bias1 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels / 4, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
    );
    let add_bias2 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels / 4, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
    );
    let add_bias3 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
    );

    BotResConv2dOperator{
      batch_cap:    batch_size,
      _capability:  capability,
      params_off:   params_offset,
      config:       config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("BotResConv2dOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      weights1:     DeviceArray2d::<f32>::zeros((1 * 1 * in_channels, in_channels / 4), ctx),
      bias1:        DeviceArray2d::<f32>::zeros((1, in_channels / 4), ctx),
      weights2:     DeviceArray2d::<f32>::zeros((3 * 3 * in_channels / 4, in_channels / 4), ctx),
      bias2:        DeviceArray2d::<f32>::zeros((1, in_channels / 4), ctx),
      weights3:     DeviceArray2d::<f32>::zeros((1 * 1 * in_channels / 4, in_channels), ctx),
      bias3:        DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      workspace:    DeviceBuffer::<u8>::zeros(workspace_size, ctx),
      conv1_fwd:    conv1_fwd,
      add_bias1:    add_bias1,
      conv2_fwd:    conv2_fwd,
      add_bias2:    add_bias2,
      conv3_fwd:    conv3_fwd,
      add_bias3:    add_bias3,
      backward:     backward,
      //hv_backward:  None,
    }
  }
}

impl<Comm> Operator for BotResConv2dOperator<Comm> where Comm: CommWorker {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn init_params(&mut self, shared_seed: [u64; 2]) {
    let BotResConv2dOperatorConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    let mut init_weights = Array2d::zeros((conv_size * conv_size * in_channels, out_channels));
    match self.config.init_weights {
      ParamsInit::Disabled => {
        panic!("BotResConv2dOperator: params init explicitly disabled");
      }
      ParamsInit::Uniform{half_range} => {
        let dist = Range::new(-half_range as f64, half_range as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInit::Normal{std} => {
        let dist = Normal::new(0.0, std as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
    }
    let init_bias = Array2d::zeros((1, out_channels));
    self.weights.as_view_mut(ctx).sync_load(&init_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&init_bias.as_view());
  }

  fn read_params(&mut self, blob: &[u8]) -> usize {
    let BotResConv2dOperatorConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("BotResConv2dOperator failed to deserialize weights!");
    let load_bias = Array2d::deserialize(&mut reader)
      .ok().expect("BotResConv2dOperator failed to deserialize bias!");
    assert_eq!((conv_size * conv_size * in_channels, out_channels), load_weights.as_view().bound());
    assert_eq!((1, out_channels), load_bias.as_view().bound());
    self.weights.as_view_mut(ctx).sync_load(&load_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&load_bias.as_view());
    let progress = reader.position() as usize;
    progress
  }

  fn write_params(&mut self, blob: &mut Vec<u8>) {
    let ctx = &(*self.context).as_ref();
    let weights = self.weights.as_view(ctx);
    let bias = self.bias.as_view(ctx);
    let mut save_weights = Array2d::zeros(weights.bound());
    let mut save_bias = Array2d::zeros(bias.bound());
    weights.sync_store(&mut save_weights.as_view_mut());
    bias.sync_store(&mut save_bias.as_view_mut());
    save_weights.serialize(blob).unwrap();
    save_bias.serialize(blob).unwrap();
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let out_dims = self.config.get_out_dims();
    let out_length = out_dims.len();

    // FIXME(20160413): the bottleneck residual consists of 3 convolutions.

    let &mut BotResConv2dOperator{
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
    /*let BotResConv2dOperatorConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = self.config;*/
    //let (in_width, in_height, in_channels) = in_dims;
    //let in_length = in_dims.len();
    let out_dims = self.config.get_out_dims();
    //let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    // FIXME(20160413): the bottleneck residual consists of 3 convolutions;
    // the output delta is copied to the input delta before applying the conv
    // backward pass.

    let &mut BotResConv2dOperator{
      ref context,
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights, //ref mut bias,
      ref mut workspace,
      ref mut backward,
      .. } = self;
    let mut backward = backward.as_mut().unwrap();
    let &mut BotResConv2dBwdOperator{
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
        1.0,
        grad_weights.as_view_mut(ctx).as_mut_ptr(),
        workspace.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    unsafe { backward.conv_bwd_w.backward_bias(
        1.0,
        out_delta.as_ptr(),
        1.0,
        grad_bias.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    if let &mut Some(ref mut in_delta) = in_delta {
      backward.conv_bwd_d.set_batch_size(batch_size).unwrap();
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);
      unsafe { backward.conv_bwd_d.backward_data(
          1.0,
          weights.as_view(ctx).as_ptr(),
          out_delta.as_ptr(),
          0.0,
          in_delta.as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
    }
  }

  fn regularize(&mut self, reg: Regularization) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    match reg {
      Regularization::L2{l2_reg_coef} => {
        assert!(l2_reg_coef >= 0.0);
        if l2_reg_coef > 0.0 {
          backward.grad_weights.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights.as_view(ctx));
          backward.grad_bias.as_view_mut(ctx)
            .row_vector_sum(l2_reg_coef, &self.bias.as_view(ctx));
        }
      }
    }
  }

  fn accumulate_grads(&mut self, scale: f32, momentum: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.acc_grad_weights.as_view_mut(ctx)
      .matrix_scale(momentum);
    backward.acc_grad_bias.as_view_mut(ctx)
      .row_vector_scale(momentum);
    backward.acc_grad_weights.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights.as_view(ctx));
    backward.acc_grad_bias.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.grad_bias.as_view(ctx));
  }

  fn update_params(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    self.weights.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights.as_view(ctx));
    self.bias.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.acc_grad_bias.as_view(ctx));
  }

  fn save_params(&mut self) {
    unimplemented!();
  }

  fn restore_params(&mut self) {
    unimplemented!();
  }

  fn set_grads_with_params_diff(&mut self) {
    unimplemented!();
  }

  fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.load(self.params_off, &mut self.weights, ctx);
    comm_worker.load(self.params_off, &mut self.bias, ctx);
  }

  fn sync_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights, ctx);
    comm_worker.store(self.params_off, &mut self.bias, ctx);
  }

  fn reset_grads(&mut self, scale: f32) {
    unimplemented!();
  }

  fn reset(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.grad_weights.as_view_mut(ctx)
      .matrix_scale(0.0);
    backward.grad_bias.as_view_mut(ctx)
      .row_vector_scale(0.0);
  }
}
*/

#[derive(Clone, Copy)]
pub struct ProjBotResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub out_dims:     (usize, usize, usize),
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}
