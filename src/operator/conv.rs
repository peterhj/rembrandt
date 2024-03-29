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
pub enum BNormMovingAverage {
  Cumulative,
  Exponential{ema_factor: f64},
}

impl BNormMovingAverage {
  pub fn at_iter(&self, t: usize) -> f64 {
    match self {
      &BNormMovingAverage::Cumulative => {
        1.0 / (1.0 + t as f64)
      }
      &BNormMovingAverage::Exponential{ema_factor} => {
        if t == 0 {
          1.0
        } else {
          ema_factor
        }
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct BNormConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub conv_size:    usize,
  pub conv_stride:  usize,
  pub conv_pad:     usize,
  pub out_channels: usize,
  pub bnorm_mov_avg:    BNormMovingAverage,
  pub bnorm_epsilon:    f64,
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}

impl BNormConv2dOperatorConfig {
  pub fn get_out_dims(&self) -> (usize, usize, usize) {
    let (in_width, in_height, _) = self.in_dims;
    let out_width = max(0, (in_width + 2 * self.conv_pad - self.conv_size + self.conv_stride) as isize) as usize / self.conv_stride;
    let out_height = max(0, (in_height + 2 * self.conv_pad - self.conv_size + self.conv_stride) as isize) as usize / self.conv_stride;
    (out_width, out_height, self.out_channels)
  }

  pub fn params_len(&self) -> usize {
    let (_, _, in_channels) = self.in_dims;
    let weights_len = self.conv_size * self.conv_size * in_channels * self.out_channels;
    let batchnorm_len = 4 * self.out_channels;
    weights_len + batchnorm_len
  }
}

pub struct BNormConv2dOperator<Comm> {
  batch_cap:    usize,
  _capability:  OpCapability,
  params_off:   usize,
  config:       BNormConv2dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights:      DeviceArray2d<f32>,

  workspace:    DeviceBuffer<u8>,
  conv_fwd:     CudnnConvFwdOp,

  tmp_act:      DeviceBuffer<f32>,
  tmp_delta:    DeviceBuffer<f32>,

  bn_scale1:            DeviceArray2d<f32>,
  bn_scale1_grad:       DeviceArray2d<f32>,
  acc_bn_scale1_grad:   DeviceArray2d<f32>,
  save_bn_scale1:       DeviceArray2d<f32>,
  bn_bias1:             DeviceArray2d<f32>,
  bn_bias1_grad:        DeviceArray2d<f32>,
  acc_bn_bias1_grad:    DeviceArray2d<f32>,
  save_bn_bias1:        DeviceArray2d<f32>,
  bn_running_mean1:     DeviceArray2d<f32>,
  save_bn_running_mean1:    DeviceArray2d<f32>,
  bn_running_ivar1:     DeviceArray2d<f32>,
  save_bn_running_ivar1:    DeviceArray2d<f32>,
  bn_cached_mean1:      DeviceArray2d<f32>,
  bn_cached_ivar1:      DeviceArray2d<f32>,
  batchnorm1:           CudnnBatchNormOp,

  backward:     Option<BNormConv2dBwdOperator<Comm>>,
  hv_backward:  Option<BNormConv2dHvBwdOperator>,
}

struct BNormConv2dBwdOperator<Comm> {
  grad_weights: DeviceArray2d<f32>,
  acc_grad_weights: DeviceArray2d<f32>,
  save_weights: DeviceArray2d<f32>,

  conv_bwd_w:   CudnnConvBwdFilterOp,
  conv_bwd_d:   CudnnConvBwdDataOp,

  first_batch1: bool,

  comm_worker:  Rc<RefCell<Comm>>,
}

struct BNormConv2dHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
}

impl<Comm> BNormConv2dOperator<Comm> where Comm: CommWorker {
  pub fn new(batch_size: usize, capability: OpCapability, params_offset: usize, config: BNormConv2dOperatorConfig, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> BNormConv2dOperator<Comm> {
    let BNormConv2dOperatorConfig{
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

      Some(BNormConv2dBwdOperator{
        grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        acc_grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        save_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        conv_bwd_w:   conv_bwd_w,
        conv_bwd_d:   conv_bwd_d,
        first_batch1: true,
        comm_worker:  comm_worker.unwrap(),
      })
    } else {
      None
    };

    let batchnorm1 = CudnnBatchNormOp::new(
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        cudnnBatchNormMode_t::Spatial,
    );

    BNormConv2dOperator{
      batch_cap:    batch_size,
      _capability:  capability,
      params_off:   params_offset,
      config:       config,
      context:      context.clone(),

      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("BNormConv2dOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),

      weights:      DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),

      workspace:    DeviceBuffer::<u8>::zeros(workspace_size, ctx),
      conv_fwd:     conv_fwd,

      tmp_act:      DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp_delta:    DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),

      bn_scale1:            DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_scale1_grad:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_scale1_grad:   DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_scale1:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias1:             DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias1_grad:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_bias1_grad:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_bias1:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_mean1:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_running_mean1:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_ivar1:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_running_ivar1:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_mean1:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_ivar1:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      batchnorm1:           batchnorm1,

      backward:     backward,
      hv_backward:  None,
    }
  }
}

impl<Comm> Operator for BNormConv2dOperator<Comm> where Comm: CommWorker {
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
    let BNormConv2dOperatorConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    let mut init_weights = Array2d::zeros((conv_size * conv_size * in_channels, out_channels));
    match self.config.init_weights {
      ParamsInit::Disabled => {
        panic!("BNormConv2dOperator: params init explicitly disabled");
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
    self.weights.as_view_mut(ctx).sync_load(&init_weights.as_view());

    //self.bn_scale1.as_view_mut(ctx).set_constant(1.0);
    //self.bn_running_ivar1.as_view_mut(ctx).set_constant(1.0);

    self.bn_scale1.as_view_mut(ctx).set_constant(1.0);
    self.bn_bias1.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_mean1.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_ivar1.as_view_mut(ctx).set_constant(1.0);

    /*self.bn_scale2.as_view_mut(ctx).set_constant(1.0);
    self.bn_bias2.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_mean2.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_ivar2.as_view_mut(ctx).set_constant(1.0);

    self.bn_scale3.as_view_mut(ctx).set_constant(1.0);
    self.bn_bias3.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_mean3.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_ivar3.as_view_mut(ctx).set_constant(1.0);*/
  }

  fn decode_params(&mut self, blob: &[u8]) -> usize {
    /*let BNormConv2dOperatorConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("BNormConv2dOperator failed to deserialize weights!");
    assert_eq!((conv_size * conv_size * in_channels, out_channels), load_weights.as_view().bound());
    self.weights.as_view_mut(ctx).sync_load(&load_weights.as_view());
    // FIXME(20160422): batch norm params.
    unimplemented!();*/

    let ctx = &(*self.context).as_ref();
    let mut reader = Cursor::new(blob);

    let load_weights = Array2d::deserialize(&mut reader).unwrap();
    self.weights.as_view_mut(ctx).sync_load(&load_weights.as_view());

    let load_bn_scale1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_scale1.as_view_mut(ctx).sync_load(&load_bn_scale1.as_view());
    let load_bn_bias1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_bias1.as_view_mut(ctx).sync_load(&load_bn_bias1.as_view());
    let load_bn_running_mean1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_mean1.as_view_mut(ctx).sync_load(&load_bn_running_mean1.as_view());
    let load_bn_running_ivar1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_ivar1.as_view_mut(ctx).sync_load(&load_bn_running_ivar1.as_view());

    let progress = reader.position() as usize;
    progress
  }

  fn encode_params(&mut self, blob: &mut Vec<u8>) {
    let ctx = &(*self.context).as_ref();

    let weights = self.weights.as_view(ctx);
    let mut save_weights = Array2d::zeros(weights.bound());
    weights.sync_store(&mut save_weights.as_view_mut());
    save_weights.serialize(blob).unwrap();

    let bn_scale1 = self.bn_scale1.as_view(ctx);
    let mut save_bn_scale1 = Array2d::zeros(bn_scale1.bound());
    bn_scale1.sync_store(&mut save_bn_scale1.as_view_mut());
    save_bn_scale1.serialize(blob).unwrap();

    let bn_bias1 = self.bn_bias1.as_view(ctx);
    let mut save_bn_bias1 = Array2d::zeros(bn_bias1.bound());
    bn_bias1.sync_store(&mut save_bn_bias1.as_view_mut());
    save_bn_bias1.serialize(blob).unwrap();

    let bn_running_mean1 = self.bn_running_mean1.as_view(ctx);
    let mut save_bn_running_mean1 = Array2d::zeros(bn_running_mean1.bound());
    bn_running_mean1.sync_store(&mut save_bn_running_mean1.as_view_mut());
    save_bn_running_mean1.serialize(blob).unwrap();

    let bn_running_ivar1 = self.bn_running_ivar1.as_view(ctx);
    let mut save_bn_running_ivar1 = Array2d::zeros(bn_running_ivar1.bound());
    bn_running_ivar1.sync_store(&mut save_bn_running_ivar1.as_view_mut());
    save_bn_running_ivar1.serialize(blob).unwrap();
  }

  fn decode_state(&mut self, blob: &[u8]) -> usize {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    let mut reader = Cursor::new(blob);

    let load_weights1 = Array2d::deserialize(&mut reader).unwrap();
    backward.acc_grad_weights.as_view_mut(ctx).sync_load(&load_weights1.as_view());

    let load_bn_scale1 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_scale1_grad.as_view_mut(ctx).sync_load(&load_bn_scale1.as_view());
    let load_bn_bias1 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_bias1_grad.as_view_mut(ctx).sync_load(&load_bn_bias1.as_view());
    let bn_cached_mean1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_mean1.as_view_mut(ctx).sync_load(&bn_cached_mean1.as_view());
    let bn_cached_ivar1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_ivar1.as_view_mut(ctx).sync_load(&bn_cached_ivar1.as_view());

    let progress = reader.position() as usize;
    progress
  }

  fn encode_state(&mut self, blob: &mut Vec<u8>) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    let weights1 = backward.acc_grad_weights.as_view(ctx);
    let mut save_weights1 = Array2d::zeros(weights1.bound());
    weights1.sync_store(&mut save_weights1.as_view_mut());
    save_weights1.serialize(blob).unwrap();

    let bn_scale1 = self.acc_bn_scale1_grad.as_view(ctx);
    let mut save_bn_scale1 = Array2d::zeros(bn_scale1.bound());
    bn_scale1.sync_store(&mut save_bn_scale1.as_view_mut());
    save_bn_scale1.serialize(blob).unwrap();

    let bn_bias1 = self.acc_bn_bias1_grad.as_view(ctx);
    let mut save_bn_bias1 = Array2d::zeros(bn_bias1.bound());
    bn_bias1.sync_store(&mut save_bn_bias1.as_view_mut());
    save_bn_bias1.serialize(blob).unwrap();

    let bn_running_mean1 = self.bn_cached_mean1.as_view(ctx);
    let mut save_bn_running_mean1 = Array2d::zeros(bn_running_mean1.bound());
    bn_running_mean1.sync_store(&mut save_bn_running_mean1.as_view_mut());
    save_bn_running_mean1.serialize(blob).unwrap();

    let bn_running_ivar1 = self.bn_cached_ivar1.as_view(ctx);
    let mut save_bn_running_ivar1 = Array2d::zeros(bn_running_ivar1.bound());
    bn_running_ivar1.sync_store(&mut save_bn_running_ivar1.as_view_mut());
    save_bn_running_ivar1.serialize(blob).unwrap();
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let out_dims = self.config.get_out_dims();
    let out_length = out_dims.len();

    let &mut BNormConv2dOperator{
      ref context,
      ref mut in_act, ref mut out_act,
      ref mut weights,
      ref mut workspace,
      ref mut tmp_act,
      .. } = self;

    let ctx = &(**context).as_ref();
    let mut out_act = out_act.borrow_mut().as_ref_mut(ctx);

    self.conv_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv_fwd.forward(
        1.0,
        in_act.borrow_mut().as_ref(ctx).as_ptr(),
        weights.as_view(ctx).as_ptr(),
        0.0,
        tmp_act.as_ref_mut(ctx).as_mut_ptr(),
        workspace.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("conv2d forward failed: {:?}", e); }
    }

    match phase {
      OpPhase::Inference => {
        self.batchnorm1.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm1.forward_inference(
            1.0,
            tmp_act.as_ref(ctx).as_ptr(),
            0.0,
            out_act.as_mut_ptr(),
            self.config.bnorm_epsilon,
            self.bn_scale1.as_view(ctx).as_ptr(),
            self.bn_bias1.as_view(ctx).as_ptr(),
            self.bn_running_mean1.as_view(ctx).as_ptr(),
            self.bn_running_ivar1.as_view(ctx).as_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
      OpPhase::Training{t} => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        /*let ema_factor = match backward.first_batch1 {
          true  => {
            backward.first_batch1 = false;
            1.0
          }
          false => 0.01,
        };*/
        let ema_factor = self.config.bnorm_mov_avg.at_iter(t);
        self.batchnorm1.set_batch_size(batch_size).unwrap();
        unsafe { self.batchnorm1.forward_training(
            1.0,
            tmp_act.as_ref(ctx).as_ptr(),
            0.0,
            out_act.as_mut_ptr(),
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

    let &mut BNormConv2dOperator{
      ref context,
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights,
      ref mut workspace,
      ref mut backward,
      ref mut tmp_act, ref mut tmp_delta,
      .. } = self;
    let mut backward = backward.as_mut().unwrap();
    let &mut BNormConv2dBwdOperator{
      ref mut grad_weights,
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

    {
      self.batchnorm1.set_batch_size(batch_size).unwrap();
      unsafe { self.batchnorm1.backward(
          1.0, 0.0,
          tmp_act.as_ref(ctx).as_ptr(),
          out_delta.as_ptr(),
          tmp_delta.as_ref_mut(ctx).as_mut_ptr(),
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

    backward.conv_bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { backward.conv_bwd_w.backward_filter(
        1.0,
        in_act.as_ptr(),
        tmp_delta.as_ref(ctx).as_ptr(),
        1.0,
        grad_weights.as_view_mut(ctx).as_mut_ptr(),
        workspace.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };

    if let &mut Some(ref mut in_delta) = in_delta {
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);
      backward.conv_bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { backward.conv_bwd_d.backward_data(
          1.0,
          weights.as_view(ctx).as_ptr(),
          tmp_delta.as_ref(ctx).as_ptr(),
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
          // XXX(20160421): Batch normalization params are not regularized.
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
    backward.acc_grad_weights.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights.as_view(ctx));

    self.acc_bn_scale1_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_scale1_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_scale1_grad.as_view(ctx));
    self.acc_bn_bias1_grad.as_view_mut(ctx)
      .row_vector_scale(momentum);
    self.acc_bn_bias1_grad.as_view_mut(ctx)
      .row_vector_sum(scale, &self.bn_bias1_grad.as_view(ctx));
  }

  fn update_params(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    self.weights.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights.as_view(ctx));

    self.bn_scale1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale1_grad.as_view(ctx));
    self.bn_bias1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias1_grad.as_view(ctx));
  }

  fn update_params2(&mut self, grad_scale: f32, update_scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    if grad_scale != 0.0 {
      self.weights.as_view_mut(ctx)
        .matrix_sum(grad_scale, &backward.grad_weights.as_view(ctx));

      self.bn_scale1.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_scale1_grad.as_view(ctx));
      self.bn_bias1.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_bias1_grad.as_view(ctx));
    }

    if update_scale != 0.0 {
      self.weights.as_view_mut(ctx)
        .matrix_sum(update_scale, &backward.acc_grad_weights.as_view(ctx));

      self.bn_scale1.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_scale1_grad.as_view(ctx));
      self.bn_bias1.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_bias1_grad.as_view(ctx));
    }
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

    self.bn_scale1.as_view(ctx)
      .send(&mut self.save_bn_scale1.as_view_mut(ctx));
    self.bn_bias1.as_view(ctx)
      .send(&mut self.save_bn_bias1.as_view_mut(ctx));
    self.bn_running_mean1.as_view(ctx)
      .send(&mut self.save_bn_running_mean1.as_view_mut(ctx));
    self.bn_running_ivar1.as_view(ctx)
      .send(&mut self.save_bn_running_ivar1.as_view_mut(ctx));
  }

  fn restore_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.save_weights.as_view(ctx)
      .send(&mut self.weights.as_view_mut(ctx));

    self.save_bn_scale1.as_view(ctx)
      .send(&mut self.bn_scale1.as_view_mut(ctx));
    self.save_bn_bias1.as_view(ctx)
      .send(&mut self.bn_bias1.as_view_mut(ctx));
    self.save_bn_running_mean1.as_view(ctx)
      .send(&mut self.bn_running_mean1.as_view_mut(ctx));
    self.save_bn_running_ivar1.as_view(ctx)
      .send(&mut self.bn_running_ivar1.as_view_mut(ctx));
  }

  fn set_grads_with_params_diff(&mut self) {
    /*assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    self.weights.as_view(ctx)
      .send(&mut backward.acc_grad_weights.as_view_mut(ctx));
    self.bias.as_view(ctx)
      .send(&mut backward.acc_grad_bias.as_view_mut(ctx));
    backward.acc_grad_weights.as_view_mut(ctx)
      .matrix_sum(-1.0, &backward.save_weights.as_view(ctx));
    backward.acc_grad_bias.as_view_mut(ctx)
      .row_vector_sum(-1.0, &backward.save_bias.as_view(ctx));*/
    unimplemented!();
  }

  /*fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    /*assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.load(self.params_off, &mut self.weights); //, ctx);*/
    unimplemented!();
  }

  fn sync_params(&mut self) {
    /*assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights); //, ctx);*/
    unimplemented!();
  }*/

  fn stage_grads_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    assert!(self.backward.is_some());
    let mut backward = self.backward.as_mut().unwrap();

    let mut offset = offset;

    comm_worker.load(offset, &mut backward.grad_weights);
    offset += backward.grad_weights.len();

    comm_worker.load(offset, &mut self.bn_scale1_grad);
    offset += self.bn_scale1_grad.len();
    comm_worker.load(offset, &mut self.bn_bias1_grad);
    offset += self.bn_bias1_grad.len();
    offset += self.bn_running_mean1.len();
    offset += self.bn_running_ivar1.len();

    self.config.params_len()
  }

  fn merge_grads_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    assert!(self.backward.is_some());
    let mut backward = self.backward.as_mut().unwrap();

    let mut offset = offset;

    comm_worker.store(offset, &mut backward.grad_weights);
    offset += backward.grad_weights.len();

    comm_worker.store(offset, &mut self.bn_scale1_grad);
    offset += self.bn_scale1_grad.len();
    comm_worker.store(offset, &mut self.bn_bias1_grad);
    offset += self.bn_bias1_grad.len();
    offset += self.bn_running_mean1.len();
    offset += self.bn_running_ivar1.len();

    self.config.params_len()
  }

  fn stage_params_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    let mut offset = offset;

    comm_worker.load(offset, &mut self.weights);
    offset += self.weights.len();

    comm_worker.load(offset, &mut self.bn_scale1);
    offset += self.bn_scale1.len();
    comm_worker.load(offset, &mut self.bn_bias1);
    offset += self.bn_bias1.len();
    comm_worker.load(offset, &mut self.bn_running_mean1);
    offset += self.bn_running_mean1.len();
    comm_worker.load(offset, &mut self.bn_running_ivar1);
    offset += self.bn_running_ivar1.len();

    self.config.params_len()
  }

  fn merge_params_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    let mut offset = offset;

    comm_worker.store(offset, &mut self.weights);
    offset += self.weights.len();

    comm_worker.store(offset, &mut self.bn_scale1);
    offset += self.bn_scale1.len();
    comm_worker.store(offset, &mut self.bn_bias1);
    offset += self.bn_bias1.len();
    comm_worker.store(offset, &mut self.bn_running_mean1);
    offset += self.bn_running_mean1.len();
    comm_worker.store(offset, &mut self.bn_running_ivar1);
    offset += self.bn_running_ivar1.len();

    self.config.params_len()
  }

  /*fn reset_grads(&mut self, scale: f32) {
    /*assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.grad_weights.as_view_mut(ctx)
      .matrix_scale(scale);
    backward.grad_bias.as_view_mut(ctx)
      .row_vector_scale(scale);*/
    unimplemented!();
  }*/

  fn reset(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.grad_weights.as_view_mut(ctx)
      .matrix_scale(0.0);

    self.bn_scale1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
    self.bn_bias1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
  }
}

#[derive(Clone, Copy, Debug)]
pub struct StackResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  //pub out_dims:     (usize, usize, usize),
  pub bnorm_mov_avg:    BNormMovingAverage,
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
    let batchnorm1_len = 4 * in_channels;
    let weights2_len = 3 * 3 * in_channels * in_channels;
    let batchnorm2_len = 4 * in_channels;
    weights1_len + batchnorm1_len + weights2_len + batchnorm2_len
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
  weights2:     DeviceArray2d<f32>,

  tmp1_pre_act:     DeviceBuffer<f32>,
  tmp1_pre_delta:   DeviceBuffer<f32>,
  tmp1_post_act:    DeviceBuffer<f32>,
  tmp1_post_delta:  DeviceBuffer<f32>,
  tmp2_pre_act:     DeviceBuffer<f32>,
  tmp2_pre_delta:   DeviceBuffer<f32>,

  bn_scale1:        DeviceArray2d<f32>,
  bn_scale1_grad:   DeviceArray2d<f32>,
  acc_bn_scale1_grad:   DeviceArray2d<f32>,
  save_bn_scale1:       DeviceArray2d<f32>,
  bn_bias1:         DeviceArray2d<f32>,
  bn_bias1_grad:    DeviceArray2d<f32>,
  acc_bn_bias1_grad:    DeviceArray2d<f32>,
  save_bn_bias1:        DeviceArray2d<f32>,
  bn_running_mean1: DeviceArray2d<f32>,
  save_bn_running_mean1:    DeviceArray2d<f32>,
  bn_running_ivar1: DeviceArray2d<f32>,
  save_bn_running_ivar1:    DeviceArray2d<f32>,
  bn_cached_mean1:  DeviceArray2d<f32>,
  bn_cached_ivar1:  DeviceArray2d<f32>,
  batchnorm1:   CudnnBatchNormOp,

  bn_scale2:        DeviceArray2d<f32>,
  bn_scale2_grad:   DeviceArray2d<f32>,
  acc_bn_scale2_grad:   DeviceArray2d<f32>,
  save_bn_scale2:       DeviceArray2d<f32>,
  bn_bias2:         DeviceArray2d<f32>,
  bn_bias2_grad:    DeviceArray2d<f32>,
  acc_bn_bias2_grad:    DeviceArray2d<f32>,
  save_bn_bias2:        DeviceArray2d<f32>,
  bn_running_mean2: DeviceArray2d<f32>,
  save_bn_running_mean2:    DeviceArray2d<f32>,
  bn_running_ivar2: DeviceArray2d<f32>,
  save_bn_running_ivar2:    DeviceArray2d<f32>,
  bn_cached_mean2:  DeviceArray2d<f32>,
  bn_cached_ivar2:  DeviceArray2d<f32>,
  batchnorm2:   CudnnBatchNormOp,

  workspace:    DeviceBuffer<u8>,
  conv1_fwd:    CudnnConvFwdOp,
  conv2_fwd:    CudnnConvFwdOp,

  backward:     Option<StackResConv2dBwdOperator<Comm>>,
  //hv_backward:  Option<BotResConv2dHvBwdOperator>,
}

struct StackResConv2dBwdOperator<Comm> {
  grad_weights1:      DeviceArray2d<f32>,
  acc_grad_weights1:  DeviceArray2d<f32>,
  save_weights1:    DeviceArray2d<f32>,
  conv1_bwd_w:  CudnnConvBwdFilterOp,
  conv1_bwd_d:  CudnnConvBwdDataOp,

  first_batch1: bool,

  grad_weights2:      DeviceArray2d<f32>,
  acc_grad_weights2:  DeviceArray2d<f32>,
  save_weights2:    DeviceArray2d<f32>,
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
        acc_grad_weights1:  DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),
        save_weights1:      DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),

        conv1_bwd_w:  conv1_bwd_w,
        conv1_bwd_d:  conv1_bwd_d,

        first_batch1: true,

        grad_weights2:      DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),
        acc_grad_weights2:  DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),
        save_weights2:      DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),

        conv2_bwd_w:  conv2_bwd_w,
        conv2_bwd_d:  conv2_bwd_d,

        first_batch2: true,

        comm_worker:  comm_worker.unwrap(),
      })
    } else {
      None
    };

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
      weights2:     DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, in_channels), ctx),

      tmp1_pre_act:     DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp1_pre_delta:   DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp1_post_act:    DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp1_post_delta:  DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp2_pre_act:     DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),
      tmp2_pre_delta:   DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx),

      bn_scale1:            bn_scale1,
      bn_scale1_grad:       DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      acc_bn_scale1_grad:   DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      save_bn_scale1:       DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_bias1:             DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_bias1_grad:        DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      acc_bn_bias1_grad:    DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      save_bn_bias1:        DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_running_mean1: DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      save_bn_running_mean1:    DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_running_ivar1: DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      save_bn_running_ivar1:    DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_cached_mean1:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_cached_ivar1:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      batchnorm1: batchnorm1,

      bn_scale2:            bn_scale2,
      bn_scale2_grad:       DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      acc_bn_scale2_grad:   DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      save_bn_scale2:       DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_bias2:             DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_bias2_grad:        DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      acc_bn_bias2_grad:    DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      save_bn_bias2:        DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_running_mean2: DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      save_bn_running_mean2:    DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_running_ivar2: DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      save_bn_running_ivar2:    DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_cached_mean2:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      bn_cached_ivar2:  DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      batchnorm2: batchnorm2,

      workspace:    DeviceBuffer::<u8>::zeros(workspace_size, ctx),
      conv1_fwd:    conv1_fwd,
      conv2_fwd:    conv2_fwd,

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
    self.weights1.as_view_mut(ctx).sync_load(&init_weights1.as_view());
    self.weights2.as_view_mut(ctx).sync_load(&init_weights2.as_view());

    /*self.bn_scale1.as_view_mut(ctx).set_constant(1.0);
    self.bn_running_ivar1.as_view_mut(ctx).set_constant(1.0);
    self.bn_scale2.as_view_mut(ctx).set_constant(1.0);
    self.bn_running_ivar2.as_view_mut(ctx).set_constant(1.0);*/

    self.bn_scale1.as_view_mut(ctx).set_constant(1.0);
    self.bn_bias1.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_mean1.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_ivar1.as_view_mut(ctx).set_constant(1.0);

    self.bn_scale2.as_view_mut(ctx).set_constant(1.0);
    self.bn_bias2.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_mean2.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_ivar2.as_view_mut(ctx).set_constant(1.0);

    /*self.bn_scale3.as_view_mut(ctx).set_constant(1.0);
    self.bn_bias3.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_mean3.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_ivar3.as_view_mut(ctx).set_constant(1.0);*/
  }

  fn decode_params(&mut self, blob: &[u8]) -> usize {
    /*let StackResConv2dOperatorConfig{in_dims, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut reader = Cursor::new(blob);
    let load_weights1 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize weights!");
    let load_weights2 = Array2d::deserialize(&mut reader)
      .ok().expect("StackResConv2dOperator failed to deserialize weights!");
    assert_eq!((3 * 3 * in_channels, in_channels), load_weights1.as_view().bound());
    assert_eq!((3 * 3 * in_channels, in_channels), load_weights2.as_view().bound());
    self.weights1.as_view_mut(ctx).sync_load(&load_weights1.as_view());
    self.weights2.as_view_mut(ctx).sync_load(&load_weights2.as_view());
    // FIXME(20160422): batch norm params.
    unimplemented!();*/

    let ctx = &(*self.context).as_ref();
    let mut reader = Cursor::new(blob);

    let load_weights1 = Array2d::deserialize(&mut reader).unwrap();
    self.weights1.as_view_mut(ctx).sync_load(&load_weights1.as_view());

    let load_bn_scale1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_scale1.as_view_mut(ctx).sync_load(&load_bn_scale1.as_view());
    let load_bn_bias1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_bias1.as_view_mut(ctx).sync_load(&load_bn_bias1.as_view());
    let load_bn_running_mean1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_mean1.as_view_mut(ctx).sync_load(&load_bn_running_mean1.as_view());
    let load_bn_running_ivar1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_ivar1.as_view_mut(ctx).sync_load(&load_bn_running_ivar1.as_view());

    let load_weights2 = Array2d::deserialize(&mut reader).unwrap();
    self.weights2.as_view_mut(ctx).sync_load(&load_weights2.as_view());

    let load_bn_scale2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_scale2.as_view_mut(ctx).sync_load(&load_bn_scale2.as_view());
    let load_bn_bias2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_bias2.as_view_mut(ctx).sync_load(&load_bn_bias2.as_view());
    let load_bn_running_mean2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_mean2.as_view_mut(ctx).sync_load(&load_bn_running_mean2.as_view());
    let load_bn_running_ivar2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_ivar2.as_view_mut(ctx).sync_load(&load_bn_running_ivar2.as_view());

    let progress = reader.position() as usize;
    progress
  }

  fn encode_params(&mut self, blob: &mut Vec<u8>) {
    let ctx = &(*self.context).as_ref();

    let weights1 = self.weights1.as_view(ctx);
    let mut save_weights1 = Array2d::zeros(weights1.bound());
    weights1.sync_store(&mut save_weights1.as_view_mut());
    save_weights1.serialize(blob).unwrap();

    let bn_scale1 = self.bn_scale1.as_view(ctx);
    let mut save_bn_scale1 = Array2d::zeros(bn_scale1.bound());
    bn_scale1.sync_store(&mut save_bn_scale1.as_view_mut());
    save_bn_scale1.serialize(blob).unwrap();

    let bn_bias1 = self.bn_bias1.as_view(ctx);
    let mut save_bn_bias1 = Array2d::zeros(bn_bias1.bound());
    bn_bias1.sync_store(&mut save_bn_bias1.as_view_mut());
    save_bn_bias1.serialize(blob).unwrap();

    let bn_running_mean1 = self.bn_running_mean1.as_view(ctx);
    let mut save_bn_running_mean1 = Array2d::zeros(bn_running_mean1.bound());
    bn_running_mean1.sync_store(&mut save_bn_running_mean1.as_view_mut());
    save_bn_running_mean1.serialize(blob).unwrap();

    let bn_running_ivar1 = self.bn_running_ivar1.as_view(ctx);
    let mut save_bn_running_ivar1 = Array2d::zeros(bn_running_ivar1.bound());
    bn_running_ivar1.sync_store(&mut save_bn_running_ivar1.as_view_mut());
    save_bn_running_ivar1.serialize(blob).unwrap();

    let weights2 = self.weights2.as_view(ctx);
    let mut save_weights2 = Array2d::zeros(weights2.bound());
    weights2.sync_store(&mut save_weights2.as_view_mut());
    save_weights2.serialize(blob).unwrap();

    let bn_scale2 = self.bn_scale2.as_view(ctx);
    let mut save_bn_scale2 = Array2d::zeros(bn_scale2.bound());
    bn_scale2.sync_store(&mut save_bn_scale2.as_view_mut());
    save_bn_scale2.serialize(blob).unwrap();

    let bn_bias2 = self.bn_bias2.as_view(ctx);
    let mut save_bn_bias2 = Array2d::zeros(bn_bias2.bound());
    bn_bias2.sync_store(&mut save_bn_bias2.as_view_mut());
    save_bn_bias2.serialize(blob).unwrap();

    let bn_running_mean2 = self.bn_running_mean2.as_view(ctx);
    let mut save_bn_running_mean2 = Array2d::zeros(bn_running_mean2.bound());
    bn_running_mean2.sync_store(&mut save_bn_running_mean2.as_view_mut());
    save_bn_running_mean2.serialize(blob).unwrap();

    let bn_running_ivar2 = self.bn_running_ivar2.as_view(ctx);
    let mut save_bn_running_ivar2 = Array2d::zeros(bn_running_ivar2.bound());
    bn_running_ivar2.sync_store(&mut save_bn_running_ivar2.as_view_mut());
    save_bn_running_ivar2.serialize(blob).unwrap();
  }

  fn decode_state(&mut self, blob: &[u8]) -> usize {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    let mut reader = Cursor::new(blob);

    let load_weights1 = Array2d::deserialize(&mut reader).unwrap();
    backward.acc_grad_weights1.as_view_mut(ctx).sync_load(&load_weights1.as_view());

    let load_bn_scale1 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_scale1_grad.as_view_mut(ctx).sync_load(&load_bn_scale1.as_view());
    let load_bn_bias1 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_bias1_grad.as_view_mut(ctx).sync_load(&load_bn_bias1.as_view());
    let bn_cached_mean1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_mean1.as_view_mut(ctx).sync_load(&bn_cached_mean1.as_view());
    let bn_cached_ivar1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_ivar1.as_view_mut(ctx).sync_load(&bn_cached_ivar1.as_view());

    let load_weights2 = Array2d::deserialize(&mut reader).unwrap();
    backward.acc_grad_weights2.as_view_mut(ctx).sync_load(&load_weights2.as_view());

    let load_bn_scale2 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_scale2_grad.as_view_mut(ctx).sync_load(&load_bn_scale2.as_view());
    let load_bn_bias2 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_bias2_grad.as_view_mut(ctx).sync_load(&load_bn_bias2.as_view());
    let bn_cached_mean2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_mean2.as_view_mut(ctx).sync_load(&bn_cached_mean2.as_view());
    let bn_cached_ivar2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_ivar2.as_view_mut(ctx).sync_load(&bn_cached_ivar2.as_view());

    let progress = reader.position() as usize;
    progress
  }

  fn encode_state(&mut self, blob: &mut Vec<u8>) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    let weights1 = backward.acc_grad_weights1.as_view(ctx);
    let mut save_weights1 = Array2d::zeros(weights1.bound());
    weights1.sync_store(&mut save_weights1.as_view_mut());
    save_weights1.serialize(blob).unwrap();

    let bn_scale1 = self.acc_bn_scale1_grad.as_view(ctx);
    let mut save_bn_scale1 = Array2d::zeros(bn_scale1.bound());
    bn_scale1.sync_store(&mut save_bn_scale1.as_view_mut());
    save_bn_scale1.serialize(blob).unwrap();

    let bn_bias1 = self.acc_bn_bias1_grad.as_view(ctx);
    let mut save_bn_bias1 = Array2d::zeros(bn_bias1.bound());
    bn_bias1.sync_store(&mut save_bn_bias1.as_view_mut());
    save_bn_bias1.serialize(blob).unwrap();

    let bn_running_mean1 = self.bn_cached_mean1.as_view(ctx);
    let mut save_bn_running_mean1 = Array2d::zeros(bn_running_mean1.bound());
    bn_running_mean1.sync_store(&mut save_bn_running_mean1.as_view_mut());
    save_bn_running_mean1.serialize(blob).unwrap();

    let bn_running_ivar1 = self.bn_cached_ivar1.as_view(ctx);
    let mut save_bn_running_ivar1 = Array2d::zeros(bn_running_ivar1.bound());
    bn_running_ivar1.sync_store(&mut save_bn_running_ivar1.as_view_mut());
    save_bn_running_ivar1.serialize(blob).unwrap();

    let weights2 = backward.acc_grad_weights2.as_view(ctx);
    let mut save_weights2 = Array2d::zeros(weights2.bound());
    weights2.sync_store(&mut save_weights2.as_view_mut());
    save_weights2.serialize(blob).unwrap();

    let bn_scale2 = self.acc_bn_scale2_grad.as_view(ctx);
    let mut save_bn_scale2 = Array2d::zeros(bn_scale2.bound());
    bn_scale2.sync_store(&mut save_bn_scale2.as_view_mut());
    save_bn_scale2.serialize(blob).unwrap();

    let bn_bias2 = self.acc_bn_bias2_grad.as_view(ctx);
    let mut save_bn_bias2 = Array2d::zeros(bn_bias2.bound());
    bn_bias2.sync_store(&mut save_bn_bias2.as_view_mut());
    save_bn_bias2.serialize(blob).unwrap();

    let bn_running_mean2 = self.bn_cached_mean2.as_view(ctx);
    let mut save_bn_running_mean2 = Array2d::zeros(bn_running_mean2.bound());
    bn_running_mean2.sync_store(&mut save_bn_running_mean2.as_view_mut());
    save_bn_running_mean2.serialize(blob).unwrap();

    let bn_running_ivar2 = self.bn_cached_ivar2.as_view(ctx);
    let mut save_bn_running_ivar2 = Array2d::zeros(bn_running_ivar2.bound());
    bn_running_ivar2.sync_store(&mut save_bn_running_ivar2.as_view_mut());
    save_bn_running_ivar2.serialize(blob).unwrap();
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
      ref mut weights1,
      ref mut weights2,
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
      OpPhase::Training{t} => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        /*let ema_factor = match backward.first_batch1 {
          true  => {
            backward.first_batch1 = false;
            1.0
          }
          false => 0.01,
        };*/
        let ema_factor = self.config.bnorm_mov_avg.at_iter(t);
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
      OpPhase::Training{t} => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        /*let ema_factor = match backward.first_batch2 {
          true  => {
            backward.first_batch2 = false;
            1.0
          }
          false => 0.01,
        };*/
        let ema_factor = self.config.bnorm_mov_avg.at_iter(t);
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
      ref mut weights1,
      ref mut weights2,
      ref mut tmp1_pre_act, ref mut tmp1_pre_delta,
      ref mut tmp1_post_act, ref mut tmp1_post_delta,
      ref mut tmp2_pre_act, ref mut tmp2_pre_delta,
      ref mut workspace,
      ref mut backward,
      .. } = self;
    let mut backward = backward.as_mut().unwrap();
    let &mut StackResConv2dBwdOperator{
      ref mut grad_weights1,
      ref mut grad_weights2,
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
          backward.grad_weights2.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights2.as_view(ctx));
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
    backward.acc_grad_weights1.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights1.as_view(ctx));

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
    backward.acc_grad_weights2.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights2.as_view(ctx));

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

    self.bn_scale1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale1_grad.as_view(ctx));
    self.bn_bias1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias1_grad.as_view(ctx));

    self.weights2.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights2.as_view(ctx));

    self.bn_scale2.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale2_grad.as_view(ctx));
    self.bn_bias2.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias2_grad.as_view(ctx));
  }

  fn update_params2(&mut self, grad_scale: f32, update_scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    if grad_scale != 0.0 {
      self.weights1.as_view_mut(ctx)
        .matrix_sum(grad_scale, &backward.grad_weights1.as_view(ctx));

      self.bn_scale1.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_scale1_grad.as_view(ctx));
      self.bn_bias1.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_bias1_grad.as_view(ctx));

      self.weights2.as_view_mut(ctx)
        .matrix_sum(grad_scale, &backward.grad_weights2.as_view(ctx));

      self.bn_scale2.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_scale2_grad.as_view(ctx));
      self.bn_bias2.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_bias2_grad.as_view(ctx));
    }

    if update_scale != 0.0 {
      self.weights1.as_view_mut(ctx)
        .matrix_sum(update_scale, &backward.acc_grad_weights1.as_view(ctx));

      self.bn_scale1.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_scale1_grad.as_view(ctx));
      self.bn_bias1.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_bias1_grad.as_view(ctx));

      self.weights2.as_view_mut(ctx)
        .matrix_sum(update_scale, &backward.acc_grad_weights2.as_view(ctx));

      self.bn_scale2.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_scale2_grad.as_view(ctx));
      self.bn_bias2.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_bias2_grad.as_view(ctx));
    }
  }

  fn save_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    self.weights1.as_view(ctx)
      .send(&mut backward.save_weights1.as_view_mut(ctx));

    self.bn_scale1.as_view(ctx)
      .send(&mut self.save_bn_scale1.as_view_mut(ctx));
    self.bn_bias1.as_view(ctx)
      .send(&mut self.save_bn_bias1.as_view_mut(ctx));
    self.bn_running_mean1.as_view(ctx)
      .send(&mut self.save_bn_running_mean1.as_view_mut(ctx));
    self.bn_running_ivar1.as_view(ctx)
      .send(&mut self.save_bn_running_ivar1.as_view_mut(ctx));

    self.weights2.as_view(ctx)
      .send(&mut backward.save_weights2.as_view_mut(ctx));

    self.bn_scale2.as_view(ctx)
      .send(&mut self.save_bn_scale2.as_view_mut(ctx));
    self.bn_bias2.as_view(ctx)
      .send(&mut self.save_bn_bias2.as_view_mut(ctx));
    self.bn_running_mean2.as_view(ctx)
      .send(&mut self.save_bn_running_mean2.as_view_mut(ctx));
    self.bn_running_ivar2.as_view(ctx)
      .send(&mut self.save_bn_running_ivar2.as_view_mut(ctx));
  }

  fn restore_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.save_weights1.as_view(ctx)
      .send(&mut self.weights1.as_view_mut(ctx));

    self.save_bn_scale1.as_view(ctx)
      .send(&mut self.bn_scale1.as_view_mut(ctx));
    self.save_bn_bias1.as_view(ctx)
      .send(&mut self.bn_bias1.as_view_mut(ctx));
    self.save_bn_running_mean1.as_view(ctx)
      .send(&mut self.bn_running_mean1.as_view_mut(ctx));
    self.save_bn_running_ivar1.as_view(ctx)
      .send(&mut self.bn_running_ivar1.as_view_mut(ctx));

    backward.save_weights2.as_view(ctx)
      .send(&mut self.weights2.as_view_mut(ctx));

    self.save_bn_scale2.as_view(ctx)
      .send(&mut self.bn_scale2.as_view_mut(ctx));
    self.save_bn_bias2.as_view(ctx)
      .send(&mut self.bn_bias2.as_view_mut(ctx));
    self.save_bn_running_mean2.as_view(ctx)
      .send(&mut self.bn_running_mean2.as_view_mut(ctx));
    self.save_bn_running_ivar2.as_view(ctx)
      .send(&mut self.bn_running_ivar2.as_view_mut(ctx));
  }

  fn set_grads_with_params_diff(&mut self) {
    unimplemented!();
  }

  /*fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    /*assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.load(self.params_off, &mut self.weights1); //, ctx);
    comm_worker.load(self.params_off, &mut self.weights2); //, ctx);*/
    unimplemented!();
  }

  fn sync_params(&mut self) {
    /*assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights1); //, ctx);
    comm_worker.store(self.params_off, &mut self.weights2); //, ctx);*/
    unimplemented!();
  }*/

  fn stage_grads_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    assert!(self.backward.is_some());
    let mut backward = self.backward.as_mut().unwrap();

    let mut offset = offset;

    comm_worker.load(offset, &mut backward.grad_weights1);
    offset += backward.grad_weights1.len();

    comm_worker.load(offset, &mut self.bn_scale1_grad);
    offset += self.bn_scale1_grad.len();
    comm_worker.load(offset, &mut self.bn_bias1_grad);
    offset += self.bn_bias1_grad.len();
    offset += self.bn_running_mean1.len();
    offset += self.bn_running_ivar1.len();

    comm_worker.load(offset, &mut backward.grad_weights2);
    offset += backward.grad_weights2.len();

    comm_worker.load(offset, &mut self.bn_scale2_grad);
    offset += self.bn_scale2_grad.len();
    comm_worker.load(offset, &mut self.bn_bias2_grad);
    offset += self.bn_bias2_grad.len();
    offset += self.bn_running_mean2.len();
    offset += self.bn_running_ivar2.len();

    self.config.params_len()
  }

  fn merge_grads_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    assert!(self.backward.is_some());
    let mut backward = self.backward.as_mut().unwrap();

    let mut offset = offset;

    comm_worker.store(offset, &mut backward.grad_weights1);
    offset += backward.grad_weights1.len();

    comm_worker.store(offset, &mut self.bn_scale1_grad);
    offset += self.bn_scale1_grad.len();
    comm_worker.store(offset, &mut self.bn_bias1_grad);
    offset += self.bn_bias1_grad.len();
    offset += self.bn_running_mean1.len();
    offset += self.bn_running_ivar1.len();

    comm_worker.store(offset, &mut backward.grad_weights2);
    offset += backward.grad_weights2.len();

    comm_worker.store(offset, &mut self.bn_scale2_grad);
    offset += self.bn_scale2_grad.len();
    comm_worker.store(offset, &mut self.bn_bias2_grad);
    offset += self.bn_bias2_grad.len();
    offset += self.bn_running_mean2.len();
    offset += self.bn_running_ivar2.len();

    self.config.params_len()
  }

  fn stage_params_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    let mut offset = offset;

    comm_worker.load(offset, &mut self.weights1);
    offset += self.weights1.len();

    comm_worker.load(offset, &mut self.bn_scale1);
    offset += self.bn_scale1.len();
    comm_worker.load(offset, &mut self.bn_bias1);
    offset += self.bn_bias1.len();
    comm_worker.load(offset, &mut self.bn_running_mean1);
    offset += self.bn_running_mean1.len();
    comm_worker.load(offset, &mut self.bn_running_ivar1);
    offset += self.bn_running_ivar1.len();

    comm_worker.load(offset, &mut self.weights2);
    offset += self.weights2.len();

    comm_worker.load(offset, &mut self.bn_scale2);
    offset += self.bn_scale2.len();
    comm_worker.load(offset, &mut self.bn_bias2);
    offset += self.bn_bias2.len();
    comm_worker.load(offset, &mut self.bn_running_mean2);
    offset += self.bn_running_mean2.len();
    comm_worker.load(offset, &mut self.bn_running_ivar2);
    offset += self.bn_running_ivar2.len();

    self.config.params_len()
  }

  fn merge_params_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    let mut offset = offset;

    comm_worker.store(offset, &mut self.weights1);
    offset += self.weights1.len();

    comm_worker.store(offset, &mut self.bn_scale1);
    offset += self.bn_scale1.len();
    comm_worker.store(offset, &mut self.bn_bias1);
    offset += self.bn_bias1.len();
    comm_worker.store(offset, &mut self.bn_running_mean1);
    offset += self.bn_running_mean1.len();
    comm_worker.store(offset, &mut self.bn_running_ivar1);
    offset += self.bn_running_ivar1.len();

    comm_worker.store(offset, &mut self.weights2);
    offset += self.weights2.len();

    comm_worker.store(offset, &mut self.bn_scale2);
    offset += self.bn_scale2.len();
    comm_worker.store(offset, &mut self.bn_bias2);
    offset += self.bn_bias2.len();
    comm_worker.store(offset, &mut self.bn_running_mean2);
    offset += self.bn_running_mean2.len();
    comm_worker.store(offset, &mut self.bn_running_ivar2);
    offset += self.bn_running_ivar2.len();

    self.config.params_len()
  }

  /*fn reset_grads(&mut self, scale: f32) {
    unimplemented!();
  }*/

  fn reset(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.grad_weights1.as_view_mut(ctx)
      .matrix_scale(0.0);

    self.bn_scale1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
    self.bn_bias1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);

    backward.grad_weights2.as_view_mut(ctx)
      .matrix_scale(0.0);

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
  pub bnorm_mov_avg:    BNormMovingAverage,
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
    let batchnorm1_len = 4 * out_channels;
    let weights2_len = 3 * 3 * out_channels * out_channels;
    let batchnorm2_len = 4 * out_channels;
    let weights3_len = 1 * 1 * in_channels * out_channels;
    let batchnorm3_len = 4 * out_channels;
    weights1_len + batchnorm1_len + weights2_len + batchnorm2_len + weights3_len + batchnorm3_len
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
  weights2:     DeviceArray2d<f32>,
  weights3:     DeviceArray2d<f32>,

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
  conv2_fwd:    CudnnConvFwdOp,
  conv3_fwd:    CudnnConvFwdOp,

  bn_scale1:            DeviceArray2d<f32>,
  bn_scale1_grad:       DeviceArray2d<f32>,
  acc_bn_scale1_grad:   DeviceArray2d<f32>,
  save_bn_scale1:       DeviceArray2d<f32>,
  bn_bias1:             DeviceArray2d<f32>,
  bn_bias1_grad:        DeviceArray2d<f32>,
  acc_bn_bias1_grad:    DeviceArray2d<f32>,
  save_bn_bias1:        DeviceArray2d<f32>,
  bn_running_mean1:     DeviceArray2d<f32>,
  save_bn_running_mean1:    DeviceArray2d<f32>,
  bn_running_ivar1:     DeviceArray2d<f32>,
  save_bn_running_ivar1:    DeviceArray2d<f32>,
  bn_cached_mean1:      DeviceArray2d<f32>,
  bn_cached_ivar1:      DeviceArray2d<f32>,
  batchnorm1:           CudnnBatchNormOp,

  bn_scale2:            DeviceArray2d<f32>,
  bn_scale2_grad:       DeviceArray2d<f32>,
  acc_bn_scale2_grad:   DeviceArray2d<f32>,
  save_bn_scale2:       DeviceArray2d<f32>,
  bn_bias2:             DeviceArray2d<f32>,
  bn_bias2_grad:        DeviceArray2d<f32>,
  acc_bn_bias2_grad:    DeviceArray2d<f32>,
  save_bn_bias2:        DeviceArray2d<f32>,
  bn_running_mean2:     DeviceArray2d<f32>,
  save_bn_running_mean2:    DeviceArray2d<f32>,
  bn_running_ivar2:     DeviceArray2d<f32>,
  save_bn_running_ivar2:    DeviceArray2d<f32>,
  bn_cached_mean2:      DeviceArray2d<f32>,
  bn_cached_ivar2:      DeviceArray2d<f32>,
  batchnorm2:           CudnnBatchNormOp,

  bn_scale3:            DeviceArray2d<f32>,
  bn_scale3_grad:       DeviceArray2d<f32>,
  acc_bn_scale3_grad:   DeviceArray2d<f32>,
  save_bn_scale3:       DeviceArray2d<f32>,
  bn_bias3:             DeviceArray2d<f32>,
  bn_bias3_grad:        DeviceArray2d<f32>,
  acc_bn_bias3_grad:    DeviceArray2d<f32>,
  save_bn_bias3:        DeviceArray2d<f32>,
  bn_running_mean3:     DeviceArray2d<f32>,
  save_bn_running_mean3:    DeviceArray2d<f32>,
  bn_running_ivar3:     DeviceArray2d<f32>,
  save_bn_running_ivar3:    DeviceArray2d<f32>,
  bn_cached_mean3:      DeviceArray2d<f32>,
  bn_cached_ivar3:      DeviceArray2d<f32>,
  batchnorm3:           CudnnBatchNormOp,

  backward:     Option<ProjStackResConv2dBwdOperator<Comm>>,
  //hv_backward:  Option<BotResConv2dHvBwdOperator>,
}

struct ProjStackResConv2dBwdOperator<Comm> {
  grad_weights1:      DeviceArray2d<f32>,
  acc_grad_weights1:  DeviceArray2d<f32>,
  save_weights1:      DeviceArray2d<f32>,
  conv1_bwd_w:  CudnnConvBwdFilterOp,
  conv1_bwd_d:  CudnnConvBwdDataOp,

  first_batch1:    bool,

  grad_weights2:      DeviceArray2d<f32>,
  acc_grad_weights2:  DeviceArray2d<f32>,
  save_weights2:      DeviceArray2d<f32>,
  conv2_bwd_w:  CudnnConvBwdFilterOp,
  conv2_bwd_d:  CudnnConvBwdDataOp,

  first_batch2:    bool,

  grad_weights3:      DeviceArray2d<f32>,
  acc_grad_weights3:  DeviceArray2d<f32>,
  save_weights3:      DeviceArray2d<f32>,
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
        acc_grad_weights1:  DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, out_channels), ctx),
        save_weights1:      DeviceArray2d::<f32>::zeros((3 * 3 * in_channels, out_channels), ctx),

        grad_weights2:      DeviceArray2d::<f32>::zeros((3 * 3 * out_channels, out_channels), ctx),
        acc_grad_weights2:  DeviceArray2d::<f32>::zeros((3 * 3 * out_channels, out_channels), ctx),
        save_weights2:      DeviceArray2d::<f32>::zeros((3 * 3 * out_channels, out_channels), ctx),

        grad_weights3:      DeviceArray2d::<f32>::zeros((1 * 1 * in_channels, out_channels), ctx),
        acc_grad_weights3:  DeviceArray2d::<f32>::zeros((1 * 1 * in_channels, out_channels), ctx),
        save_weights3:      DeviceArray2d::<f32>::zeros((1 * 1 * in_channels, out_channels), ctx),

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
      weights2:     DeviceArray2d::<f32>::zeros((3 * 3 * out_channels, out_channels), ctx),
      weights3:     DeviceArray2d::<f32>::zeros((1 * 1 * in_channels, out_channels), ctx),

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
      conv2_fwd:    conv2_fwd,
      conv3_fwd:    conv3_fwd,

      //bn_scale1:            DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_scale1:            bn_scale1,
      bn_scale1_grad:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_scale1_grad:   DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_scale1:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias1:             DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias1_grad:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_bias1_grad:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_bias1:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_mean1:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_running_mean1:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_ivar1:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_running_ivar1:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_mean1:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_ivar1:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      batchnorm1:           batchnorm1,

      //bn_scale2:            DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_scale2:            bn_scale2,
      bn_scale2_grad:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_scale2_grad:   DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_scale2:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias2:             DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias2_grad:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_bias2_grad:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_bias2:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_mean2:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_running_mean2:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_ivar2:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_running_ivar2:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_mean2:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_cached_ivar2:      DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      batchnorm2:           batchnorm2,

      //bn_scale3:            DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_scale3:            bn_scale3,
      bn_scale3_grad:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_scale3_grad:   DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_scale3:       DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias3:             DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_bias3_grad:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      acc_bn_bias3_grad:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_bias3:        DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_mean3:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_running_mean3:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      bn_running_ivar3:     DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      save_bn_running_ivar3:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
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
    self.weights1.as_view_mut(ctx).sync_load(&init_weights1.as_view());
    self.weights2.as_view_mut(ctx).sync_load(&init_weights2.as_view());
    self.weights3.as_view_mut(ctx).sync_load(&init_weights3.as_view());

    self.bn_scale1.as_view_mut(ctx).set_constant(1.0);
    self.bn_bias1.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_mean1.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_ivar1.as_view_mut(ctx).set_constant(1.0);

    self.bn_scale2.as_view_mut(ctx).set_constant(1.0);
    self.bn_bias2.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_mean2.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_ivar2.as_view_mut(ctx).set_constant(1.0);

    self.bn_scale3.as_view_mut(ctx).set_constant(1.0);
    self.bn_bias3.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_mean3.as_view_mut(ctx).set_constant(0.0);
    self.bn_running_ivar3.as_view_mut(ctx).set_constant(1.0);
  }

  fn decode_params(&mut self, blob: &[u8]) -> usize {
    let ctx = &(*self.context).as_ref();
    let mut reader = Cursor::new(blob);

    let load_weights1 = Array2d::deserialize(&mut reader).unwrap();
    self.weights1.as_view_mut(ctx).sync_load(&load_weights1.as_view());

    let load_bn_scale1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_scale1.as_view_mut(ctx).sync_load(&load_bn_scale1.as_view());
    let load_bn_bias1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_bias1.as_view_mut(ctx).sync_load(&load_bn_bias1.as_view());
    let load_bn_running_mean1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_mean1.as_view_mut(ctx).sync_load(&load_bn_running_mean1.as_view());
    let load_bn_running_ivar1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_ivar1.as_view_mut(ctx).sync_load(&load_bn_running_ivar1.as_view());

    let load_weights2 = Array2d::deserialize(&mut reader).unwrap();
    self.weights2.as_view_mut(ctx).sync_load(&load_weights2.as_view());

    let load_bn_scale2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_scale2.as_view_mut(ctx).sync_load(&load_bn_scale2.as_view());
    let load_bn_bias2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_bias2.as_view_mut(ctx).sync_load(&load_bn_bias2.as_view());
    let load_bn_running_mean2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_mean2.as_view_mut(ctx).sync_load(&load_bn_running_mean2.as_view());
    let load_bn_running_ivar2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_ivar2.as_view_mut(ctx).sync_load(&load_bn_running_ivar2.as_view());

    let load_weights3 = Array2d::deserialize(&mut reader).unwrap();
    self.weights3.as_view_mut(ctx).sync_load(&load_weights3.as_view());

    let load_bn_scale3 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_scale3.as_view_mut(ctx).sync_load(&load_bn_scale3.as_view());
    let load_bn_bias3 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_bias3.as_view_mut(ctx).sync_load(&load_bn_bias3.as_view());
    let load_bn_running_mean3 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_mean3.as_view_mut(ctx).sync_load(&load_bn_running_mean3.as_view());
    let load_bn_running_ivar3 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_running_ivar3.as_view_mut(ctx).sync_load(&load_bn_running_ivar3.as_view());

    let progress = reader.position() as usize;
    progress
  }

  fn encode_params(&mut self, blob: &mut Vec<u8>) {
    let ctx = &(*self.context).as_ref();

    let weights1 = self.weights1.as_view(ctx);
    let mut save_weights1 = Array2d::zeros(weights1.bound());
    weights1.sync_store(&mut save_weights1.as_view_mut());
    save_weights1.serialize(blob).unwrap();

    let bn_scale1 = self.bn_scale1.as_view(ctx);
    let mut save_bn_scale1 = Array2d::zeros(bn_scale1.bound());
    bn_scale1.sync_store(&mut save_bn_scale1.as_view_mut());
    save_bn_scale1.serialize(blob).unwrap();

    let bn_bias1 = self.bn_bias1.as_view(ctx);
    let mut save_bn_bias1 = Array2d::zeros(bn_bias1.bound());
    bn_bias1.sync_store(&mut save_bn_bias1.as_view_mut());
    save_bn_bias1.serialize(blob).unwrap();

    let bn_running_mean1 = self.bn_running_mean1.as_view(ctx);
    let mut save_bn_running_mean1 = Array2d::zeros(bn_running_mean1.bound());
    bn_running_mean1.sync_store(&mut save_bn_running_mean1.as_view_mut());
    save_bn_running_mean1.serialize(blob).unwrap();

    let bn_running_ivar1 = self.bn_running_ivar1.as_view(ctx);
    let mut save_bn_running_ivar1 = Array2d::zeros(bn_running_ivar1.bound());
    bn_running_ivar1.sync_store(&mut save_bn_running_ivar1.as_view_mut());
    save_bn_running_ivar1.serialize(blob).unwrap();

    let weights2 = self.weights2.as_view(ctx);
    let mut save_weights2 = Array2d::zeros(weights2.bound());
    weights2.sync_store(&mut save_weights2.as_view_mut());
    save_weights2.serialize(blob).unwrap();

    let bn_scale2 = self.bn_scale2.as_view(ctx);
    let mut save_bn_scale2 = Array2d::zeros(bn_scale2.bound());
    bn_scale2.sync_store(&mut save_bn_scale2.as_view_mut());
    save_bn_scale2.serialize(blob).unwrap();

    let bn_bias2 = self.bn_bias2.as_view(ctx);
    let mut save_bn_bias2 = Array2d::zeros(bn_bias2.bound());
    bn_bias2.sync_store(&mut save_bn_bias2.as_view_mut());
    save_bn_bias2.serialize(blob).unwrap();

    let bn_running_mean2 = self.bn_running_mean2.as_view(ctx);
    let mut save_bn_running_mean2 = Array2d::zeros(bn_running_mean2.bound());
    bn_running_mean2.sync_store(&mut save_bn_running_mean2.as_view_mut());
    save_bn_running_mean2.serialize(blob).unwrap();

    let bn_running_ivar2 = self.bn_running_ivar2.as_view(ctx);
    let mut save_bn_running_ivar2 = Array2d::zeros(bn_running_ivar2.bound());
    bn_running_ivar2.sync_store(&mut save_bn_running_ivar2.as_view_mut());
    save_bn_running_ivar2.serialize(blob).unwrap();

    let weights3 = self.weights3.as_view(ctx);
    let mut save_weights3 = Array2d::zeros(weights3.bound());
    weights3.sync_store(&mut save_weights3.as_view_mut());
    save_weights3.serialize(blob).unwrap();

    let bn_scale3 = self.bn_scale3.as_view(ctx);
    let mut save_bn_scale3 = Array2d::zeros(bn_scale3.bound());
    bn_scale3.sync_store(&mut save_bn_scale3.as_view_mut());
    save_bn_scale3.serialize(blob).unwrap();

    let bn_bias3 = self.bn_bias3.as_view(ctx);
    let mut save_bn_bias3 = Array2d::zeros(bn_bias3.bound());
    bn_bias3.sync_store(&mut save_bn_bias3.as_view_mut());
    save_bn_bias3.serialize(blob).unwrap();

    let bn_running_mean3 = self.bn_running_mean3.as_view(ctx);
    let mut save_bn_running_mean3 = Array2d::zeros(bn_running_mean3.bound());
    bn_running_mean3.sync_store(&mut save_bn_running_mean3.as_view_mut());
    save_bn_running_mean3.serialize(blob).unwrap();

    let bn_running_ivar3 = self.bn_running_ivar3.as_view(ctx);
    let mut save_bn_running_ivar3 = Array2d::zeros(bn_running_ivar3.bound());
    bn_running_ivar3.sync_store(&mut save_bn_running_ivar3.as_view_mut());
    save_bn_running_ivar3.serialize(blob).unwrap();
  }

  fn decode_state(&mut self, blob: &[u8]) -> usize {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    let mut reader = Cursor::new(blob);

    let load_weights1 = Array2d::deserialize(&mut reader).unwrap();
    backward.acc_grad_weights1.as_view_mut(ctx).sync_load(&load_weights1.as_view());

    let load_bn_scale1 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_scale1_grad.as_view_mut(ctx).sync_load(&load_bn_scale1.as_view());
    let load_bn_bias1 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_bias1_grad.as_view_mut(ctx).sync_load(&load_bn_bias1.as_view());
    let bn_cached_mean1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_mean1.as_view_mut(ctx).sync_load(&bn_cached_mean1.as_view());
    let bn_cached_ivar1 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_ivar1.as_view_mut(ctx).sync_load(&bn_cached_ivar1.as_view());

    let load_weights2 = Array2d::deserialize(&mut reader).unwrap();
    backward.acc_grad_weights2.as_view_mut(ctx).sync_load(&load_weights2.as_view());

    let load_bn_scale2 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_scale2_grad.as_view_mut(ctx).sync_load(&load_bn_scale2.as_view());
    let load_bn_bias2 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_bias2_grad.as_view_mut(ctx).sync_load(&load_bn_bias2.as_view());
    let bn_cached_mean2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_mean2.as_view_mut(ctx).sync_load(&bn_cached_mean2.as_view());
    let bn_cached_ivar2 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_ivar2.as_view_mut(ctx).sync_load(&bn_cached_ivar2.as_view());

    let load_weights3 = Array2d::deserialize(&mut reader).unwrap();
    backward.acc_grad_weights3.as_view_mut(ctx).sync_load(&load_weights3.as_view());

    let load_bn_scale3 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_scale3_grad.as_view_mut(ctx).sync_load(&load_bn_scale3.as_view());
    let load_bn_bias3 = Array2d::deserialize(&mut reader).unwrap();
    self.acc_bn_bias3_grad.as_view_mut(ctx).sync_load(&load_bn_bias3.as_view());
    let bn_cached_mean3 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_mean3.as_view_mut(ctx).sync_load(&bn_cached_mean3.as_view());
    let bn_cached_ivar3 = Array2d::deserialize(&mut reader).unwrap();
    self.bn_cached_ivar3.as_view_mut(ctx).sync_load(&bn_cached_ivar3.as_view());

    let progress = reader.position() as usize;
    progress
  }

  fn encode_state(&mut self, blob: &mut Vec<u8>) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    let weights1 = backward.acc_grad_weights1.as_view(ctx);
    let mut save_weights1 = Array2d::zeros(weights1.bound());
    weights1.sync_store(&mut save_weights1.as_view_mut());
    save_weights1.serialize(blob).unwrap();

    let bn_scale1 = self.acc_bn_scale1_grad.as_view(ctx);
    let mut save_bn_scale1 = Array2d::zeros(bn_scale1.bound());
    bn_scale1.sync_store(&mut save_bn_scale1.as_view_mut());
    save_bn_scale1.serialize(blob).unwrap();

    let bn_bias1 = self.acc_bn_bias1_grad.as_view(ctx);
    let mut save_bn_bias1 = Array2d::zeros(bn_bias1.bound());
    bn_bias1.sync_store(&mut save_bn_bias1.as_view_mut());
    save_bn_bias1.serialize(blob).unwrap();

    let bn_running_mean1 = self.bn_cached_mean1.as_view(ctx);
    let mut save_bn_running_mean1 = Array2d::zeros(bn_running_mean1.bound());
    bn_running_mean1.sync_store(&mut save_bn_running_mean1.as_view_mut());
    save_bn_running_mean1.serialize(blob).unwrap();

    let bn_running_ivar1 = self.bn_cached_ivar1.as_view(ctx);
    let mut save_bn_running_ivar1 = Array2d::zeros(bn_running_ivar1.bound());
    bn_running_ivar1.sync_store(&mut save_bn_running_ivar1.as_view_mut());
    save_bn_running_ivar1.serialize(blob).unwrap();

    let weights2 = backward.acc_grad_weights2.as_view(ctx);
    let mut save_weights2 = Array2d::zeros(weights2.bound());
    weights2.sync_store(&mut save_weights2.as_view_mut());
    save_weights2.serialize(blob).unwrap();

    let bn_scale2 = self.acc_bn_scale2_grad.as_view(ctx);
    let mut save_bn_scale2 = Array2d::zeros(bn_scale2.bound());
    bn_scale2.sync_store(&mut save_bn_scale2.as_view_mut());
    save_bn_scale2.serialize(blob).unwrap();

    let bn_bias2 = self.acc_bn_bias2_grad.as_view(ctx);
    let mut save_bn_bias2 = Array2d::zeros(bn_bias2.bound());
    bn_bias2.sync_store(&mut save_bn_bias2.as_view_mut());
    save_bn_bias2.serialize(blob).unwrap();

    let bn_running_mean2 = self.bn_cached_mean2.as_view(ctx);
    let mut save_bn_running_mean2 = Array2d::zeros(bn_running_mean2.bound());
    bn_running_mean2.sync_store(&mut save_bn_running_mean2.as_view_mut());
    save_bn_running_mean2.serialize(blob).unwrap();

    let bn_running_ivar2 = self.bn_cached_ivar2.as_view(ctx);
    let mut save_bn_running_ivar2 = Array2d::zeros(bn_running_ivar2.bound());
    bn_running_ivar2.sync_store(&mut save_bn_running_ivar2.as_view_mut());
    save_bn_running_ivar2.serialize(blob).unwrap();

    let weights3 = backward.acc_grad_weights3.as_view(ctx);
    let mut save_weights3 = Array2d::zeros(weights3.bound());
    weights3.sync_store(&mut save_weights3.as_view_mut());
    save_weights3.serialize(blob).unwrap();

    let bn_scale3 = self.acc_bn_scale3_grad.as_view(ctx);
    let mut save_bn_scale3 = Array2d::zeros(bn_scale3.bound());
    bn_scale3.sync_store(&mut save_bn_scale3.as_view_mut());
    save_bn_scale3.serialize(blob).unwrap();

    let bn_bias3 = self.acc_bn_bias3_grad.as_view(ctx);
    let mut save_bn_bias3 = Array2d::zeros(bn_bias3.bound());
    bn_bias3.sync_store(&mut save_bn_bias3.as_view_mut());
    save_bn_bias3.serialize(blob).unwrap();

    let bn_running_mean3 = self.bn_cached_mean3.as_view(ctx);
    let mut save_bn_running_mean3 = Array2d::zeros(bn_running_mean3.bound());
    bn_running_mean3.sync_store(&mut save_bn_running_mean3.as_view_mut());
    save_bn_running_mean3.serialize(blob).unwrap();

    let bn_running_ivar3 = self.bn_cached_ivar3.as_view(ctx);
    let mut save_bn_running_ivar3 = Array2d::zeros(bn_running_ivar3.bound());
    bn_running_ivar3.sync_store(&mut save_bn_running_ivar3.as_view_mut());
    save_bn_running_ivar3.serialize(blob).unwrap();
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
      ref mut weights1,
      ref mut weights2,
      ref mut weights3,
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
      OpPhase::Training{t} => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        /*let ema_factor = match backward.first_batch1 {
          true  => {
            backward.first_batch1 = false;
            1.0
          }
          false => 0.01,
        };*/
        let ema_factor = self.config.bnorm_mov_avg.at_iter(t);
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
      OpPhase::Training{t} => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        /*let ema_factor = match backward.first_batch2 {
          true  => {
            backward.first_batch2 = false;
            1.0
          }
          false => 0.01,
        };*/
        let ema_factor = self.config.bnorm_mov_avg.at_iter(t);
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
      OpPhase::Training{t} => {
        let mut backward = match self.backward.as_mut() {
          Some(backward) => backward,
          None => panic!("batch norm training missing backward operator"),
        };
        /*let ema_factor = match backward.first_batch3 {
          true  => {
            backward.first_batch3 = false;
            1.0
          }
          false => 0.01,
        };*/
        let ema_factor = self.config.bnorm_mov_avg.at_iter(t);
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
      ref mut weights1,
      ref mut weights2,
      ref mut weights3,
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
      ref mut grad_weights1,
      ref mut grad_weights2,
      ref mut grad_weights3,
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
          backward.grad_weights2.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights2.as_view(ctx));
          backward.grad_weights3.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights3.as_view(ctx));
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
    backward.acc_grad_weights1.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights1.as_view(ctx));

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
    backward.acc_grad_weights2.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights2.as_view(ctx));

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
    backward.acc_grad_weights3.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights3.as_view(ctx));

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

    self.bn_scale1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale1_grad.as_view(ctx));
    self.bn_bias1.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias1_grad.as_view(ctx));

    self.weights2.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights2.as_view(ctx));

    self.bn_scale2.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale2_grad.as_view(ctx));
    self.bn_bias2.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias2_grad.as_view(ctx));

    self.weights3.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights3.as_view(ctx));

    self.bn_scale3.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_scale3_grad.as_view(ctx));
    self.bn_bias3.as_view_mut(ctx)
      .row_vector_sum(scale, &self.acc_bn_bias3_grad.as_view(ctx));
  }

  fn update_params2(&mut self, grad_scale: f32, update_scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    if grad_scale != 0.0 {
      self.weights1.as_view_mut(ctx)
        .matrix_sum(grad_scale, &backward.grad_weights1.as_view(ctx));

      self.bn_scale1.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_scale1_grad.as_view(ctx));
      self.bn_bias1.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_bias1_grad.as_view(ctx));

      self.weights2.as_view_mut(ctx)
        .matrix_sum(grad_scale, &backward.grad_weights2.as_view(ctx));

      self.bn_scale2.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_scale2_grad.as_view(ctx));
      self.bn_bias2.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_bias2_grad.as_view(ctx));

      self.weights3.as_view_mut(ctx)
        .matrix_sum(grad_scale, &backward.grad_weights3.as_view(ctx));

      self.bn_scale3.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_scale3_grad.as_view(ctx));
      self.bn_bias3.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &self.bn_bias3_grad.as_view(ctx));
    }

    if update_scale != 0.0 {
      self.weights1.as_view_mut(ctx)
        .matrix_sum(update_scale, &backward.acc_grad_weights1.as_view(ctx));

      self.bn_scale1.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_scale1_grad.as_view(ctx));
      self.bn_bias1.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_bias1_grad.as_view(ctx));

      self.weights2.as_view_mut(ctx)
        .matrix_sum(update_scale, &backward.acc_grad_weights2.as_view(ctx));

      self.bn_scale2.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_scale2_grad.as_view(ctx));
      self.bn_bias2.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_bias2_grad.as_view(ctx));

      self.weights3.as_view_mut(ctx)
        .matrix_sum(update_scale, &backward.acc_grad_weights3.as_view(ctx));

      self.bn_scale3.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_scale3_grad.as_view(ctx));
      self.bn_bias3.as_view_mut(ctx)
        .row_vector_sum(update_scale, &self.acc_bn_bias3_grad.as_view(ctx));
    }
  }

  fn save_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    self.weights1.as_view(ctx)
      .send(&mut backward.save_weights1.as_view_mut(ctx));

    self.bn_scale1.as_view(ctx)
      .send(&mut self.save_bn_scale1.as_view_mut(ctx));
    self.bn_bias1.as_view(ctx)
      .send(&mut self.save_bn_bias1.as_view_mut(ctx));
    self.bn_running_mean1.as_view(ctx)
      .send(&mut self.save_bn_running_mean1.as_view_mut(ctx));
    self.bn_running_ivar1.as_view(ctx)
      .send(&mut self.save_bn_running_ivar1.as_view_mut(ctx));

    self.weights2.as_view(ctx)
      .send(&mut backward.save_weights2.as_view_mut(ctx));

    self.bn_scale2.as_view(ctx)
      .send(&mut self.save_bn_scale2.as_view_mut(ctx));
    self.bn_bias2.as_view(ctx)
      .send(&mut self.save_bn_bias2.as_view_mut(ctx));
    self.bn_running_mean2.as_view(ctx)
      .send(&mut self.save_bn_running_mean2.as_view_mut(ctx));
    self.bn_running_ivar2.as_view(ctx)
      .send(&mut self.save_bn_running_ivar2.as_view_mut(ctx));

    self.weights3.as_view(ctx)
      .send(&mut backward.save_weights3.as_view_mut(ctx));

    self.bn_scale3.as_view(ctx)
      .send(&mut self.save_bn_scale3.as_view_mut(ctx));
    self.bn_bias3.as_view(ctx)
      .send(&mut self.save_bn_bias3.as_view_mut(ctx));
    self.bn_running_mean3.as_view(ctx)
      .send(&mut self.save_bn_running_mean3.as_view_mut(ctx));
    self.bn_running_ivar3.as_view(ctx)
      .send(&mut self.save_bn_running_ivar3.as_view_mut(ctx));
  }

  fn restore_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.save_weights1.as_view(ctx)
      .send(&mut self.weights1.as_view_mut(ctx));

    self.save_bn_scale1.as_view(ctx)
      .send(&mut self.bn_scale1.as_view_mut(ctx));
    self.save_bn_bias1.as_view(ctx)
      .send(&mut self.bn_bias1.as_view_mut(ctx));
    self.save_bn_running_mean1.as_view(ctx)
      .send(&mut self.bn_running_mean1.as_view_mut(ctx));
    self.save_bn_running_ivar1.as_view(ctx)
      .send(&mut self.bn_running_ivar1.as_view_mut(ctx));

    backward.save_weights2.as_view(ctx)
      .send(&mut self.weights2.as_view_mut(ctx));

    self.save_bn_scale2.as_view(ctx)
      .send(&mut self.bn_scale2.as_view_mut(ctx));
    self.save_bn_bias2.as_view(ctx)
      .send(&mut self.bn_bias2.as_view_mut(ctx));
    self.save_bn_running_mean2.as_view(ctx)
      .send(&mut self.bn_running_mean2.as_view_mut(ctx));
    self.save_bn_running_ivar2.as_view(ctx)
      .send(&mut self.bn_running_ivar2.as_view_mut(ctx));

    backward.save_weights3.as_view(ctx)
      .send(&mut self.weights3.as_view_mut(ctx));

    self.save_bn_scale3.as_view(ctx)
      .send(&mut self.bn_scale3.as_view_mut(ctx));
    self.save_bn_bias3.as_view(ctx)
      .send(&mut self.bn_bias3.as_view_mut(ctx));
    self.save_bn_running_mean3.as_view(ctx)
      .send(&mut self.bn_running_mean3.as_view_mut(ctx));
    self.save_bn_running_ivar3.as_view(ctx)
      .send(&mut self.bn_running_ivar3.as_view_mut(ctx));
  }

  fn set_grads_with_params_diff(&mut self) {
    unimplemented!();
  }

  /*fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    /*assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.load(self.params_off, &mut self.weights1); //, ctx);
    comm_worker.load(self.params_off, &mut self.weights2); //, ctx);
    comm_worker.load(self.params_off, &mut self.weights3); //, ctx);*/
    unimplemented!();
  }

  fn sync_params(&mut self) {
    /*assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights1); //, ctx);
    comm_worker.store(self.params_off, &mut self.weights2); //, ctx);
    comm_worker.store(self.params_off, &mut self.weights3); //, ctx);*/
    unimplemented!();
  }*/

  fn stage_grads_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    assert!(self.backward.is_some());
    let mut backward = self.backward.as_mut().unwrap();

    let mut offset = offset;

    comm_worker.load(offset, &mut backward.grad_weights1);
    offset += backward.grad_weights1.len();

    comm_worker.load(offset, &mut self.bn_scale1_grad);
    offset += self.bn_scale1_grad.len();
    comm_worker.load(offset, &mut self.bn_bias1_grad);
    offset += self.bn_bias1_grad.len();
    offset += self.bn_running_mean1.len();
    offset += self.bn_running_ivar1.len();

    comm_worker.load(offset, &mut backward.grad_weights2);
    offset += backward.grad_weights2.len();

    comm_worker.load(offset, &mut self.bn_scale2_grad);
    offset += self.bn_scale2_grad.len();
    comm_worker.load(offset, &mut self.bn_bias2_grad);
    offset += self.bn_bias2_grad.len();
    offset += self.bn_running_mean2.len();
    offset += self.bn_running_ivar2.len();

    comm_worker.load(offset, &mut backward.grad_weights3);
    offset += backward.grad_weights3.len();

    comm_worker.load(offset, &mut self.bn_scale3_grad);
    offset += self.bn_scale3_grad.len();
    comm_worker.load(offset, &mut self.bn_bias3_grad);
    offset += self.bn_bias3_grad.len();
    offset += self.bn_running_mean3.len();
    offset += self.bn_running_ivar3.len();

    self.config.params_len()
  }

  fn merge_grads_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    assert!(self.backward.is_some());
    let mut backward = self.backward.as_mut().unwrap();

    let mut offset = offset;

    comm_worker.store(offset, &mut backward.grad_weights1);
    offset += backward.grad_weights1.len();

    comm_worker.store(offset, &mut self.bn_scale1_grad);
    offset += self.bn_scale1_grad.len();
    comm_worker.store(offset, &mut self.bn_bias1_grad);
    offset += self.bn_bias1_grad.len();
    offset += self.bn_running_mean1.len();
    offset += self.bn_running_ivar1.len();

    comm_worker.store(offset, &mut backward.grad_weights2);
    offset += backward.grad_weights2.len();

    comm_worker.store(offset, &mut self.bn_scale2_grad);
    offset += self.bn_scale2_grad.len();
    comm_worker.store(offset, &mut self.bn_bias2_grad);
    offset += self.bn_bias2_grad.len();
    offset += self.bn_running_mean2.len();
    offset += self.bn_running_ivar2.len();

    comm_worker.store(offset, &mut backward.grad_weights3);
    offset += backward.grad_weights3.len();

    comm_worker.store(offset, &mut self.bn_scale3_grad);
    offset += self.bn_scale3_grad.len();
    comm_worker.store(offset, &mut self.bn_bias3_grad);
    offset += self.bn_bias3_grad.len();
    offset += self.bn_running_mean3.len();
    offset += self.bn_running_ivar3.len();

    self.config.params_len()
  }

  fn stage_params_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    let mut offset = offset;

    comm_worker.load(offset, &mut self.weights1);
    offset += self.weights1.len();

    comm_worker.load(offset, &mut self.bn_scale1);
    offset += self.bn_scale1.len();
    comm_worker.load(offset, &mut self.bn_bias1);
    offset += self.bn_bias1.len();
    comm_worker.load(offset, &mut self.bn_running_mean1);
    offset += self.bn_running_mean1.len();
    comm_worker.load(offset, &mut self.bn_running_ivar1);
    offset += self.bn_running_ivar1.len();

    comm_worker.load(offset, &mut self.weights2);
    offset += self.weights2.len();

    comm_worker.load(offset, &mut self.bn_scale2);
    offset += self.bn_scale2.len();
    comm_worker.load(offset, &mut self.bn_bias2);
    offset += self.bn_bias2.len();
    comm_worker.load(offset, &mut self.bn_running_mean2);
    offset += self.bn_running_mean2.len();
    comm_worker.load(offset, &mut self.bn_running_ivar2);
    offset += self.bn_running_ivar2.len();

    comm_worker.load(offset, &mut self.weights3);
    offset += self.weights3.len();

    comm_worker.load(offset, &mut self.bn_scale3);
    offset += self.bn_scale3.len();
    comm_worker.load(offset, &mut self.bn_bias3);
    offset += self.bn_bias3.len();
    comm_worker.load(offset, &mut self.bn_running_mean3);
    offset += self.bn_running_mean3.len();
    comm_worker.load(offset, &mut self.bn_running_ivar3);
    offset += self.bn_running_ivar3.len();

    self.config.params_len()
  }

  fn merge_params_v2(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    let mut offset = offset;

    comm_worker.store(offset, &mut self.weights1);
    offset += self.weights1.len();

    comm_worker.store(offset, &mut self.bn_scale1);
    offset += self.bn_scale1.len();
    comm_worker.store(offset, &mut self.bn_bias1);
    offset += self.bn_bias1.len();
    comm_worker.store(offset, &mut self.bn_running_mean1);
    offset += self.bn_running_mean1.len();
    comm_worker.store(offset, &mut self.bn_running_ivar1);
    offset += self.bn_running_ivar1.len();

    comm_worker.store(offset, &mut self.weights2);
    offset += self.weights2.len();

    comm_worker.store(offset, &mut self.bn_scale2);
    offset += self.bn_scale2.len();
    comm_worker.store(offset, &mut self.bn_bias2);
    offset += self.bn_bias2.len();
    comm_worker.store(offset, &mut self.bn_running_mean2);
    offset += self.bn_running_mean2.len();
    comm_worker.store(offset, &mut self.bn_running_ivar2);
    offset += self.bn_running_ivar2.len();

    comm_worker.store(offset, &mut self.weights3);
    offset += self.weights3.len();

    comm_worker.store(offset, &mut self.bn_scale3);
    offset += self.bn_scale3.len();
    comm_worker.store(offset, &mut self.bn_bias3);
    offset += self.bn_bias3.len();
    comm_worker.store(offset, &mut self.bn_running_mean3);
    offset += self.bn_running_mean3.len();
    comm_worker.store(offset, &mut self.bn_running_ivar3);
    offset += self.bn_running_ivar3.len();

    self.config.params_len()
  }

  /*fn reset_grads(&mut self, scale: f32) {
    unimplemented!();
  }*/

  fn reset(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    backward.grad_weights1.as_view_mut(ctx)
      .matrix_scale(0.0);

    self.bn_scale1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
    self.bn_bias1_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);

    backward.grad_weights2.as_view_mut(ctx)
      .matrix_scale(0.0);

    self.bn_scale2_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);
    self.bn_bias2_grad.as_view_mut(ctx)
      .row_vector_scale(0.0);

    backward.grad_weights3.as_view_mut(ctx)
      .matrix_scale(0.0);

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

#[derive(Clone, Copy)]
pub struct ProjBotResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub out_dims:     (usize, usize, usize),
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}
