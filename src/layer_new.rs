use data_new::{SampleDatum, SampleLabel};

use array::{
  Shape, Array, AsyncArray, ArrayView, ArrayViewMut,
  ArrayZeroExt, NdArraySerialize,
  Array2d, Array3d,
};
use array_cuda::device::{
  DeviceCtxRef, DeviceArray2d, DeviceArray3d, DeviceBuffer,
};
use array_cuda::device::comm::allreduce::{DeviceAllReduceWorker};
use array_cuda::device::ext::{DeviceCastBytesExt, DeviceNumExt};
use array_cuda::device::linalg::{BlasVectorExt, BlasMatrixExt, Transpose};
use array_cuda::device::memory::{DeviceZeroExt};
//use array_dist::comm::{DistAllReduceWorker};
use cuda_dnn::v4::{
  CudnnConvFwdOp, CudnnConvBwdFilterOp, CudnnConvBwdDataOp,
  CudnnAddOp, CudnnActKind, CudnnActOp, CudnnSoftmaxOp,
  CudnnTensorDesc, CudnnFilterDesc, CudnnConvDesc,
};
use cuda_dnn::v4::ffi::{cudnnConvolutionFwdAlgo_t};
//use float::ord::{F32InfNan};
use rembrandt_kernels::ffi::*;
use rng::xorshift::{Xorshiftplus128Rng};
use toggle::{Toggle, Disable, Enable};

use rand::{Rng, SeedableRng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{max};
use std::io::{Read, Write, Cursor};
use std::iter::{repeat};
use std::rc::{Rc};

/*pub trait LayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize);
  fn get_out_dims(&self) -> (usize, usize, usize);
  fn params_len(&self) -> usize;
}*/

pub trait Layer {
  // Introspection.
  fn config(&self) -> LayerConfig;
  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>>;
  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>>;

  // Layer parameters. By default, layers do not have any params.
  fn initialize_params(&mut self, shared_seed: [u64; 2], ctx: &DeviceCtxRef) {}
  fn initialize_gradients(&mut self, ctx: &DeviceCtxRef) {}
  fn load_params(&mut self, blob: &[u8], ctx: &DeviceCtxRef) -> usize { 0 }
  fn save_params(&mut self, blob: &mut Vec<u8>, ctx: &DeviceCtxRef) {}

  // The core layer functionality.
  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {}
  fn backward(&mut self, batch_size: usize, loss_scale: f32, ctx: &DeviceCtxRef) {}
  fn accumulate_gradients(&mut self, accum_mode: GradAccumMode, ctx: &DeviceCtxRef) {}
  fn reset_gradients(&mut self, scale: f32, ctx: &DeviceCtxRef) {}
  fn descend(&mut self, scale: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef) {}

  // Hessian-vector product.
  fn hv_forward(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {}
  fn hv_backward(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {}
  fn hv_cgsolve(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {}

  // Synchronous parallelism.
  fn dev_allreduce_load(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {}
  fn dev_allreduce_store(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {}
  //fn local_allreduce_gradients(&mut self, worker: &mut DeviceAllReduceWorker<f32>, ctx: &DeviceCtxRef) {}
  //fn global_allreduce_gradients(&mut self, worker: &mut DistAllReduceWorker<f32>, ctx: &DeviceCtxRef) {}
}

/*pub trait GradientLayer {
  fn g_backward(&mut self, batch_size: usize, loss_scale: f32, ctx: &DeviceCtxRef) {}
  fn g_descend(&mut self, step: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef) {}
  fn g_reset_gradient(&mut self, scale: f32, ctx: &DeviceCtxRef) {}
  fn g_allreduce_load_params(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {}
  fn g_allreduce_store_params(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {}
}

pub trait HessianVectorLayer {
  fn hv_reset_direction(&mut self, ctx: &DeviceCtxRef) {}
  fn hv_backward(&mut self, batch_size: usize, loss_scale: f32, ctx: &DeviceCtxRef) {}
  fn hv_descend(&mut self, step: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef) {}
  fn hv_reset_gradient(&mut self, scale: f32, ctx: &DeviceCtxRef) {}
  fn hv_allreduce_load_params(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {}
  fn hv_allreduce_store_params(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {}
}*/

pub trait InputLayer: Layer {
  fn downcast(&self) -> &Layer;
  fn preload_frame(&mut self, batch_idx: usize, frame: &SampleDatum, ctx: &DeviceCtxRef);
  //fn preload_frame(&mut self, batch_idx: usize, frame: &Array3d<u8>, ctx: &DeviceCtxRef);
  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8];
  fn load_frames(&mut self, batch_size: usize, ctx: &DeviceCtxRef);
}

pub trait LossLayer: Layer {
  fn downcast(&self) -> &Layer;

  fn preload_label(&mut self, batch_idx: usize, label: &SampleLabel, phase: Phase);
  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceCtxRef);

  fn preload_weight(&mut self, batch_idx: usize, weight: f32) { unimplemented!(); }
  fn load_weights(&mut self, batch_size: usize, ctx: &DeviceCtxRef) { unimplemented!(); }

  fn store_labels(&mut self, batch_idx: usize, phase: Phase, ctx: &DeviceCtxRef);
  fn get_labels(&self, batch_size: usize) -> &Array2d<i32>;
  fn count_accuracy(&self, batch_size: usize, phase: Phase) -> (usize, usize);
  //fn store_ranked_labels(&mut self, batch_size: usize, ctx: &DeviceCtxRef) { unimplemented!(); }
  //fn get_ranked_labels(&mut self, batch_size: usize) -> &Array2d<f32> { unimplemented!(); }

  fn store_probs(&mut self, batch_size: usize, ctx: &DeviceCtxRef);
  fn get_probs(&self, batch_size: usize) -> &Array2d<f32>;

  fn preload_mask(&mut self, batch_idx: usize, bytemask: &[u8]);
  fn expose_host_mask_buf(&mut self, batch_idx: usize) -> &mut [u8];
  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef);
  fn apply_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef);

  fn reset_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef);
  fn accumulate_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef);
  fn store_loss(&mut self, ctx: &DeviceCtxRef);
  fn get_loss(&self) -> f32;
}

pub type SharedDeviceBuf<T> = Rc<RefCell<DeviceBuffer<T>>>;

#[derive(Clone, Copy, Debug)]
pub enum Phase {
  Inference,
  Training,
}

#[derive(Clone, Copy, Debug)]
pub enum GradAccumMode {
  Add{scale: f32},
}

#[derive(Clone, Copy, Debug)]
pub enum GradSyncMode {
  Sum,
  Average,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ParamsInitialization {
  None,
  Zeros,
  Uniform{half_range: f32},
  Normal{mean: f32, std: f32},
  Glorot,
}

impl Eq for ParamsInitialization {
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ActivationFunction {
  Identity,
  Rect,
  Sigmoid,
  ScaledSigmoid{beta: f32},
  Tanh,
}

impl Eq for ActivationFunction {
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AffineBackend {
  CublasGemm,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConvBackend {
  CudnnGemm,
  CudnnImplicitGemm,
  CudnnImplicitPrecompGemm,
  CudnnFft,
  CudnnFftTiling,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LayerConfig {
  Data3d(Data3dLayerConfig),
  Affine(AffineLayerConfig),
  Conv2d(Conv2dLayerConfig),
  Conv2d_V2(Conv2dLayerConfigV2),
  /*ResidualConv2d(Conv2dLayerConfig),
  AffineConv2d(AffineConv2dLayerConfig),*/
  SoftmaxKLLoss(CategoricalLossLayerConfig),
  SoftmaxIndicatorLoss(CategoricalLossLayerConfig),
  LogisticIndicatorLoss(CategoricalLossLayerConfig),
  AntilogisticKLLoss(CategoricalLossLayerConfig),
  AntilogisticIndicatorLoss(CategoricalLossLayerConfig),
  MultiSoftmaxKLLoss(MultiCategoricalLossLayerConfig),
}

impl LayerConfig {
  pub fn params_len(&self) -> usize {
    match *self {
      LayerConfig::Affine(cfg) => cfg.params_len(),
      LayerConfig::Conv2d(cfg) => cfg.params_len(),
      LayerConfig::Conv2d_V2(cfg) => cfg.params_len(),
      _ => unimplemented!(),
    }
  }

  pub fn build_layer(&self, batch_size: usize, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> Box<Layer> {
    match *self {
      LayerConfig::Affine(cfg) => {
        Box::new(AffineLayer::new(batch_size, cfg, prev_layer, ctx))
      }
      LayerConfig::Conv2d(cfg) => {
        Box::new(Conv2dLayer::new(batch_size, cfg, prev_layer, ctx))
      }
      LayerConfig::Conv2d_V2(cfg) => {
        Box::new(Conv2dLayer::new_v2(batch_size, cfg, prev_layer, ctx))
      }
      _ => unimplemented!(),
    }
  }

  pub fn build_input_layer(&self, batch_size: usize, ctx: &DeviceCtxRef) -> Box<InputLayer> {
    match *self {
      LayerConfig::Data3d(cfg) => {
        Box::new(Data3dLayer::new(batch_size, cfg, ctx))
      }
      _ => unimplemented!(),
    }
  }

  pub fn build_loss_layer(&self, batch_size: usize, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> Box<LossLayer> {
    match *self {
      LayerConfig::SoftmaxKLLoss(cfg) => {
        Box::new(SoftmaxKLLossLayer::new(batch_size, cfg, prev_layer, ctx))
      }
      LayerConfig::SoftmaxIndicatorLoss(cfg) => {
        Box::new(SoftmaxIndicatorLossLayer::new(batch_size, cfg, prev_layer, ctx))
      }
      LayerConfig::LogisticIndicatorLoss(cfg) => {
        Box::new(LogisticIndicatorLossLayer::new(batch_size, cfg, prev_layer, ctx))
      }
      /*LayerConfig::AntilogisticKLLoss(cfg) => {
        Box::new(AntilogisticKLLossLayer::new(batch_size, cfg, prev_layer, ctx))
      }*/
      LayerConfig::MultiSoftmaxKLLoss(cfg) => {
        Box::new(MultiSoftmaxKLLossLayer::new(batch_size, cfg, prev_layer, ctx))
      }
      _ => unimplemented!(),
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Data3dLayerConfig {
  pub dims:         (usize, usize, usize),
  pub normalize:    bool,
}

pub struct Data3dLayer {
  batch_limit:  usize,
  config:       Data3dLayerConfig,

  in_buf_h:     Vec<u8>,
  in_buf:       DeviceBuffer<u8>,
  out_buf:      SharedDeviceBuf<f32>,
}

impl Data3dLayer {
  pub fn new(batch_size: usize, config: Data3dLayerConfig, ctx: &DeviceCtxRef) -> Data3dLayer {
    let length = config.dims.len();
    Data3dLayer{
      batch_limit:  batch_size,
      config:       config,
      in_buf_h:     repeat(0).take(length * batch_size).collect(),
      in_buf:       DeviceBuffer::<u8>::zeros(length * batch_size, ctx),
      out_buf:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(length * batch_size, ctx))),
    }
  }
}

impl Layer for Data3dLayer {
  fn config(&self) -> LayerConfig {
    LayerConfig::Data3d(self.config)
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_buf.clone())
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    let length = self.config.dims.len();
    let in_buf = self.in_buf.as_ref(ctx);
    let mut out_buf = self.out_buf.borrow_mut().as_ref_mut(ctx);
    if self.config.normalize {
      /*unsafe { rembrandt_kernel_map_cast_byte_to_float_normalized(
          in_buf.as_ptr(),
          (length * batch_size) as i32,
          out_buf.as_mut_ptr(),
          ctx.stream.ptr,
      ) };*/
      in_buf.cast_bytes_normalized(&mut out_buf);
    } else {
      /*unsafe { rembrandt_kernel_map_cast_byte_to_float(
          in_buf.as_ptr(),
          (length * batch_size) as i32,
          out_buf.as_mut_ptr(),
          ctx.stream.ptr,
      ) };*/
      in_buf.cast_bytes(&mut out_buf);
    }
  }
}

impl InputLayer for Data3dLayer {
  fn downcast(&self) -> &Layer {
    self
  }

  fn preload_frame(&mut self, batch_idx: usize, frame: &SampleDatum, ctx: &DeviceCtxRef) {
    match frame {
      &SampleDatum::WHCBytes(ref frame_bytes) => {
        self.expose_host_frame_buf(batch_idx)
          .copy_from_slice(frame_bytes.as_slice());
      }
      _ => unimplemented!(),
    }
  }

  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    assert!(batch_idx < self.batch_limit);
    let length = self.config.dims.len();
    &mut self.in_buf_h[batch_idx * length .. (batch_idx + 1) * length]
  }

  fn load_frames(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    {
      let in_buf_h = &self.in_buf_h;
      let mut in_buf = self.in_buf.as_ref_mut(ctx);
      in_buf.sync_load(in_buf_h);
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AffineLayerConfig {
  pub in_channels:  usize,
  pub out_channels: usize,
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInitialization,
  pub backend:      AffineBackend,
}

impl AffineLayerConfig {
  pub fn params_len(&self) -> usize {
    let weights_len = self.in_channels * self.out_channels;
    let bias_len = self.out_channels;
    weights_len + bias_len
  }
}

pub struct AffineLayer {
  batch_limit:  usize,
  config:       AffineLayerConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights:      DeviceArray2d<f32>,
  bias:         DeviceArray2d<f32>,
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,

  add_bias:     CudnnAddOp,
  unit:         DeviceBuffer<f32>,
}

impl AffineLayer {
  pub fn new(batch_size: usize, config: AffineLayerConfig, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> AffineLayer {
    let AffineLayerConfig{
      in_channels, out_channels, .. } = config;

    let add_bias = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, batch_size).unwrap(),
    );
    
    let mut unit = DeviceBuffer::<f32>::zeros(batch_size, ctx);
    unit.as_ref_mut(ctx).set_constant(1.0);

    AffineLayer{
      batch_limit:  batch_size,
      config:       config,

      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_channels * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_channels * batch_size, ctx))),

      weights:      DeviceArray2d::<f32>::zeros((in_channels, out_channels), ctx),
      bias:         DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      grad_weights: DeviceArray2d::<f32>::zeros((in_channels, out_channels), ctx),
      grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),

      add_bias:     add_bias,
      unit:         unit,
    }
  }
}

impl Layer for AffineLayer {
  fn config(&self) -> LayerConfig {
    LayerConfig::Affine(self.config)
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn initialize_params(&mut self, seed: [u64; 2], ctx: &DeviceCtxRef) {
    let AffineLayerConfig{in_channels, out_channels, ..} = self.config;
    let mut rng = Xorshiftplus128Rng::from_seed(seed);
    let mut init_weights = Array2d::zeros((in_channels, out_channels));
    match self.config.init_weights {
      ParamsInitialization::None => {
        panic!("AffineLayer: params initialization explicitly disabled!");
      }
      ParamsInitialization::Uniform{half_range} => {
        let dist = Range::new(-half_range as f64, half_range as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInitialization::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      _ => unimplemented!(),
    }
    let init_bias = Array2d::zeros((1, out_channels));
    self.weights.as_view_mut(ctx).sync_load(&init_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&init_bias.as_view());
  }

  fn initialize_gradients(&mut self, ctx: &DeviceCtxRef) {
    self.grad_weights.as_view_mut(ctx).set_constant(0.0);
    self.grad_bias.as_view_mut(ctx).set_constant(0.0);
  }

  fn load_params(&mut self, blob: &[u8], ctx: &DeviceCtxRef) -> usize {
    let AffineLayerConfig{in_channels, out_channels, ..} = self.config;
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("AffineLayer failed to deserialize weights!");
    let load_bias = Array2d::deserialize(&mut reader)
      .ok().expect("AffineLayer failed to deserialize bias!");
    assert_eq!((in_channels, out_channels), load_weights.as_view().bound());
    assert_eq!((1, out_channels), load_bias.as_view().bound());
    self.weights.as_view_mut(ctx).sync_load(&load_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&load_bias.as_view());
    let progress = reader.position() as usize;
    //println!("DEBUG: AffineLayer: load params: read {}", progress);
    progress
  }

  fn save_params(&mut self, blob: &mut Vec<u8>, ctx: &DeviceCtxRef) {
    let weights = self.weights.as_view(ctx);
    let bias = self.bias.as_view(ctx);
    let mut save_weights = Array2d::zeros(weights.bound());
    let mut save_bias = Array2d::zeros(bias.bound());
    weights.sync_store(&mut save_weights.as_view_mut());
    bias.sync_store(&mut save_bias.as_view_mut());
    save_weights.serialize(blob).unwrap();
    save_bias.serialize(blob).unwrap();
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);

    let &mut AffineLayer{
      ref config,
      ref mut in_act, ref mut out_act,
      ref mut weights, ref mut bias,
      ref mut add_bias,
      .. } = self;

    let &AffineLayerConfig{
      in_channels, out_channels, .. } = config;

    let weights = weights.as_view(ctx);
    let bias = bias.as_view(ctx);
    let in_act = in_act.borrow_mut().as_ref(ctx).into_2d_view((in_channels, batch_size));
    let mut out_act = out_act.borrow_mut().as_ref_mut(ctx).into_2d_view_mut((out_channels, batch_size));

    out_act.matrix_prod(1.0, &weights, Transpose::T, &in_act, Transpose::N, 0.0);

    add_bias.set_batch_size(batch_size).unwrap();
    unsafe { add_bias.forward(
        1.0,
        bias.as_ptr(),
        1.0,
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

  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);

    let &mut AffineLayer{
      ref config,
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights,
      ref mut grad_weights, ref mut grad_bias,
      ref mut unit,
      .. } = self;

    let &AffineLayerConfig{
      in_channels, out_channels, .. } = config;

    let weights = weights.as_view(ctx);
    let mut grad_weights = grad_weights.as_view_mut(ctx);
    let mut grad_bias = grad_bias.as_view_mut(ctx);
    let in_act = in_act.borrow_mut().as_ref(ctx).into_2d_view((in_channels, batch_size));
    let out_act = out_act.borrow_mut().as_ref(ctx).into_2d_view((out_channels, batch_size));
    let unit = unit.as_ref(ctx).into_2d_view((1, batch_size));

    {
      let mut out_delta = out_delta.borrow_mut().as_ref_mut(ctx).into_2d_view_mut((out_channels, batch_size));
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

    let out_delta = out_delta.borrow_mut().as_ref(ctx).into_2d_view((out_channels, batch_size));

    grad_weights.matrix_prod(1.0, &in_act, Transpose::N, &out_delta, Transpose::T, 1.0);
    grad_bias.matrix_prod(1.0, &unit, Transpose::N, &out_delta, Transpose::T, 1.0);

    if let Some(in_delta) = in_delta.as_mut() {
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx).into_2d_view_mut((in_channels, batch_size));
      in_delta.matrix_prod(1.0, &weights, Transpose::N, &out_delta, Transpose::N, 0.0);
    }
  }

  fn accumulate_gradients(&mut self, accum_mode: GradAccumMode, ctx: &DeviceCtxRef) {
    match accum_mode {
      GradAccumMode::Add{scale} => {
        // FIXME(20160223)
        unimplemented!();
      }
    }
  }

  fn reset_gradients(&mut self, scale: f32, ctx: &DeviceCtxRef) {
    self.grad_weights.as_view_mut(ctx)
      .matrix_scale(scale);
    self.grad_bias.as_view_mut(ctx)
      .row_vector_scale(scale);
  }

  fn descend(&mut self, scale: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef) {
    assert!(l2_reg_coef >= 0.0);
    if l2_reg_coef > 0.0 {
      self.grad_weights.as_view_mut(ctx)
        .matrix_sum(l2_reg_coef, &self.weights.as_view(ctx));
      self.grad_bias.as_view_mut(ctx)
        .row_vector_sum(l2_reg_coef, &self.bias.as_view(ctx));
    }
    {
      self.weights.as_view_mut(ctx)
        .matrix_sum(-scale, &self.grad_weights.as_view(ctx));
      self.bias.as_view_mut(ctx)
        .row_vector_sum(-scale, &self.grad_bias.as_view(ctx));
    }
  }

  fn dev_allreduce_load(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {
    let bias_offset = worker.load(offset, &mut self.grad_weights.as_view(ctx).data);
    let end_offset = worker.load(bias_offset, &mut self.grad_bias.as_view(ctx).data);
    assert_eq!(end_offset - offset, self.config.params_len());
  }

  fn dev_allreduce_store(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {
    let bias_offset = worker.store(offset, &mut self.grad_weights.as_view_mut(ctx).data);
    let end_offset = worker.store(bias_offset, &mut self.grad_bias.as_view_mut(ctx).data);
    assert_eq!(end_offset - offset, self.config.params_len());
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Conv2dLayerConfig {
  pub in_dims:      (usize, usize, usize),
  pub conv_size:    usize,
  pub conv_stride:  usize,
  pub conv_pad:     usize,
  pub out_channels: usize,
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInitialization,
  //pub backend:      ConvBackend,
}

/*impl LayerConfig for Conv2dLayerConfig {
}*/

impl Conv2dLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    //(self.in_width, self.in_height, self.in_channels)
    self.in_dims
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    // XXX(20150926): Note that pool layer uses ceil((L+2*pad-size)/stride)
    // whereas conv layer uses floor((L+2*pad-size)/stride).
    let (in_width, in_height, _) = self.in_dims;
    let out_width = max(0, (in_width + 2 * self.conv_pad - self.conv_size) as isize) as usize / self.conv_stride + 1;
    let out_height = max(0, (in_height + 2 * self.conv_pad - self.conv_size) as isize) as usize / self.conv_stride + 1;
    (out_width, out_height, self.out_channels)
  }

  fn params_len(&self) -> usize {
    let (_, _, in_channels) = self.in_dims;
    let weights_len = self.conv_size * self.conv_size * in_channels * self.out_channels;
    let bias_len = self.out_channels;
    weights_len + bias_len
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Conv2dLayerConfigV2 {
  pub in_dims:      (usize, usize, usize),
  pub conv_size:    usize,
  pub conv_stride:  usize,
  pub conv_pad:     usize,
  pub out_channels: usize,
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInitialization,
  pub backend:      ConvBackend,
}

/*impl LayerConfig for Conv2dLayerConfig {
}*/

impl Conv2dLayerConfigV2 {
  fn into_v1_config(&self) -> Conv2dLayerConfig {
    Conv2dLayerConfig{
      in_dims:      self.in_dims,
      conv_size:    self.conv_size,
      conv_stride:  self.conv_stride,
      conv_pad:     self.conv_pad,
      out_channels: self.out_channels,
      act_func:     self.act_func,
      init_weights: self.init_weights,
    }
  }

  fn get_in_dims(&self) -> (usize, usize, usize) {
    //(self.in_width, self.in_height, self.in_channels)
    self.in_dims
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    // XXX(20150926): Note that pool layer uses ceil((L+2*pad-size)/stride)
    // whereas conv layer uses floor((L+2*pad-size)/stride).
    let (in_width, in_height, _) = self.in_dims;
    let out_width = max(0, (in_width + 2 * self.conv_pad - self.conv_size) as isize) as usize / self.conv_stride + 1;
    let out_height = max(0, (in_height + 2 * self.conv_pad - self.conv_size) as isize) as usize / self.conv_stride + 1;
    (out_width, out_height, self.out_channels)
  }

  fn params_len(&self) -> usize {
    let (_, _, in_channels) = self.in_dims;
    let weights_len = self.conv_size * self.conv_size * in_channels * self.out_channels;
    let bias_len = self.out_channels;
    weights_len + bias_len
  }
}

struct Conv2dHvData {
  r_in_act:     SharedDeviceBuf<f32>,
  r_in_delta:   Option<SharedDeviceBuf<f32>>,
  r_out_act:    SharedDeviceBuf<f32>,
  r_out_delta:  SharedDeviceBuf<f32>,

  direction_w:  DeviceArray2d<f32>,
  direction_b:  DeviceArray2d<f32>,
  r_grad_w:     DeviceArray2d<f32>,
  r_grad_b:     DeviceArray2d<f32>,
}

pub struct Conv2dLayer<HvToggle=Disable<Conv2dHvData>>
where HvToggle: Toggle<Conv2dHvData> {
  batch_limit:  usize,
  config:       Conv2dLayerConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights:      DeviceArray2d<f32>,
  bias:         DeviceArray2d<f32>,
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,

  hv_data:      HvToggle,

  work_space:   DeviceBuffer<u8>,
  conv_fwd:     CudnnConvFwdOp,
  conv_bwd_w:   CudnnConvBwdFilterOp,
  conv_bwd_d:   CudnnConvBwdDataOp,
  add_bias:     CudnnAddOp,
}

impl Conv2dLayer<Disable<Conv2dHvData>> {
  pub fn new(batch_size: usize, config: Conv2dLayerConfig, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> Conv2dLayer {
    let Conv2dLayerConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    let mut work_size = 0;
    let conv_fwd = CudnnConvFwdOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    work_size = max(work_size, conv_fwd.work_size);
    println!("DEBUG: Conv2dLayer: forward algo: {:?}", conv_fwd.algo);
    println!("DEBUG: Conv2dLayer: forward work: {:?}", conv_fwd.work_size);
    //let conv_bwd_w = CudnnConvBwdFilterOp::create_deterministic(
    let conv_bwd_w = CudnnConvBwdFilterOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    work_size = max(work_size, conv_bwd_w.work_size);
    //let conv_bwd_d = CudnnConvBwdDataOp::create_deterministic(
    let conv_bwd_d = CudnnConvBwdDataOp::create_fastest(
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    work_size = max(work_size, conv_bwd_d.work_size);

    let add_bias = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
    );

    Conv2dLayer{
      batch_limit:  batch_size,
      config:       config,

      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),

      weights:      DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
      bias:         DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
      grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),

      hv_data:      Disable::new(),

      work_space:   DeviceBuffer::<u8>::zeros(work_size, ctx),

      conv_fwd:     conv_fwd,
      conv_bwd_w:   conv_bwd_w,
      conv_bwd_d:   conv_bwd_d,
      add_bias:     add_bias,
    }
  }

  pub fn new_v2(batch_size: usize, config: Conv2dLayerConfigV2, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> Conv2dLayer {
    let Conv2dLayerConfigV2{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    let algo = match config.backend {
      ConvBackend::CudnnImplicitPrecompGemm => cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
      ConvBackend::CudnnFftTiling           => cudnnConvolutionFwdAlgo_t::FftTiling,
      _ => unimplemented!(),
    };

    let mut work_size = 0;
    let conv_fwd = CudnnConvFwdOp::create_algo(
        algo,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    work_size = max(work_size, conv_fwd.work_size);
    println!("DEBUG: Conv2dLayer: forward algo: {:?}", conv_fwd.algo);
    println!("DEBUG: Conv2dLayer: forward work: {:?}", conv_fwd.work_size);
    //let conv_bwd_w = CudnnConvBwdFilterOp::create_deterministic(
    let conv_bwd_w = CudnnConvBwdFilterOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    work_size = max(work_size, conv_bwd_w.work_size);
    //let conv_bwd_d = CudnnConvBwdDataOp::create_deterministic(
    let conv_bwd_d = CudnnConvBwdDataOp::create_fastest(
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    work_size = max(work_size, conv_bwd_d.work_size);

    let add_bias = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
    );

    Conv2dLayer{
      batch_limit:  batch_size,
      config:       config.into_v1_config(),

      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),

      weights:      DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
      bias:         DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
      grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
      grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),

      hv_data:      Disable::new(),

      work_space:   DeviceBuffer::<u8>::zeros(work_size, ctx),

      conv_fwd:     conv_fwd,
      conv_bwd_w:   conv_bwd_w,
      conv_bwd_d:   conv_bwd_d,
      add_bias:     add_bias,
    }
  }
}

impl Layer for Conv2dLayer<Disable<Conv2dHvData>> {
  fn config(&self) -> LayerConfig {
    LayerConfig::Conv2d(self.config)
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn initialize_params(&mut self, seed: [u64; 2], ctx: &DeviceCtxRef) {
    let Conv2dLayerConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let (_, _, in_channels) = in_dims;
    let mut rng = Xorshiftplus128Rng::from_seed(seed);
    let mut init_weights = Array2d::zeros((conv_size * conv_size * in_channels, out_channels));
    match self.config.init_weights {
      ParamsInitialization::None => {
        panic!("params initialization explicitly disabled!");
      }
      ParamsInitialization::Uniform{half_range} => {
        let dist = Range::new(-half_range as f64, half_range as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInitialization::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      _ => unimplemented!(),
    }
    let init_bias = Array2d::zeros((1, out_channels));
    self.weights.as_view_mut(ctx).sync_load(&init_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&init_bias.as_view());
  }

  fn initialize_gradients(&mut self, ctx: &DeviceCtxRef) {
    self.grad_weights.as_view_mut(ctx).set_constant(0.0);
    self.grad_bias.as_view_mut(ctx).set_constant(0.0);
  }

  fn load_params(&mut self, blob: &[u8], ctx: &DeviceCtxRef) -> usize {
    let Conv2dLayerConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let (_, _, in_channels) = in_dims;
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("Conv2dLayer failed to deserialize weights!");
    let load_bias = Array2d::deserialize(&mut reader)
      .ok().expect("Conv2dLayer failed to deserialize bias!");
    assert_eq!((conv_size * conv_size * in_channels, out_channels), load_weights.as_view().bound());
    assert_eq!((1, out_channels), load_bias.as_view().bound());
    self.weights.as_view_mut(ctx).sync_load(&load_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&load_bias.as_view());
    let progress = reader.position() as usize;
    //println!("DEBUG: Conv2dLayer: load params: read {}", progress);
    progress
  }

  fn save_params(&mut self, blob: &mut Vec<u8>, ctx: &DeviceCtxRef) {
    let weights = self.weights.as_view(ctx);
    let bias = self.bias.as_view(ctx);
    let mut save_weights = Array2d::zeros(weights.bound());
    let mut save_bias = Array2d::zeros(bias.bound());
    weights.sync_store(&mut save_weights.as_view_mut());
    bias.sync_store(&mut save_bias.as_view_mut());
    save_weights.serialize(blob).unwrap();
    save_bias.serialize(blob).unwrap();
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    let Conv2dLayerConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = self.config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = self.config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let in_length = in_dims.len(); //in_width * in_height * in_channels;
    let out_length = out_dims.len();

    let &mut Conv2dLayer{
      ref mut in_act, ref mut out_act,
      ref mut work_space,
      ref mut weights, ref mut bias,
      .. } = self;

    let mut out_act = out_act.borrow_mut().as_ref_mut(ctx);

    //println!("DEBUG: Conv2dLayer: forward batch size: {}", batch_size);
    self.conv_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv_fwd.forward(
        1.0,
        in_act.borrow_mut().as_ref(ctx).as_ptr(),
        weights.as_view(ctx).as_ptr(),
        0.0,
        out_act.as_mut_ptr(),
        work_space.as_ref_mut(ctx).as_mut_ptr(),
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
      ActivationFunction::ScaledSigmoid{beta} => {
        unsafe { rembrandt_kernel_batch_map_sigmoid_inplace_forward(
            out_act.as_mut_ptr(),
            out_length as i32,
            batch_size as i32,
            beta,
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }
  }

  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    let Conv2dLayerConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = self.config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = self.config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let in_length = in_width * in_height * in_channels;
    let out_length = out_dims.len();

    let &mut Conv2dLayer{
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights, ref mut bias,
      ref mut grad_weights, ref mut grad_bias,
      ref mut work_space,
      .. } = self;

    let mut in_act = in_act.borrow_mut().as_ref(ctx);
    let mut out_act = out_act.borrow_mut().as_ref(ctx);
    let mut out_delta = out_delta.borrow_mut().as_ref_mut(ctx);
    let mut work_space = work_space.as_ref_mut(ctx);

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
      ActivationFunction::ScaledSigmoid{beta} => {
        unsafe { rembrandt_kernel_batch_map_sigmoid_inplace_backward(
            out_act.as_ptr(),
            out_length as i32,
            batch_size as i32,
            out_delta.as_mut_ptr(),
            beta,
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }

    self.conv_bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { self.conv_bwd_w.backward_filter(
        scale,
        in_act.as_ptr(),
        out_delta.as_ptr(),
        1.0,
        grad_weights.as_view_mut(ctx).as_mut_ptr(),
        work_space.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    unsafe { self.conv_bwd_w.backward_bias(
        scale,
        out_delta.as_ptr(),
        1.0,
        grad_bias.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    if let &mut Some(ref mut in_delta) = in_delta {
      self.conv_bwd_d.set_batch_size(batch_size).unwrap();
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);
      unsafe { self.conv_bwd_d.backward_data(
          1.0,
          weights.as_view(ctx).as_ptr(),
          out_delta.as_ptr(),
          0.0,
          in_delta.as_mut_ptr(),
          work_space.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
    }
  }

  fn accumulate_gradients(&mut self, accum_mode: GradAccumMode, ctx: &DeviceCtxRef) {
    match accum_mode {
      GradAccumMode::Add{scale} => {
        // FIXME(20160223)
        unimplemented!();
      }
    }
  }

  fn reset_gradients(&mut self, scale: f32, ctx: &DeviceCtxRef) {
    self.grad_weights.as_view_mut(ctx)
      .matrix_scale(scale);
    self.grad_bias.as_view_mut(ctx)
      .row_vector_scale(scale);
  }

  fn descend(&mut self, scale: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef) {
    assert!(l2_reg_coef >= 0.0);
    if l2_reg_coef > 0.0 {
      self.grad_weights.as_view_mut(ctx)
        .matrix_sum(l2_reg_coef, &self.weights.as_view(ctx));
      self.grad_bias.as_view_mut(ctx)
        .row_vector_sum(l2_reg_coef, &self.bias.as_view(ctx));
    }
    {
      self.weights.as_view_mut(ctx)
        .matrix_sum(-scale, &self.grad_weights.as_view(ctx));
      self.bias.as_view_mut(ctx)
        .row_vector_sum(-scale, &self.grad_bias.as_view(ctx));
    }
  }

  fn dev_allreduce_load(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {
    let bias_offset = worker.load(offset, &mut self.grad_weights.as_view(ctx).data);
    let end_offset = worker.load(bias_offset, &mut self.grad_bias.as_view(ctx).data);
    assert_eq!(end_offset - offset, self.config.params_len());
  }

  fn dev_allreduce_store(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {
    let bias_offset = worker.store(offset, &mut self.grad_weights.as_view_mut(ctx).data);
    let end_offset = worker.store(bias_offset, &mut self.grad_bias.as_view_mut(ctx).data);
    assert_eq!(end_offset - offset, self.config.params_len());
  }

  /*fn local_allreduce_gradients(&mut self, worker: &mut DeviceAllReduceWorker<f32>, ctx: &DeviceCtxRef) {
    // TODO(20151227)
  }

  fn global_allreduce_gradients(&mut self, worker: &mut DistAllReduceWorker<f32>, ctx: &DeviceCtxRef) {
    // TODO(20151227)
  }*/
}

// FIXME(20160202): residual conv layer.

pub struct ResidualConv2dLayer {
  inner:    Conv2dLayer,

  res_unit: DeviceBuffer<f32>,
  add_res:  CudnnAddOp,
}

impl ResidualConv2dLayer {
  pub fn new(batch_size: usize, config: Conv2dLayerConfig, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> ResidualConv2dLayer {
    assert_eq!(config.get_in_dims(), config.get_out_dims());
    let out_dims = config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;

    let inner = Conv2dLayer::new(batch_size, config, prev_layer, ctx);
    let res_unit = DeviceBuffer::zeros(out_dims.len(), ctx);
    let add_res = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
    );

    ResidualConv2dLayer{
      inner:    inner,
      res_unit: res_unit,
      add_res:  add_res,
    }
  }
}

impl Layer for ResidualConv2dLayer {
  fn config(&self) -> LayerConfig {
    self.inner.config()
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    self.inner.output_activation()
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    self.inner.output_delta()
  }

  fn initialize_params(&mut self, shared_seed: [u64; 2], ctx: &DeviceCtxRef) {
    self.inner.initialize_params(shared_seed, ctx);
  }

  fn initialize_gradients(&mut self, ctx: &DeviceCtxRef) {
    self.inner.initialize_gradients(ctx);
  }

  fn load_params(&mut self, blob: &[u8], ctx: &DeviceCtxRef) -> usize {
    self.inner.load_params(blob, ctx)
  }

  fn save_params(&mut self, blob: &mut Vec<u8>, ctx: &DeviceCtxRef) {
    self.inner.save_params(blob, ctx);
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.inner.batch_limit);
    let Conv2dLayerConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = self.inner.config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = self.inner.config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let in_length = in_dims.len(); //in_width * in_height * in_channels;
    let out_length = out_dims.len();

    let &mut ResidualConv2dLayer{
      ref mut inner,
      ref mut res_unit,
      ref mut add_res,
      .. } = self;
    let &mut Conv2dLayer{
      ref mut in_act, ref mut out_act,
      ref mut work_space,
      ref mut weights, ref mut bias,
      ref mut conv_fwd,
      ref mut add_bias,
      ref config,
      .. } = inner;

    let in_act = in_act.borrow_mut().as_ref(ctx);
    let mut out_act = out_act.borrow_mut().as_ref_mut(ctx);

    conv_fwd.set_batch_size(batch_size).unwrap();
    unsafe { conv_fwd.forward(
        1.0,
        in_act.as_ptr(),
        weights.as_view(ctx).as_ptr(),
        0.0,
        out_act.as_mut_ptr(),
        work_space.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };

    add_bias.set_batch_size(batch_size).unwrap();
    unsafe { add_bias.forward(
        1.0,
        bias.as_view(ctx).as_ptr(),
        1.0,
        out_act.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };

    add_res.set_batch_size(batch_size).unwrap();
    unsafe { add_res.forward(
        1.0,
        in_act.as_ptr(),
        1.0,
        out_act.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };

    match config.act_func {
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

  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.inner.batch_limit);
    let Conv2dLayerConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = self.inner.config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = self.inner.config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let in_length = in_width * in_height * in_channels;
    let out_length = out_dims.len();

    let &mut ResidualConv2dLayer{
      ref mut inner,
      ref mut res_unit,
      ref mut add_res,
      .. } = self;
    let &mut Conv2dLayer{
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights, ref mut bias,
      ref mut grad_weights, ref mut grad_bias,
      ref mut work_space,
      ref mut conv_bwd_w,
      ref mut conv_bwd_d,
      ref mut add_bias,
      ref config,
      .. } = inner;

    let mut in_act = in_act.borrow_mut().as_ref(ctx);
    let mut out_act = out_act.borrow_mut().as_ref(ctx);
    let mut out_delta = out_delta.borrow_mut().as_ref_mut(ctx);
    let mut work_space = work_space.as_ref_mut(ctx);

    match config.act_func {
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

    conv_bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { conv_bwd_w.backward_filter(
        scale,
        in_act.as_ptr(),
        out_delta.as_ptr(),
        1.0,
        grad_weights.as_view_mut(ctx).as_mut_ptr(),
        work_space.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };

    unsafe { conv_bwd_w.backward_bias(
        scale,
        out_delta.as_ptr(),
        1.0,
        grad_bias.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };

    if let &mut Some(ref mut in_delta) = in_delta {
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);

      conv_bwd_d.set_batch_size(batch_size).unwrap();
      unsafe { conv_bwd_d.backward_data(
          1.0,
          weights.as_view(ctx).as_ptr(),
          out_delta.as_ptr(),
          0.0,
          in_delta.as_mut_ptr(),
          work_space.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };

      add_res.set_batch_size(batch_size).unwrap();
      unsafe { add_res.forward(
          1.0,
          res_unit.as_ref(ctx).as_ptr(),
          1.0,
          in_delta.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
    }
  }

  fn descend(&mut self, scale: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef) {
    self.inner.descend(scale, l2_reg_coef, ctx);
  }

  fn reset_gradients(&mut self, scale: f32, ctx: &DeviceCtxRef) {
    self.inner.reset_gradients(scale, ctx);
  }

  fn dev_allreduce_load(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {
    self.inner.dev_allreduce_load(worker, offset, ctx);
  }

  fn dev_allreduce_store(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {
    self.inner.dev_allreduce_store(worker, offset, ctx);
  }
}

// FIXME(20160202): stacked residual conv layer.

pub struct StackedResidualConv2dLayer;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct AffineConv2dLayerConfig {
  pub in_dims:      (usize, usize, usize),
  pub out_channels: usize,
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInitialization,
}

impl AffineConv2dLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    self.in_dims
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    // XXX(20150926): Note that pool layer uses ceil((L+2*pad-size)/stride)
    // whereas conv layer uses floor((L+2*pad-size)/stride).
    let (in_width, in_height, _) = self.in_dims;
    (in_width, in_height, self.out_channels)
  }

  fn params_len(&self) -> usize {
    let (in_width, in_height, in_channels) = self.in_dims;
    let weights_len = in_channels * self.out_channels;
    let bias_len = in_width * in_height * self.out_channels;
    weights_len + bias_len
  }
}

pub struct AffineConv2dLayer {
  batch_limit:  usize,
  config:       AffineConv2dLayerConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights:      DeviceArray2d<f32>,
  bias:         DeviceArray2d<f32>,
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,
  unit_bias:    DeviceArray2d<f32>,

  work_space:   DeviceBuffer<u8>,
  conv_fwd:     CudnnConvFwdOp,
  conv_bwd_w:   CudnnConvBwdFilterOp,
  conv_bwd_d:   CudnnConvBwdDataOp,
  add_bias:     CudnnAddOp,
}

impl AffineConv2dLayer {
  pub fn new(batch_size: usize, config: AffineConv2dLayerConfig, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> AffineConv2dLayer {
    let AffineConv2dLayerConfig{in_dims, .. } = config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    let conv_size = 1;
    let conv_stride = 1;
    let conv_pad = 0;

    let mut work_size = 0;
    let conv_fwd = CudnnConvFwdOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    work_size = max(work_size, conv_fwd.work_size);
    // XXX(20160201): The usual conv bias is unused.
    //let conv_bwd_w = CudnnConvBwdFilterOp::create_deterministic(
    let conv_bwd_w = CudnnConvBwdFilterOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    work_size = max(work_size, conv_bwd_w.work_size);
    //let conv_bwd_d = CudnnConvBwdDataOp::create_deterministic(
    let conv_bwd_d = CudnnConvBwdDataOp::create_fastest(
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    work_size = max(work_size, conv_bwd_d.work_size);

    let add_bias = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
    );

    let mut unit_bias = DeviceArray2d::<f32>::zeros((batch_size, 1), ctx);
    unit_bias.as_view_mut(ctx).set_constant(1.0);

    AffineConv2dLayer{
      batch_limit:  batch_size,
      config:       config,

      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),

      weights:      DeviceArray2d::<f32>::zeros((in_channels, out_channels), ctx),
      bias:         DeviceArray2d::<f32>::zeros((out_length, 1), ctx),
      grad_weights: DeviceArray2d::<f32>::zeros((in_channels, out_channels), ctx),
      grad_bias:    DeviceArray2d::<f32>::zeros((out_length, 1), ctx),
      unit_bias:    unit_bias,

      work_space:   DeviceBuffer::<u8>::zeros(work_size, ctx),
      conv_fwd:     conv_fwd,
      conv_bwd_w:   conv_bwd_w,
      conv_bwd_d:   conv_bwd_d,
      add_bias:     add_bias,
    }
  }
}

impl Layer for AffineConv2dLayer {
  fn config(&self) -> LayerConfig {
    //LayerConfig::AffineConv2d(self.config)
    unimplemented!();
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn initialize_params(&mut self, seed: [u64; 2], ctx: &DeviceCtxRef) {
    let AffineConv2dLayerConfig{in_dims, out_channels, ..} = self.config;
    let (_, _, in_channels) = in_dims;
    let mut rng = Xorshiftplus128Rng::from_seed(seed);
    let mut init_weights = Array2d::zeros((in_channels, out_channels));
    match self.config.init_weights {
      ParamsInitialization::None => {
        panic!("params initialization explicitly disabled!");
      }
      ParamsInitialization::Uniform{half_range} => {
        let dist = Range::new(-half_range as f64, half_range as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInitialization::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      _ => unimplemented!(),
    }
    let init_bias = Array2d::zeros((1, out_channels));
    self.weights.as_view_mut(ctx).sync_load(&init_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&init_bias.as_view());
  }

  fn initialize_gradients(&mut self, ctx: &DeviceCtxRef) {
    self.grad_weights.as_view_mut(ctx).set_constant(0.0);
    self.grad_bias.as_view_mut(ctx).set_constant(0.0);
  }

  fn load_params(&mut self, blob: &[u8], ctx: &DeviceCtxRef) -> usize {
    let AffineConv2dLayerConfig{in_dims, out_channels, ..} = self.config;
    let (_, _, in_channels) = in_dims;
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("Conv2dLayer failed to deserialize weights!");
    let load_bias = Array2d::deserialize(&mut reader)
      .ok().expect("Conv2dLayer failed to deserialize bias!");
    assert_eq!((in_channels, out_channels), load_weights.as_view().bound());
    assert_eq!((1, out_channels), load_bias.as_view().bound());
    self.weights.as_view_mut(ctx).sync_load(&load_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&load_bias.as_view());
    let progress = reader.position() as usize;
    println!("DEBUG: Conv2dLayer: load params: read {}", progress);
    progress
  }

  fn save_params(&mut self, blob: &mut Vec<u8>, ctx: &DeviceCtxRef) {
    let weights = self.weights.as_view(ctx);
    let bias = self.bias.as_view(ctx);
    let mut save_weights = Array2d::zeros(weights.bound());
    let mut save_bias = Array2d::zeros(bias.bound());
    weights.sync_store(&mut save_weights.as_view_mut());
    bias.sync_store(&mut save_bias.as_view_mut());
    save_weights.serialize(blob).unwrap();
    save_bias.serialize(blob).unwrap();
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    let AffineConv2dLayerConfig{in_dims, .. } = self.config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = self.config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let in_length = in_dims.len(); //in_width * in_height * in_channels;
    let out_length = out_dims.len();

    let &mut AffineConv2dLayer{
      ref mut in_act, ref mut out_act,
      ref mut work_space,
      ref mut weights, ref mut bias,
      .. } = self;

    let mut out_act = out_act.borrow_mut().as_ref_mut(ctx);

    self.conv_fwd.set_batch_size(batch_size).unwrap();
    unsafe { self.conv_fwd.forward(
        1.0,
        in_act.borrow_mut().as_ref(ctx).as_ptr(),
        weights.as_view(ctx).as_ptr(),
        0.0,
        out_act.as_mut_ptr(),
        work_space.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
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

  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    let AffineConv2dLayerConfig{in_dims, .. } = self.config;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = self.config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let in_length = in_width * in_height * in_channels;
    let out_length = out_dims.len();

    let &mut AffineConv2dLayer{
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights, ref mut bias,
      ref mut grad_weights, ref mut grad_bias,
      ref mut unit_bias,
      ref mut work_space,
      .. } = self;

    let mut work_space = work_space.as_ref_mut(ctx);
    let mut in_act = in_act.borrow_mut().as_ref(ctx);
    let mut out_act = out_act.borrow_mut().as_ref(ctx);
    let mut out_delta = out_delta.borrow_mut();

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_backprop_inplace(
            out_act.as_ptr(),
            out_length as i32,
            batch_size as i32,
            out_delta.as_ref_mut(ctx).as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }

    self.conv_bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { self.conv_bwd_w.backward_filter(
        scale,
        in_act.as_ptr(),
        out_delta.as_ref(ctx).as_ptr(),
        1.0,
        grad_weights.as_view_mut(ctx).as_mut_ptr(),
        work_space.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    // XXX(20160201): The bias gradient is just the output delta dotted along
    // the batch size dim.
    grad_bias.as_view_mut(ctx).matrix_prod(
        scale,
        &out_delta.as_ref(ctx).into_2d_view((out_length, batch_size)), Transpose::N,
        &unit_bias.as_view(ctx), Transpose::N,
        1.0,
    );
    if let &mut Some(ref mut in_delta) = in_delta {
      self.conv_bwd_d.set_batch_size(batch_size).unwrap();
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);
      unsafe { self.conv_bwd_d.backward_data(
          1.0,
          weights.as_view(ctx).as_ptr(),
          out_delta.as_ref(ctx).as_ptr(),
          0.0,
          in_delta.as_mut_ptr(),
          work_space.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
    }
  }

  fn descend(&mut self, scale: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef) {
    assert!(l2_reg_coef >= 0.0);
    if l2_reg_coef > 0.0 {
      self.grad_weights.as_view_mut(ctx)
        .matrix_sum(l2_reg_coef, &self.weights.as_view(ctx));
      self.grad_bias.as_view_mut(ctx)
        .row_vector_sum(l2_reg_coef, &self.bias.as_view(ctx));
    }
    {
      self.weights.as_view_mut(ctx)
        .matrix_sum(-scale, &self.grad_weights.as_view(ctx));
      self.bias.as_view_mut(ctx)
        .row_vector_sum(-scale, &self.grad_bias.as_view(ctx));
    }
  }

  fn reset_gradients(&mut self, scale: f32, ctx: &DeviceCtxRef) {
    self.grad_weights.as_view_mut(ctx)
      .matrix_scale(scale);
    self.grad_bias.as_view_mut(ctx)
      .row_vector_scale(scale);
  }

  fn dev_allreduce_load(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {
    let bias_offset = worker.load(offset, &mut self.grad_weights.as_view(ctx).data);
    let end_offset = worker.load(bias_offset, &mut self.grad_bias.as_view(ctx).data);
    assert_eq!(end_offset - offset, self.config.params_len());
  }

  fn dev_allreduce_store(&mut self, worker: &mut DeviceAllReduceWorker<f32>, offset: usize, ctx: &DeviceCtxRef) {
    let bias_offset = worker.store(offset, &mut self.grad_weights.as_view_mut(ctx).data);
    let end_offset = worker.store(bias_offset, &mut self.grad_bias.as_view_mut(ctx).data);
    assert_eq!(end_offset - offset, self.config.params_len());
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CategoricalLossLayerConfig {
  pub num_categories:   usize,
}

struct SoftmaxKLLossHvData {
  r_in_act:     SharedDeviceBuf<f32>,
  r_in_delta:   SharedDeviceBuf<f32>,
  r_out_probs:  DeviceArray2d<f32>,
  max_logit:    DeviceArray2d<f32>,
}

pub struct SoftmaxKLLossLayer<HvToggle=Disable<SoftmaxKLLossHvData>>
where HvToggle: Toggle<SoftmaxKLLossHvData> {
  batch_limit:  usize,
  config:       CategoricalLossLayerConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     SharedDeviceBuf<f32>,

  max_logit:    DeviceArray2d<f32>,
  norm_logits:  DeviceArray2d<f32>,
  factors:      DeviceArray2d<f32>,
  sum_factor:   DeviceArray2d<f32>,
  max_prob:     DeviceArray2d<f32>,
  sum_prob:     DeviceArray2d<f32>,
  out_probs:    DeviceArray2d<f32>,
  out_probs_h:  Array2d<f32>,

  hv_data:      HvToggle,

  bytemasks_h:  Vec<u8>,
  bytemasks:    DeviceBuffer<u8>,
  masks:        DeviceBuffer<f32>,

  out_loss1:    DeviceArray2d<f32>,
  out_loss:     DeviceBuffer<f32>,
  out_loss_h:   Vec<f32>,

  pred_cats:    DeviceArray2d<i32>,
  pred_cats_h:  Array2d<i32>,
  true_cats:    DeviceArray2d<i32>,
  true_cats_h:  Array2d<i32>,

  weights:      DeviceArray2d<f32>,
  weights_h:    Array2d<f32>,

  softmax:      CudnnSoftmaxOp,
}

impl SoftmaxKLLossLayer<Disable<SoftmaxKLLossHvData>> {
  pub fn new(batch_size: usize, config: CategoricalLossLayerConfig, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> SoftmaxKLLossLayer {
    let softmax = CudnnSoftmaxOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, batch_size).unwrap(),
    );
    SoftmaxKLLossLayer{
      batch_limit:  batch_size,
      config:       config,

      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta().unwrap(),

      max_logit:    DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      norm_logits:  DeviceArray2d::<f32>::zeros((config.num_categories, batch_size), ctx),
      factors:      DeviceArray2d::<f32>::zeros((config.num_categories, batch_size), ctx),
      sum_factor:   DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      max_prob:     DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      sum_prob:     DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      out_probs:    DeviceArray2d::<f32>::zeros((config.num_categories, batch_size), ctx),
      out_probs_h:  Array2d::zeros((config.num_categories, batch_size)),

      hv_data:      Disable::new(),

      bytemasks_h:  repeat(0).take(config.num_categories * batch_size).collect(),
      bytemasks:    DeviceBuffer::<u8>::zeros(config.num_categories * batch_size, ctx),
      masks:        DeviceBuffer::<f32>::zeros(config.num_categories * batch_size, ctx),

      out_loss1:    DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      out_loss:     DeviceBuffer::<f32>::zeros(1, ctx),
      out_loss_h:   vec![0.0],

      pred_cats:    DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      pred_cats_h:  Array2d::<i32>::zeros((1, batch_size)),
      true_cats:    DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      true_cats_h:  Array2d::<i32>::zeros((1, batch_size)),

      weights:      DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      weights_h:    Array2d::<f32>::zeros((1, batch_size)),

      softmax:      softmax,
    }
  }
}

impl Layer for SoftmaxKLLossLayer<Disable<SoftmaxKLLossHvData>> {
  fn config(&self) -> LayerConfig {
    LayerConfig::SoftmaxKLLoss(self.config)
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);

    self.softmax.set_batch_size(batch_size).unwrap();
    unsafe { self.softmax.forward(
        self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
        self.out_probs.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) }.unwrap();

    // FIXME(20160307)
    /*unimplemented!();
    assert!(batch_size <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_max(
        self.in_act.borrow_mut().as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.max_logit.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_batch_map_subtract_scalar(
        self.in_act.borrow_mut().as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.max_logit.as_view(ctx).as_ptr(),
        self.norm_logits.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_map_exp(
        self.norm_logits.as_view(ctx).as_ptr(),
        (self.config.num_categories * batch_size) as i32,
        self.factors.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_batch_blockreduce_sum(
        self.factors.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.sum_factor.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_batch_softmax_forward(
        self.factors.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.sum_factor.as_view(ctx).as_ptr(),
        self.out_probs.as_view_mut(ctx).as_ptr_mut(),
        ctx.stream.ptr,
    ) };*/
  }

  fn backward(&mut self, batch_size: usize, _loss_scale: f32, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    /*unsafe { rembrandt_kernel_batch_map_softmax_cross_entropy_loss_backprop(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        // XXX(20151218): the minibatch size is now applied during
        // parameter descent.
        1.0,
        ctx.stream.ptr,
    ) };*/
    unsafe { rembrandt_kernel_batch_map_softmax_kl_backward(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.weights.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

impl LossLayer for SoftmaxKLLossLayer<Disable<SoftmaxKLLossHvData>> {
  fn downcast(&self) -> &Layer {
    self
  }

  fn preload_label(&mut self, batch_idx: usize, label: &SampleLabel, phase: Phase) {
    match label {
      &SampleLabel::Category{category} => {
        self.true_cats_h.as_mut_slice()[batch_idx] = category;
      }
      _ => unimplemented!(),
    }
  }

  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.true_cats.as_view_mut(ctx)
      .sync_load(&self.true_cats_h.as_view());
  }

  fn preload_weight(&mut self, batch_idx: usize, weight: f32) {
    self.weights_h.as_mut_slice()[batch_idx] = weight;
  }

  fn load_weights(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.weights.as_view_mut(ctx)
      .sync_load(&self.weights_h.as_view());
  }

  fn store_labels(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    assert!(self.config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_argmax(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.max_prob.as_view_mut(ctx).as_mut_ptr(),
        self.pred_cats.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.pred_cats.as_view(ctx)
      .sync_store(&mut self.pred_cats_h.as_view_mut());
  }

  fn get_labels(&self, batch_size: usize) -> &Array2d<i32> {
    &self.pred_cats_h
  }

  fn count_accuracy(&self, batch_size: usize, phase: Phase) -> (usize, usize) {
    assert!(batch_size <= self.batch_limit);
    let mut num_correct = 0;
    for (&y_truth, &y_hat) in self.true_cats_h.as_slice().iter().zip(self.pred_cats_h.as_slice().iter()).take(batch_size) {
      if y_truth == y_hat {
        num_correct += 1;
      }
    }
    //println!("DEBUG: count accuracy ({}): true: {:?}", batch_size, self.true_cats_h.as_slice());
    //println!("DEBUG: count accuracy ({}): pred: {:?}", batch_size, self.pred_cats_h.as_slice());
    (num_correct, batch_size)
  }

  fn store_probs(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.out_probs.as_view(ctx)
      .sync_store(&mut self.out_probs_h.as_view_mut());
  }

  fn get_probs(&self, batch_size: usize) -> &Array2d<f32> {
    &self.out_probs_h
  }

  fn preload_mask(&mut self, batch_idx: usize, bytemask: &[u8]) {
    assert!(batch_idx < self.batch_limit);
    let size = self.config.num_categories;
    assert_eq!(size, bytemask.len());
    (&mut self.bytemasks_h[batch_idx * size .. (batch_idx + 1) * size])
      .copy_from_slice(bytemask);
  }

  fn expose_host_mask_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    assert!(batch_idx < self.batch_limit);
    let size = self.config.num_categories;
    &mut self.bytemasks_h[batch_idx * size .. (batch_idx + 1) * size]
  }

  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    {
      let bytemasks_h = &self.bytemasks_h;
      let mut bytemasks = self.bytemasks.as_ref_mut(ctx);
      bytemasks.sync_load(bytemasks_h);
    }
    {
      let bytemasks = self.bytemasks.as_ref(ctx);
      let mut masks = self.masks.as_ref_mut(ctx);
      bytemasks.cast_bytes(&mut masks);
    }
  }

  fn apply_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    // XXX(20160211): First zero out the masked probs, then renormalize.
    let len = self.config.num_categories;
    {
      let masks = self.masks.as_ref(ctx);
      let mut out_probs = self.out_probs.as_view_mut(ctx);
      unsafe { rembrandt_kernel_map_multiply_float(
          masks.as_ptr(),
          (len * batch_size) as i32,
          out_probs.as_mut_ptr(),
          ctx.stream.ptr,
      ) };
    }
    {
      let out_probs = self.out_probs.as_view(ctx);
      let mut sum_prob = self.sum_prob.as_view_mut(ctx);
      assert!(len <= 1024);
      unsafe { rembrandt_kernel_batch_blockreduce_sum(
          out_probs.as_ptr(),
          len as i32,
          batch_size as i32,
          sum_prob.as_mut_ptr(),
          0.0,
          ctx.stream.ptr,
      ) };
    }
    {
      let sum_prob = self.sum_prob.as_view(ctx);
      let mut out_probs = self.out_probs.as_view_mut(ctx);
      assert!(len <= 1024);
      unsafe { rembrandt_kernel_batch_blockmap_normalize(
          out_probs.as_mut_ptr(),
          len as i32,
          batch_size as i32,
          sum_prob.as_ptr(),
          ctx.stream.ptr,
      ) };
    }
  }

  fn reset_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.out_loss1.as_view_mut(ctx).set_constant(0.0);
    self.out_loss.as_ref_mut(ctx).set_constant(0.0);
  }

  fn accumulate_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    assert!(self.config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_map_softmax_cross_entropy_loss(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.out_loss1.as_view_mut(ctx).as_mut_ptr(),
        1.0,
        ctx.stream.ptr,
    ) };
    assert!(batch_size <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_sum(
        self.out_loss1.as_view(ctx).as_ptr(),
        batch_size as i32,
        1,
        self.out_loss.as_ref_mut(ctx).as_mut_ptr(),
        1.0,
        ctx.stream.ptr,
    ) };
  }

  fn store_loss(&mut self, ctx: &DeviceCtxRef) {
    self.out_loss.as_ref(ctx)
      .sync_store(&mut self.out_loss_h);
  }

  fn get_loss(&self) -> f32 {
    self.out_loss_h[0]
  }
}

pub struct SoftmaxIndicatorLossLayer {
  batch_limit:  usize,
  config:       CategoricalLossLayerConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     SharedDeviceBuf<f32>,

  max_logit:    DeviceArray2d<f32>,
  norm_logits:  DeviceArray2d<f32>,
  factors:      DeviceArray2d<f32>,
  sum_factor:   DeviceArray2d<f32>,
  max_prob:     DeviceArray2d<f32>,
  sum_prob:     DeviceArray2d<f32>,
  out_probs:    DeviceArray2d<f32>,
  out_probs_h:  Array2d<f32>,

  bytemasks_h:  Vec<u8>,
  bytemasks:    DeviceBuffer<u8>,
  masks:        DeviceBuffer<f32>,

  out_loss1:    DeviceArray2d<f32>,
  out_loss:     DeviceBuffer<f32>,
  out_loss_h:   Vec<f32>,

  pred_cats:    DeviceArray2d<i32>,
  pred_cats_h:  Array2d<i32>,
  true_cats:    DeviceArray2d<i32>,
  true_cats_h:  Array2d<i32>,

  weights:      DeviceArray2d<f32>,
  weights_h:    Array2d<f32>,

  softmax:      CudnnSoftmaxOp,
}

impl SoftmaxIndicatorLossLayer{
  pub fn new(batch_size: usize, config: CategoricalLossLayerConfig, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> SoftmaxIndicatorLossLayer {
    let softmax = CudnnSoftmaxOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, batch_size).unwrap(),
    );
    SoftmaxIndicatorLossLayer{
      batch_limit:  batch_size,
      config:       config,

      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta().unwrap(),

      max_logit:    DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      norm_logits:  DeviceArray2d::<f32>::zeros((config.num_categories, batch_size), ctx),
      factors:      DeviceArray2d::<f32>::zeros((config.num_categories, batch_size), ctx),
      sum_factor:   DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      max_prob:     DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      sum_prob:     DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      out_probs:    DeviceArray2d::<f32>::zeros((config.num_categories, batch_size), ctx),
      out_probs_h:  Array2d::zeros((config.num_categories, batch_size)),

      bytemasks_h:  repeat(0).take(config.num_categories * batch_size).collect(),
      bytemasks:    DeviceBuffer::<u8>::zeros(config.num_categories * batch_size, ctx),
      masks:        DeviceBuffer::<f32>::zeros(config.num_categories * batch_size, ctx),

      out_loss1:    DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      out_loss:     DeviceBuffer::<f32>::zeros(1, ctx),
      out_loss_h:   vec![0.0],

      pred_cats:    DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      pred_cats_h:  Array2d::<i32>::zeros((1, batch_size)),
      true_cats:    DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      true_cats_h:  Array2d::<i32>::zeros((1, batch_size)),

      weights:      DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      weights_h:    Array2d::<f32>::zeros((1, batch_size)),

      softmax:      softmax,
    }
  }
}

impl Layer for SoftmaxIndicatorLossLayer {
  fn config(&self) -> LayerConfig {
    LayerConfig::SoftmaxIndicatorLoss(self.config)
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);

    self.softmax.set_batch_size(batch_size).unwrap();
    unsafe { self.softmax.forward(
        self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
        self.out_probs.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) }.unwrap();
  }

  fn backward(&mut self, batch_size: usize, _loss_scale: f32, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    /*unsafe { rembrandt_kernel_batch_map_softmax_kl_backward(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.weights.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };*/
    unsafe { rembrandt_kernel_batch_map_softmax_ind_backward(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.weights.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

impl LossLayer for SoftmaxIndicatorLossLayer {
  fn downcast(&self) -> &Layer {
    self
  }

  fn preload_label(&mut self, batch_idx: usize, label: &SampleLabel, phase: Phase) {
    match label {
      &SampleLabel::Category{category} => {
        self.true_cats_h.as_mut_slice()[batch_idx] = category;
      }
      _ => unimplemented!(),
    }
  }

  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.true_cats.as_view_mut(ctx)
      .sync_load(&self.true_cats_h.as_view());
  }

  fn preload_weight(&mut self, batch_idx: usize, weight: f32) {
    self.weights_h.as_mut_slice()[batch_idx] = weight;
  }

  fn load_weights(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.weights.as_view_mut(ctx)
      .sync_load(&self.weights_h.as_view());
  }

  fn store_labels(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    assert!(self.config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_argmax(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.max_prob.as_view_mut(ctx).as_mut_ptr(),
        self.pred_cats.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.pred_cats.as_view(ctx)
      .sync_store(&mut self.pred_cats_h.as_view_mut());
  }

  fn get_labels(&self, batch_size: usize) -> &Array2d<i32> {
    &self.pred_cats_h
  }

  fn count_accuracy(&self, batch_size: usize, phase: Phase) -> (usize, usize) {
    assert!(batch_size <= self.batch_limit);
    let mut num_correct = 0;
    for (&y_truth, &y_hat) in self.true_cats_h.as_slice().iter().zip(self.pred_cats_h.as_slice().iter()).take(batch_size) {
      if y_truth == y_hat {
        num_correct += 1;
      }
    }
    //println!("DEBUG: count accuracy ({}): true: {:?}", batch_size, self.true_cats_h.as_slice());
    //println!("DEBUG: count accuracy ({}): pred: {:?}", batch_size, self.pred_cats_h.as_slice());
    (num_correct, batch_size)
  }

  fn store_probs(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.out_probs.as_view(ctx)
      .sync_store(&mut self.out_probs_h.as_view_mut());
  }

  fn get_probs(&self, batch_size: usize) -> &Array2d<f32> {
    &self.out_probs_h
  }

  fn preload_mask(&mut self, batch_idx: usize, bytemask: &[u8]) {
    assert!(batch_idx < self.batch_limit);
    let size = self.config.num_categories;
    assert_eq!(size, bytemask.len());
    (&mut self.bytemasks_h[batch_idx * size .. (batch_idx + 1) * size])
      .copy_from_slice(bytemask);
  }

  fn expose_host_mask_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    assert!(batch_idx < self.batch_limit);
    let size = self.config.num_categories;
    &mut self.bytemasks_h[batch_idx * size .. (batch_idx + 1) * size]
  }

  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    {
      let bytemasks_h = &self.bytemasks_h;
      let mut bytemasks = self.bytemasks.as_ref_mut(ctx);
      bytemasks.sync_load(bytemasks_h);
    }
    {
      let bytemasks = self.bytemasks.as_ref(ctx);
      let mut masks = self.masks.as_ref_mut(ctx);
      bytemasks.cast_bytes(&mut masks);
    }
  }

  fn apply_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    // XXX(20160211): First zero out the masked probs, then renormalize.
    let len = self.config.num_categories;
    {
      let masks = self.masks.as_ref(ctx);
      let mut out_probs = self.out_probs.as_view_mut(ctx);
      unsafe { rembrandt_kernel_map_multiply_float(
          masks.as_ptr(),
          (len * batch_size) as i32,
          out_probs.as_mut_ptr(),
          ctx.stream.ptr,
      ) };
    }
    {
      let out_probs = self.out_probs.as_view(ctx);
      let mut sum_prob = self.sum_prob.as_view_mut(ctx);
      assert!(len <= 1024);
      unsafe { rembrandt_kernel_batch_blockreduce_sum(
          out_probs.as_ptr(),
          len as i32,
          batch_size as i32,
          sum_prob.as_mut_ptr(),
          0.0,
          ctx.stream.ptr,
      ) };
    }
    {
      let sum_prob = self.sum_prob.as_view(ctx);
      let mut out_probs = self.out_probs.as_view_mut(ctx);
      assert!(len <= 1024);
      unsafe { rembrandt_kernel_batch_blockmap_normalize(
          out_probs.as_mut_ptr(),
          len as i32,
          batch_size as i32,
          sum_prob.as_ptr(),
          ctx.stream.ptr,
      ) };
    }
  }

  fn reset_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.out_loss1.as_view_mut(ctx).set_constant(0.0);
    self.out_loss.as_ref_mut(ctx).set_constant(0.0);
  }

  fn accumulate_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    assert!(self.config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_map_softmax_cross_entropy_loss(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.out_loss1.as_view_mut(ctx).as_mut_ptr(),
        1.0,
        ctx.stream.ptr,
    ) };
    assert!(batch_size <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_sum(
        self.out_loss1.as_view(ctx).as_ptr(),
        batch_size as i32,
        1,
        self.out_loss.as_ref_mut(ctx).as_mut_ptr(),
        1.0,
        ctx.stream.ptr,
    ) };
  }

  fn store_loss(&mut self, ctx: &DeviceCtxRef) {
    self.out_loss.as_ref(ctx)
      .sync_store(&mut self.out_loss_h);
  }

  fn get_loss(&self) -> f32 {
    self.out_loss_h[0]
  }
}

// FIXME(20160222)
pub struct LogisticIndicatorLossLayer {
  batch_limit:  usize,
  config:       CategoricalLossLayerConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     SharedDeviceBuf<f32>,

  out_values:   DeviceArray2d<f32>,
  out_values_h: Array2d<f32>,

  value_maxes:  DeviceArray2d<f32>,
  pred_cats:    DeviceArray2d<i32>,
  pred_cats_h:  Array2d<i32>,

  true_cats:    DeviceArray2d<i32>,
  true_cats_h:  Array2d<i32>,

  weights:      DeviceArray2d<f32>,
  weights_h:    Array2d<f32>,
}

impl LogisticIndicatorLossLayer {
  pub fn new(batch_size: usize, config: CategoricalLossLayerConfig, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> LogisticIndicatorLossLayer {
    LogisticIndicatorLossLayer{
      batch_limit:  batch_size,
      config:       config,

      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta().unwrap(),

      out_values:   DeviceArray2d::<f32>::zeros((config.num_categories, batch_size), ctx),
      out_values_h: Array2d::zeros((config.num_categories, batch_size)),

      /*bytemasks_h:  repeat(0).take(config.num_categories * batch_size).collect(),
      bytemasks:    DeviceBuffer::<u8>::zeros(config.num_categories * batch_size, ctx),
      masks:        DeviceBuffer::<f32>::zeros(config.num_categories * batch_size, ctx),*/

      /*out_loss1:    DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      out_loss:     DeviceBuffer::<f32>::zeros(1, ctx),
      out_loss_h:   vec![0.0],*/

      value_maxes:  DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      pred_cats:    DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      pred_cats_h:  Array2d::<i32>::zeros((1, batch_size)),

      true_cats:    DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      true_cats_h:  Array2d::<i32>::zeros((1, batch_size)),

      weights:      DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      weights_h:    Array2d::<f32>::zeros((1, batch_size)),
    }
  }
}

impl Layer for LogisticIndicatorLossLayer {
  fn config(&self) -> LayerConfig {
    LayerConfig::LogisticIndicatorLoss(self.config)
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);

    unsafe { rembrandt_kernel_batch_map_logistic_forward(
        self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.out_values.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }

  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    unsafe { rembrandt_kernel_batch_map_logistic_ind_backward(
        self.out_values.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.weights.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

impl LossLayer for LogisticIndicatorLossLayer {
  fn downcast(&self) -> &Layer {
    self
  }

  fn preload_label(&mut self, batch_idx: usize, label: &SampleLabel, phase: Phase) {
    match label {
      &SampleLabel::Category{category} => {
        self.true_cats_h.as_mut_slice()[batch_idx] = category;
      }
      _ => unimplemented!(),
    }
  }

  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.true_cats.as_view_mut(ctx)
      .sync_load(&self.true_cats_h.as_view());
  }

  fn preload_weight(&mut self, batch_idx: usize, weight: f32) {
    self.weights_h.as_mut_slice()[batch_idx] = weight;
  }

  fn load_weights(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.weights.as_view_mut(ctx)
      .sync_load(&self.weights_h.as_view());
  }

  fn store_labels(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    assert!(self.config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_argmax(
        self.out_values.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.value_maxes.as_view_mut(ctx).as_mut_ptr(),
        self.pred_cats.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.pred_cats.as_view(ctx)
      .sync_store(&mut self.pred_cats_h.as_view_mut());
  }

  fn get_labels(&self, batch_size: usize) -> &Array2d<i32> {
    &self.pred_cats_h
  }

  fn count_accuracy(&self, batch_size: usize, phase: Phase) -> (usize, usize) {
    assert!(batch_size <= self.batch_limit);
    let mut num_correct = 0;
    for (&y_truth, &y_hat) in self.true_cats_h.as_slice().iter().zip(self.pred_cats_h.as_slice().iter()).take(batch_size) {
      if y_truth == y_hat {
        num_correct += 1;
      }
    }
    (num_correct, batch_size)
  }

  fn store_probs(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.out_values.as_view(ctx)
      .sync_store(&mut self.out_values_h.as_view_mut());
  }

  fn get_probs(&self, batch_size: usize) -> &Array2d<f32> {
    &self.out_values_h
  }

  fn preload_mask(&mut self, batch_idx: usize, bytemask: &[u8]) {
    unimplemented!();
  }

  fn expose_host_mask_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    unimplemented!();
  }

  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn apply_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn reset_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    //unimplemented!();
  }

  fn accumulate_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    //unimplemented!();
  }

  fn store_loss(&mut self, ctx: &DeviceCtxRef) {
    //unimplemented!();
  }

  fn get_loss(&self) -> f32 {
    //unimplemented!();
    0.0
  }
}

// FIXME(20160222)
pub struct AntilogisticKLLossLayer {
  batch_limit:  usize,
  config:       CategoricalLossLayerConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     SharedDeviceBuf<f32>,

  logit_sums:   DeviceArray2d<f32>,
  out_values:   DeviceArray2d<f32>,
  out_values_h: Array2d<f32>,

  value_maxes:  DeviceArray2d<f32>,
  pred_cats:    DeviceArray2d<i32>,
  pred_cats_h:  Array2d<i32>,

  true_cats:    DeviceArray2d<i32>,
  true_cats_h:  Array2d<i32>,

  weights:      DeviceArray2d<f32>,
  weights_h:    Array2d<f32>,
}

impl AntilogisticKLLossLayer {
  pub fn new(batch_size: usize, config: CategoricalLossLayerConfig, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> AntilogisticKLLossLayer {
    AntilogisticKLLossLayer{
      batch_limit:  batch_size,
      config:       config,

      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta().unwrap(),

      logit_sums:   DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      out_values:   DeviceArray2d::<f32>::zeros((config.num_categories, batch_size), ctx),
      out_values_h: Array2d::zeros((config.num_categories, batch_size)),

      /*bytemasks_h:  repeat(0).take(config.num_categories * batch_size).collect(),
      bytemasks:    DeviceBuffer::<u8>::zeros(config.num_categories * batch_size, ctx),
      masks:        DeviceBuffer::<f32>::zeros(config.num_categories * batch_size, ctx),*/

      /*out_loss1:    DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      out_loss:     DeviceBuffer::<f32>::zeros(1, ctx),
      out_loss_h:   vec![0.0],*/

      value_maxes:  DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      pred_cats:    DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      pred_cats_h:  Array2d::<i32>::zeros((1, batch_size)),

      true_cats:    DeviceArray2d::<i32>::zeros((1, batch_size), ctx),
      true_cats_h:  Array2d::<i32>::zeros((1, batch_size)),

      weights:      DeviceArray2d::<f32>::zeros((1, batch_size), ctx),
      weights_h:    Array2d::<f32>::zeros((1, batch_size)),
    }
  }
}

impl Layer for AntilogisticKLLossLayer {
  fn config(&self) -> LayerConfig {
    LayerConfig::AntilogisticKLLoss(self.config)
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);

    assert!(self.config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_sum(
        self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.logit_sums.as_view_mut(ctx).as_mut_ptr(),
        0.0,
        ctx.stream.ptr,
    ) };

    unsafe { rembrandt_kernel_batch_map_antilogistic_forward(
        self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.logit_sums.as_view(ctx).as_ptr(),
        self.out_values.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }

  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    unsafe { rembrandt_kernel_batch_map_antilogistic_kl_backward(
        self.out_values.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.weights.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

impl LossLayer for AntilogisticKLLossLayer {
  fn downcast(&self) -> &Layer {
    self
  }

  fn preload_label(&mut self, batch_idx: usize, label: &SampleLabel, phase: Phase) {
    match label {
      &SampleLabel::Category{category} => {
        self.true_cats_h.as_mut_slice()[batch_idx] = category;
      }
      _ => unimplemented!(),
    }
  }

  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.true_cats.as_view_mut(ctx)
      .sync_load(&self.true_cats_h.as_view());
  }

  fn preload_weight(&mut self, batch_idx: usize, weight: f32) {
    self.weights_h.as_mut_slice()[batch_idx] = weight;
  }

  fn load_weights(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.weights.as_view_mut(ctx)
      .sync_load(&self.weights_h.as_view());
  }

  /*fn store_labels(&mut self, batch_idx: usize, phase: Phase, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn get_labels(&self, batch_size: usize) -> &Array2d<i32> {
    unimplemented!();
  }

  fn count_accuracy(&self, batch_size: usize, phase: Phase) -> (usize, usize) {
    unimplemented!();
  }*/

  fn store_labels(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    assert!(self.config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_argmax(
        self.out_values.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.value_maxes.as_view_mut(ctx).as_mut_ptr(),
        self.pred_cats.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.pred_cats.as_view(ctx)
      .sync_store(&mut self.pred_cats_h.as_view_mut());
  }

  fn get_labels(&self, batch_size: usize) -> &Array2d<i32> {
    &self.pred_cats_h
  }

  fn count_accuracy(&self, batch_size: usize, phase: Phase) -> (usize, usize) {
    assert!(batch_size <= self.batch_limit);
    let mut num_correct = 0;
    for (&y_truth, &y_hat) in self.true_cats_h.as_slice().iter().zip(self.pred_cats_h.as_slice().iter()).take(batch_size) {
      if y_truth == y_hat {
        num_correct += 1;
      }
    }
    (num_correct, batch_size)
  }

  fn store_probs(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.out_values.as_view(ctx)
      .sync_store(&mut self.out_values_h.as_view_mut());
  }

  fn get_probs(&self, batch_size: usize) -> &Array2d<f32> {
    &self.out_values_h
  }

  fn preload_mask(&mut self, batch_idx: usize, bytemask: &[u8]) {
    unimplemented!();
  }

  fn expose_host_mask_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    unimplemented!();
  }

  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn apply_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn reset_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    //unimplemented!();
  }

  fn accumulate_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    //unimplemented!();
  }

  fn store_loss(&mut self, ctx: &DeviceCtxRef) {
    //unimplemented!();
  }

  fn get_loss(&self) -> f32 {
    //unimplemented!();
    0.0
  }
}

// FIXME(20160222)
pub struct AntilogisticIndicatorLossLayer {
  batch_limit:  usize,
  config:       CategoricalLossLayerConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     SharedDeviceBuf<f32>,

  logit_sums:   DeviceArray2d<f32>,
  out_values:   DeviceArray2d<f32>,
  out_values_h: Array2d<f32>,

  sel_cats:     DeviceArray2d<i32>,
  sel_cats_h:   Array2d<i32>,
}

impl Layer for AntilogisticIndicatorLossLayer {
  fn config(&self) -> LayerConfig {
    LayerConfig::AntilogisticIndicatorLoss(self.config)
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
  }

  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef) {
  }
}

impl LossLayer for AntilogisticIndicatorLossLayer {
  fn downcast(&self) -> &Layer {
    self
  }

  fn preload_label(&mut self, batch_idx: usize, label: &SampleLabel, phase: Phase) {
    match label {
      &SampleLabel::Category{category} => {
        self.sel_cats_h.as_mut_slice()[batch_idx] = category;
      }
      _ => unimplemented!(),
    }
  }

  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.sel_cats.as_view_mut(ctx)
      .sync_load(&self.sel_cats_h.as_view());
  }

  fn store_labels(&mut self, batch_idx: usize, phase: Phase, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn get_labels(&self, batch_size: usize) -> &Array2d<i32> {
    unimplemented!();
  }

  fn count_accuracy(&self, batch_size: usize, phase: Phase) -> (usize, usize) {
    unimplemented!();
  }

  fn store_probs(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.out_values.as_view(ctx)
      .sync_store(&mut self.out_values_h.as_view_mut());
  }

  fn get_probs(&self, batch_size: usize) -> &Array2d<f32> {
    &self.out_values_h
  }

  fn preload_mask(&mut self, batch_idx: usize, bytemask: &[u8]) {
    unimplemented!();
  }

  fn expose_host_mask_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    unimplemented!();
  }

  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn apply_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn reset_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn accumulate_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn store_loss(&mut self, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn get_loss(&self) -> f32 {
    unimplemented!();
  }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MultiCategoricalLossLayerConfig {
  pub num_categories:   usize,
  pub train_lookahead:  usize,
  pub infer_lookahead:  usize,
}

pub struct MultiSoftmaxKLLossLayer {
  batch_limit:  usize,
  config:       MultiCategoricalLossLayerConfig,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     SharedDeviceBuf<f32>,

  max_prob:     DeviceArray2d<f32>,
  out_probs:    DeviceArray2d<f32>,
  out_probs_h:  Array2d<f32>,
  out_probs1_h: Array2d<f32>,

  pred_cats:    DeviceArray2d<i32>,
  pred_cats_h:  Array2d<i32>,
  true_cats:    DeviceArray2d<i32>,
  true_cats_h:  Array2d<i32>,

  out_loss1:    DeviceArray2d<f32>,
  out_loss:     DeviceBuffer<f32>,
  out_loss_h:   Vec<f32>,

  weights:      DeviceArray2d<f32>,
  weights_h:    Array2d<f32>,

  train_softmax:    CudnnSoftmaxOp,
  infer_softmax:    CudnnSoftmaxOp,
}

impl MultiSoftmaxKLLossLayer {
  pub fn new(batch_size: usize, config: MultiCategoricalLossLayerConfig, prev_layer: Option<&Layer>, ctx: &DeviceCtxRef) -> MultiSoftmaxKLLossLayer {
    // FIXME(20160129): A current limitation.
    assert_eq!(1, config.infer_lookahead);
    let train_lookahead = config.train_lookahead;
    let train_softmax = CudnnSoftmaxOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, train_lookahead * batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, train_lookahead * batch_size).unwrap(),
    );
    let infer_softmax = CudnnSoftmaxOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, batch_size).unwrap(),
    );
    let max_lookahead = max(config.infer_lookahead, config.train_lookahead);
    MultiSoftmaxKLLossLayer{
      batch_limit:  batch_size,
      config:       config,

      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta().unwrap(),

      max_prob:     DeviceArray2d::<f32>::zeros((1, max_lookahead * batch_size), ctx),
      out_probs:    DeviceArray2d::<f32>::zeros((config.num_categories * max_lookahead, batch_size), ctx),
      out_probs_h:  Array2d::zeros((config.num_categories * max_lookahead, batch_size)),
      out_probs1_h: Array2d::zeros((config.num_categories, batch_size)),

      pred_cats:    DeviceArray2d::<i32>::zeros((1, max_lookahead * batch_size), ctx),
      pred_cats_h:  Array2d::<i32>::zeros((1, max_lookahead * batch_size)),
      true_cats:    DeviceArray2d::<i32>::zeros((1, max_lookahead * batch_size), ctx),
      true_cats_h:  Array2d::<i32>::zeros((1, max_lookahead * batch_size)),

      out_loss1:    DeviceArray2d::<f32>::zeros((1, max_lookahead * batch_size), ctx),
      out_loss:     DeviceBuffer::<f32>::zeros(1, ctx),
      out_loss_h:   vec![0.0],

      weights:      DeviceArray2d::<f32>::zeros((1, max_lookahead * batch_size), ctx),
      weights_h:    Array2d::<f32>::zeros((1, max_lookahead * batch_size)),

      train_softmax:    train_softmax,
      infer_softmax:    infer_softmax,
    }
  }
}

impl Layer for MultiSoftmaxKLLossLayer {
  fn config(&self) -> LayerConfig {
    LayerConfig::MultiSoftmaxKLLoss(self.config)
  }

  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    match phase {
      Phase::Inference => {
        // FIXME(20160223): should use a thinner softmax for inference.
        /*assert!(batch_size <= self.batch_limit);
        self.infer_softmax.set_batch_size(batch_size);
        unsafe { self.infer_softmax.forward(
            self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
            self.out_probs.as_view_mut(ctx).as_mut_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();*/
        assert!(batch_size <= self.batch_limit);
        let lookahead = self.config.train_lookahead;
        self.train_softmax.set_batch_size(lookahead * batch_size).unwrap();
        unsafe { self.train_softmax.forward(
            self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
            self.out_probs.as_view_mut(ctx).as_mut_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
      Phase::Training => {
        assert!(batch_size <= self.batch_limit);
        let lookahead = self.config.train_lookahead;
        self.train_softmax.set_batch_size(lookahead * batch_size).unwrap();
        unsafe { self.train_softmax.forward(
            self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
            self.out_probs.as_view_mut(ctx).as_mut_ptr(),
            &*ctx.get_dnn(),
        ) }.unwrap();
      }
    }
  }

  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    /*unsafe { rembrandt_kernel_batch_map_softmax_cross_entropy_loss_backprop(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        (self.config.train_lookahead * batch_size) as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        // XXX(20151218): the minibatch size is now applied during
        // parameter descent.
        1.0,
        ctx.stream.ptr,
    ) };*/
    unsafe { rembrandt_kernel_batch_map_softmax_kl_backward(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        (self.config.train_lookahead * batch_size) as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.weights.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

impl LossLayer for MultiSoftmaxKLLossLayer {
  fn downcast(&self) -> &Layer {
    self
  }

  fn preload_label(&mut self, batch_idx: usize, label: &SampleLabel, phase: Phase) {
    match label {
      &SampleLabel::Category{category} => {
        match phase {
          Phase::Inference => {
            //self.true_cats_h.as_mut_slice()[batch_idx] = category;
            let lookahead = self.config.train_lookahead;
            self.true_cats_h.as_mut_slice()[batch_idx * lookahead] = category;
          }
          Phase::Training => {
            unimplemented!();
          }
        }
      }
      &SampleLabel::MultiCategory{ref categories} => {
        match phase {
          Phase::Training => {
            let lookahead = self.config.train_lookahead;
            assert_eq!(lookahead, categories.len());
            for k in 0 .. lookahead {
              // FIXME(20160129): correct convention.
              self.true_cats_h.as_mut_slice()[k + batch_idx * lookahead] = categories[k];
            }
          }
          Phase::Inference => {
            unimplemented!();
          }
        }
      }
    }
  }

  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.true_cats.as_view_mut(ctx)
      .sync_load(&self.true_cats_h.as_view());
  }

  fn preload_weight(&mut self, batch_idx: usize, weight: f32) {
    assert!(batch_idx < self.batch_limit);
    let lookahead = self.config.train_lookahead;
    for k in 0 .. lookahead {
      self.weights_h.as_mut_slice()[k + batch_idx * lookahead] = weight;
    }
  }

  fn load_weights(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    self.weights.as_view_mut(ctx)
      .sync_load(&self.weights_h.as_view());
  }

  fn store_labels(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    match phase {
      Phase::Inference => {
        assert!(batch_size <= self.batch_limit);
        assert!(self.config.num_categories <= 1024);
        unsafe { rembrandt_kernel_batch_blockreduce_argmax(
            self.out_probs.as_view(ctx).as_ptr(),
            self.config.num_categories as i32,
            //batch_size as i32,
            (self.config.train_lookahead * batch_size) as i32,
            self.max_prob.as_view_mut(ctx).as_mut_ptr(),
            self.pred_cats.as_view_mut(ctx).as_mut_ptr(),
            ctx.stream.ptr,
        ) };
        self.pred_cats.as_view(ctx)
          .sync_store(&mut self.pred_cats_h.as_view_mut());
      }
      Phase::Training => {
        assert!(batch_size <= self.batch_limit);
        assert!(self.config.num_categories <= 1024);
        unsafe { rembrandt_kernel_batch_blockreduce_argmax(
            self.out_probs.as_view(ctx).as_ptr(),
            self.config.num_categories as i32,
            (self.config.train_lookahead * batch_size) as i32,
            self.max_prob.as_view_mut(ctx).as_mut_ptr(),
            self.pred_cats.as_view_mut(ctx).as_mut_ptr(),
            ctx.stream.ptr,
        ) };
        self.pred_cats.as_view(ctx)
          .sync_store(&mut self.pred_cats_h.as_view_mut());
      }
    }
  }

  fn get_labels(&self, batch_size: usize) -> &Array2d<i32> {
    &self.pred_cats_h
  }

  fn count_accuracy(&self, batch_size: usize, phase: Phase) -> (usize, usize) {
    match phase {
      Phase::Inference => {
        assert!(batch_size <= self.batch_limit);
        let lookahead = self.config.train_lookahead;
        let mut num_correct = 0;
        for (i, (&y_truth, &y_hat)) in self.true_cats_h.as_slice().iter()
            .zip(self.pred_cats_h.as_slice().iter())
            //.take(batch_size)
            .take(lookahead * batch_size)
            .enumerate()
        {
          if i % lookahead == 0 {
            if y_truth == y_hat {
              num_correct += 1;
            }
          }
        }
        (num_correct, batch_size)
      }
      Phase::Training => {
        assert!(batch_size <= self.batch_limit);
        let lookahead = self.config.train_lookahead;
        let mut num_correct = 0;
        for (i, (&y_truth, &y_hat)) in self.true_cats_h.as_slice().iter()
            .zip(self.pred_cats_h.as_slice().iter())
            //.take(batch_size)
            .take(lookahead * batch_size)
            .enumerate()
        {
          if i % lookahead == 0 {
            if y_truth == y_hat {
              num_correct += 1;
            }
          }
        }
        (num_correct, batch_size)
      }
    }
  }

  fn store_probs(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.out_probs.as_view(ctx)
      .sync_store(&mut self.out_probs_h.as_view_mut());
  }

  fn get_probs(&self, batch_size: usize) -> &Array2d<f32> {
    &self.out_probs_h
  }

  fn preload_mask(&mut self, batch_idx: usize, bytemask: &[u8]) {
    // FIXME(20160211)
    unimplemented!();
  }

  fn expose_host_mask_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    // FIXME(20160211)
    unimplemented!();
  }

  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    // FIXME(20160211)
    unimplemented!();
  }

  fn apply_masks(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    // FIXME(20160211)
    unimplemented!();
  }

  fn reset_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.out_loss1.as_view_mut(ctx).set_constant(0.0);
    self.out_loss.as_ref_mut(ctx).set_constant(0.0);
  }

  fn accumulate_loss(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    assert!(batch_size <= self.batch_limit);
    assert!(self.config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_map_softmax_cross_entropy_loss(
        self.out_probs.as_view(ctx).as_ptr(),
        self.config.num_categories as i32,
        (self.config.train_lookahead * batch_size) as i32,
        self.true_cats.as_view(ctx).as_ptr(),
        self.out_loss1.as_view_mut(ctx).as_mut_ptr(),
        1.0,
        ctx.stream.ptr,
    ) };
    assert!(self.config.train_lookahead * batch_size <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_sum(
        self.out_loss1.as_view(ctx).as_ptr(),
        (self.config.train_lookahead * batch_size) as i32,
        1,
        self.out_loss.as_ref_mut(ctx).as_mut_ptr(),
        1.0,
        ctx.stream.ptr,
    ) };
  }

  fn store_loss(&mut self, ctx: &DeviceCtxRef) {
    self.out_loss.as_ref(ctx)
      .sync_store(&mut self.out_loss_h);
  }

  fn get_loss(&self) -> f32 {
    self.out_loss_h[0]
  }
}
