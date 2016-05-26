use data_new::{SampleLabel};
use operator::{
  Operator,
  ActivationFunction,
  ParamsInit,
  Regularization,
  OpCapability,
  OpPhase,
  SharedDeviceBuf,
};
use operator::comm::{CommWorker};

use array::{
  Array, AsyncArray, ArrayView, ArrayViewMut, ArrayZeroExt, NdArraySerialize,
  Shape, Array2d,
};
use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::ext::{DeviceCastBytesExt, DeviceNumExt};
use array_cuda::device::linalg::{BlasMatrixExt, BlasVectorExt, Transpose};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer};
use array_cuda::device::random::{RandomSampleExt, UniformDist};
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
pub enum AffineBackend {
  CublasGemm,
}

#[derive(Clone, Copy, Debug)]
pub struct AffineOperatorConfig {
  pub in_channels:  usize,
  pub out_channels: usize,
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub backend:      AffineBackend,
}

impl AffineOperatorConfig {
  pub fn params_len(&self) -> usize {
    let weights_len = self.in_channels * self.out_channels;
    let bias_len = self.out_channels;
    weights_len + bias_len
  }
}

pub struct AffineOperator {
  batch_cap:    usize,
  _capability:  OpCapability,
  params_off:   usize,
  config:       AffineOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights:      DeviceArray2d<f32>,
  bias:         DeviceArray2d<f32>,

  add_bias:     CudnnAddOp,

  backward:     Option<AffineBwdOperator>,
  hv_backward:  Option<AffineHvBwdOperator>,
}

struct AffineBwdOperator {
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,
  acc_grad_weights: DeviceArray2d<f32>,
  acc_grad_bias:    DeviceArray2d<f32>,
  save_weights: DeviceArray2d<f32>,
  save_bias:    DeviceArray2d<f32>,

  unit_bias:    DeviceArray2d<f32>,

  //comm_worker:  Rc<RefCell>,
}

struct AffineHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
  dir_bias:     DeviceArray2d<f32>,
}

impl AffineOperator {
  pub fn new(batch_size: usize, capability: OpCapability, params_offset: usize, config: AffineOperatorConfig, prev_op: Option<&Operator>, /*comm_worker: Option<Rc<RefCell<Comm>>>,*/ context: Rc<DeviceContext>) -> AffineOperator {
    let in_channels = config.in_channels;
    let out_channels = config.out_channels;

    let ctx = &(*context).as_ref();

    let backward = if capability.backward_enabled() {
      let mut unit_bias = DeviceArray2d::<f32>::zeros((1, batch_size), ctx);
      unit_bias.as_view_mut(ctx).set_constant(1.0);
      Some(AffineBwdOperator{
        grad_weights: DeviceArray2d::<f32>::zeros((in_channels, out_channels), ctx),
        grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        acc_grad_weights: DeviceArray2d::<f32>::zeros((in_channels, out_channels), ctx),
        acc_grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        save_weights: DeviceArray2d::<f32>::zeros((in_channels, out_channels), ctx),
        save_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        unit_bias:    unit_bias,
        //comm_worker:  comm_worker.unwrap(),
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
      params_off:   params_offset,
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

impl Operator for AffineOperator {
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
    let AffineOperatorConfig{in_channels, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    let mut init_weights = Array2d::zeros((in_channels, out_channels));
    match self.config.init_weights {
      ParamsInit::Disabled => {
        panic!("AffineOperator: params init explicitly disabled");
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
        let in_conns = self.config.in_channels;
        let out_conns = self.config.out_channels;
        let half_range = (6.0 / (in_conns + out_conns) as f64).sqrt();
        let dist = Range::new(-half_range, half_range);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInit::KaimingFwd => {
        let in_conns = self.config.in_channels;
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

  fn decode_params(&mut self, blob: &[u8]) -> usize {
    let AffineOperatorConfig{in_channels, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("AffineOperator failed to deserialize weights!");
    let load_bias = Array2d::deserialize(&mut reader)
      .ok().expect("AffineOperator failed to deserialize bias!");
    assert_eq!((in_channels, out_channels), load_weights.as_view().bound());
    assert_eq!((1, out_channels), load_bias.as_view().bound());
    self.weights.as_view_mut(ctx).sync_load(&load_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&load_bias.as_view());
    let progress = reader.position() as usize;
    progress
  }

  fn encode_params(&mut self, blob: &mut Vec<u8>) {
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

  fn decode_state(&mut self, blob: &[u8]) -> usize {
    assert!(self.backward.is_some());
    let AffineOperatorConfig{in_channels, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    let mut reader = Cursor::new(blob);

    let load_update_weights = Array2d::deserialize(&mut reader)
      .ok().expect("AffineOperator failed to deserialize weights!");
    let load_update_bias = Array2d::deserialize(&mut reader)
      .ok().expect("AffineOperator failed to deserialize bias!");
    assert_eq!((in_channels, out_channels), load_update_weights.as_view().bound());
    assert_eq!((1, out_channels), load_update_bias.as_view().bound());

    backward.acc_grad_weights.as_view_mut(ctx).sync_load(&load_update_weights.as_view());
    backward.acc_grad_bias.as_view_mut(ctx).sync_load(&load_update_bias.as_view());

    let progress = reader.position() as usize;
    progress
  }

  fn encode_state(&mut self, blob: &mut Vec<u8>) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();

    let update_weights = backward.acc_grad_weights.as_view(ctx);
    let update_bias = backward.acc_grad_bias.as_view(ctx);

    let mut save_update_weights = Array2d::zeros(update_weights.bound());
    let mut save_update_bias = Array2d::zeros(update_bias.bound());

    update_weights.sync_store(&mut save_update_weights.as_view_mut());
    update_bias.sync_store(&mut save_update_bias.as_view_mut());

    save_update_weights.serialize(blob).unwrap();
    save_update_bias.serialize(blob).unwrap();
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

  //fn update_params(&mut self, momentum: f32, nesterov: bool) {
  fn update_params(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    self.weights.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights.as_view(ctx));
    self.bias.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.acc_grad_bias.as_view(ctx));
  }

  fn update_params2(&mut self, grad_scale: f32, update_scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    if grad_scale != 0.0 {
      self.weights.as_view_mut(ctx)
        .matrix_sum(grad_scale, &backward.grad_weights.as_view(ctx));
      self.bias.as_view_mut(ctx)
        .row_vector_sum(grad_scale, &backward.grad_bias.as_view(ctx));
    }
    if update_scale != 0.0 {
      self.weights.as_view_mut(ctx)
        .matrix_sum(update_scale, &backward.acc_grad_weights.as_view(ctx));
      self.bias.as_view_mut(ctx)
        .row_vector_sum(update_scale, &backward.acc_grad_bias.as_view(ctx));
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

  /*fn set_grads_with_params_diff(&mut self) {
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
  }*/

  /*fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.load(self.params_off, &mut self.weights); //, ctx);
    comm_worker.load(self.params_off, &mut self.bias); //, ctx);
  }

  fn sync_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights); //, ctx);
    comm_worker.store(self.params_off, &mut self.bias); //, ctx);
  }*/

  fn stage_grads(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    assert!(self.backward.is_some());
    let mut backward = self.backward.as_mut().unwrap();
    let mut offset = offset;
    comm_worker.load(offset, &mut backward.grad_weights);
    offset += backward.grad_weights.len();
    comm_worker.load(offset, &mut backward.grad_bias);
    offset += backward.grad_bias.len();
    self.config.params_len()
  }

  fn merge_grads(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    assert!(self.backward.is_some());
    let mut backward = self.backward.as_mut().unwrap();
    let mut offset = offset;
    comm_worker.store(offset, &mut backward.grad_weights);
    offset += backward.grad_weights.len();
    comm_worker.store(offset, &mut backward.grad_bias);
    offset += backward.grad_bias.len();
    self.config.params_len()
  }

  fn stage_params(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    let mut offset = offset;
    comm_worker.load(offset, &mut self.weights);
    offset += self.weights.len();
    comm_worker.load(offset, &mut self.bias);
    offset += self.bias.len();
    self.config.params_len()
  }

  fn merge_params(&mut self, offset: usize, comm_worker: &mut CommWorker) -> usize {
    let mut offset = offset;
    comm_worker.store(offset, &mut self.weights);
    offset += self.weights.len();
    comm_worker.store(offset, &mut self.bias);
    offset += self.bias.len();
    self.config.params_len()
  }

  /*fn reset_grads(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.grad_weights.as_view_mut(ctx)
      .matrix_scale(scale);
    backward.grad_bias.as_view_mut(ctx)
      .row_vector_scale(scale);
  }*/

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
