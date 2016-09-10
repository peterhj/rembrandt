//use operator::{SharedDeviceBuf};
use operator::{
  Operator, OpRead, OpWrite,
  ActivationFunction,
  ParamsInit,
  Regularization,
  OpCapability,
  OpPhase,
  SharedDeviceBuf,
};

use array::{
  Array, AsyncArray, ArrayView, ArrayViewMut, ArrayZeroExt, NdArraySerialize,
  Shape, Array2d, Array3d,
};
use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::linalg::{VectorExt, BlasMatrixExt, BlasVectorExt, Transpose};
use array_cuda::device::memory::{DeviceBufferInitExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut};
use array_cuda::device::num::{CastBytesExt, NumExt};
use array_cuda::device::random::{RandomSampleExt, UniformDist, GaussianDist};
use cuda_dnn::v5::{
  CudnnConvFwdOp, CudnnConvBwdFilterOp, CudnnConvBwdDataOp,
  CudnnAddOp, CudnnActKind, CudnnActOp, CudnnSoftmaxOp, CudnnPoolingOp, CudnnTransformOp,
  CudnnTensorDesc, CudnnFilterDesc, CudnnConvDesc,
};
use cuda_dnn::v5::ffi::{cudnnConvolutionFwdAlgo_t, cudnnPoolingMode_t};
use rembrandt_kernels::*;
use rembrandt_kernels::ffi::*;
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, SeedableRng, thread_rng};
use std::cell::{RefCell, Ref, RefMut};
use std::cmp::{max, min};
use std::collections::{Bound, BTreeMap};
use std::io::{Cursor};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};

pub enum InterpolateFilter {
  Nearest,
  Linear,
  BSpline,
  CatmullRom,
  MitchellNetravali,
  Lanczos2,
}

pub struct Interpolate2dOperatorConfig {
  pub in_dims:  (usize, usize, usize),
  pub out_dims: (usize, usize, usize),
  pub filter:   InterpolateFilter,
}

pub struct Interpolate2dOperator {
  batch_cap:    usize,
  config:       Interpolate2dOperatorConfig,
}

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

  //backward:     Option<Pool2dBwdOperator>,
  r_forward:    Option<Pool2dRFwdOperator>,
}

struct Pool2dBwdOperator {
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_delta:    SharedDeviceBuf<f32>,
}

struct Pool2dRFwdOperator {
  in_r_act:     SharedDeviceBuf<f32>,
  out_r_act:    SharedDeviceBuf<f32>,
}

impl Pool2dOperator {
  pub fn new(batch_size: usize, capability: OpCapability, config: Pool2dOperatorConfig, prev_op: Option<&Operator>, context: Rc<DeviceContext>) -> Pool2dOperator {
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
    let r_forward = if capability.r_forward_enabled() {
      Some(Pool2dRFwdOperator{
        in_r_act:   prev_op.unwrap().get_output_r_act(0).unwrap(),
        out_r_act:  Rc::new(RefCell::new(DeviceBuffer::zeros(out_len * batch_size, ctx))),
      })
    } else {
      None
    };
    Pool2dOperator{
      batch_cap:    batch_size,
      config:       config,
      context:      context.clone(),
      in_act:       prev_op.unwrap().get_output_act(0),
      in_delta:     prev_op.unwrap().get_output_delta(0),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_len * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_len * batch_size, ctx))),
      pooling:      pooling,
      r_forward:    r_forward,
    }
  }
}

impl Operator for Pool2dOperator {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  /*fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }*/

  fn get_output_act(&self, _arm: usize) -> SharedDeviceBuf<f32> {
    assert_eq!(0, _arm);
    self.out_act.clone()
  }

  fn get_output_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    assert_eq!(0, _arm);
    Some(self.out_delta.clone())
  }

  fn get_output_r_act(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    assert_eq!(0, _arm);
    self.r_forward.as_ref().map(|r_fwd| r_fwd.out_r_act.clone())
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

  fn r_forward(&mut self, batch_size: usize) {
    assert!(self.r_forward.is_some());
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    let mut r_forward = self.r_forward.as_mut().unwrap();
    self.pooling.set_batch_size(batch_size).unwrap();
    unsafe { self.pooling.forward(
        r_forward.in_r_act.borrow_mut().as_ref(ctx).as_ptr(),
        r_forward.out_r_act.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) }.unwrap();
  }
}
