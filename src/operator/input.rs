use data_new::{SampleLabel};
use operator::{
  SharedDeviceBuf,
  Data3dPreproc,
  Data3dPreprocOperator,
};
use operator::comm::{CommWorker};
use operator::conv::{
  StackResConv2dOperatorConfig,
  StackResConv2dOperator,
  ProjStackResConv2dOperatorConfig,
  ProjStackResConv2dOperator,
};
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
use std::io::{Cursor};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::rc::{Rc};

#[derive(Clone)]
pub struct VarData3dOperatorConfig {
  pub max_in_dims:  (usize, usize, usize),
  pub out_dims:     (usize, usize, usize),
  pub normalize:    bool,
  pub preprocs:     Vec<Data3dPreproc>,
}

pub struct VarData3dOperator {
  batch_cap:    usize,
  config:       VarData3dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_buf_h:     Vec<u8>,
  in_buf:       DeviceBuffer<u8>,
  tmp_buf:      DeviceBuffer<f32>,
  out_buf:      SharedDeviceBuf<f32>,

  rng:          Xorshiftplus128Rng,
  preprocs:     Vec<Data3dPreprocOperator>,
}

impl VarData3dOperator {
  pub fn new(batch_size: usize, config: VarData3dOperatorConfig, context: Rc<DeviceContext>) -> VarData3dOperator {
    let ctx = &(*context).as_ref();
    let in_dims = config.max_in_dims;
    let (in_width, in_height, in_channels) = in_dims;
    let in_frame_len = in_dims.len();
    let out_dims = config.out_dims;
    let out_frame_len = out_dims.len();
    let max_frame_len = max(in_frame_len, out_frame_len);
    // FIXME(20160421)
    unimplemented!();
  }
}
