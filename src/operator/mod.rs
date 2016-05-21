use data_new::{SampleLabel};
use operator::comm::{CommWorker};
use operator::conv::{
  BNormConv2dOperatorConfig,
  BNormConv2dOperator,
  StackResConv2dOperatorConfig,
  StackResConv2dOperator,
  ProjStackResConv2dOperatorConfig,
  ProjStackResConv2dOperator,
};
use operator::input::{
  VarData3dOperatorConfig,
  VarData3dOperator,
};
use operator::loss::{
  //LossOperator,
  CategoricalLossConfig,
  SoftmaxKLLossOperator,
};

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::ext::{DeviceCastBytesExt, DeviceNumExt};
use array_cuda::device::linalg::{BlasMatrixExt, BlasVectorExt, Transpose};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut};
use array_cuda::device::random::{RandomSampleExt, UniformDist, GaussianDist};
use array_new::{
  Array, AsyncArray, ArrayView, ArrayViewMut, ArrayZeroExt, NdArraySerialize,
  Shape, Array2d, Array3d,
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
use std::fs::{File};
use std::io::{Cursor};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::path::{PathBuf};
use std::rc::{Rc};

pub mod comm;
pub mod conv;
pub mod data;
pub mod input;
pub mod loss;
pub mod worker;

pub trait OpRead {
  fn read<'ctx>(&mut self, offset: usize, dst: &mut DeviceBufferRefMut<'ctx, f32>) -> usize;
}

pub trait OpWrite {
  fn write<'ctx>(&mut self, offset: usize, src: &mut DeviceBufferRef<'ctx, f32>) -> usize;
}

pub trait Operator {
  fn batch_size(&self) -> usize;
  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> { None }
  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> { None }
  fn init_params(&mut self, _shared_seed: [u64; 2]) {}
  fn decode_params(&mut self, _blob: &[u8]) -> usize { 0 }
  fn encode_params(&mut self, _blob: &mut Vec<u8>) {}
  fn decode_state(&mut self, _blob: &[u8]) -> usize { 0 }
  fn encode_state(&mut self, _blob: &mut Vec<u8>) {}
  fn read_params(&mut self, _offset: usize, _reader: ()) -> usize { 0 }
  fn write_params(&mut self, _offset: usize, _writer: ()) -> usize { 0 }
  fn forward(&mut self, batch_size: usize, phase: OpPhase);

  // Requires `Backward` capability.
  fn reset(&mut self) {}
  fn read_grads(&mut self, _offset: usize, _reader: ()) -> usize { 0 }
  fn write_grads(&mut self, _offset: usize, _writer: ()) -> usize { 0 }
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

  // Requires `HVBackward` capability.
  fn read_direction(&mut self, _offset: usize, _reader: ()) -> usize { 0 }
  fn write_direction(&mut self, _offset: usize, _writer: ()) -> usize { 0 }
  fn r_forward(&mut self, _batch_size: usize) { unimplemented!(); }
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
  fn store_loss(&mut self, batch_size: usize) -> f32;
  fn store_output_values(&mut self, batch_size: usize);
  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32>;
  fn store_output_categories(&mut self, batch_size: usize);
  fn get_output_categories(&self, batch_size: usize) -> &Array2d<i32>;
  fn accuracy_count(&self, batch_size: usize) -> usize;
  //fn reset_loss(&mut self);

  // Requires `HVBackward` capability.
  fn hv_stage_hessian_weight(&mut self, _batch_idx: usize, _h_weight: f32) { unimplemented!(); }
  fn hv_load_hessian_weights(&mut self, _batch_size: usize) { unimplemented!(); }
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
  HVBackward,
  GNHVBackward,
}

impl OpCapability {
  pub fn backward_enabled(&self) -> bool {
    match *self {
      OpCapability::Forward         => false,
      OpCapability::Backward        => true,
      OpCapability::HVBackward      => true,
      OpCapability::GNHVBackward    => true,
    }
  }

  pub fn hv_backward_enabled(&self) -> bool {
    match *self {
      OpCapability::Forward         => false,
      OpCapability::Backward        => false,
      OpCapability::HVBackward      => true,
      OpCapability::GNHVBackward    => true,
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

  /*fn sync_grads(&mut self) {
    // Do nothing.
  }

  fn sync_params(&mut self) {
    // Do nothing.
  }

  fn reset_grads(&mut self, _scale: f32) {
    // Do nothing.
  }*/
}

pub struct JoinOperator;

#[derive(Clone, Debug)]
pub enum Data3dPreproc {
  AddPixelwiseColorNoise{
    brightness_hrange:  f32,
    contrast_hrange:    f32,
    saturation_hrange:  f32,
  },
  AddPixelwisePCALightingNoise{
    //sing_vectors_path:  PathBuf,
    //sing_values_path:   PathBuf,
    singular_vecs:  Vec<Vec<f32>>,
    singular_vals:  Vec<f32>,
    /*singular_vecs:  vec![
      vec![-0.5675,  0.7192,  0.4009],
      vec![-0.5808, -0.0045, -0.8140],
      vec![-0.5836, -0.6948,  0.4203],
    ],
    singular_vals:  vec![
      0.2175, 0.0188, 0.0045,
    ],*/
    std_dev:    f32,
  },
  Crop{
    crop_width:     usize,
    crop_height:    usize,
  },
  FlipX,
  NormalizePixelwise{
    mean:       Vec<f32>,
    std_dev:    Vec<f32>,
    /*mean:       vec![0.485, 0.456, 0.406],
    std_dev:    vec![0.229, 0.224, 0.225],*/
  },
  SubtractElemwiseMean{
    mean_path:  PathBuf,
    //normalize:  bool,
  },
}

#[derive(Clone, Debug)]
pub struct Data3dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub normalize:    bool,
  pub preprocs:     Vec<Data3dPreproc>,
}

impl Data3dOperatorConfig {
  pub fn get_out_dims(&self) -> (usize, usize, usize) {
    let mut out_dims = self.in_dims;
    for preproc in self.preprocs.iter() {
      match preproc {
        &Data3dPreproc::AddPixelwiseColorNoise{..} => {
          // Do nothing.
        }
        &Data3dPreproc::AddPixelwisePCALightingNoise{..} => {
          // Do nothing.
        }
        &Data3dPreproc::Crop{crop_width, crop_height} => {
          assert!(crop_width <= out_dims.0);
          assert!(crop_height <= out_dims.1);
          out_dims = (crop_width, crop_height, out_dims.2);
          assert!(self.in_dims.len() >= out_dims.len());
        }
        &Data3dPreproc::FlipX => {
          // Do nothing.
        }
        &Data3dPreproc::NormalizePixelwise{..} => {
          // Do nothing.
        }
        &Data3dPreproc::SubtractElemwiseMean{..} => {
          // Do nothing.
        }
      }
    }
    out_dims
  }
}

pub enum Data3dPreprocOperator {
  AddPixelwiseColorNoise,
  AddPixelwisePCALightingNoise{
    dist:       GaussianDist<f32>,
    svecs_buf:  DeviceBuffer<f32>,
    svals_buf:  DeviceBuffer<f32>,
    alphas_buf: DeviceBuffer<f32>,
  },
  Crop{
    transform:  CudnnTransformOp,
    woff_range: Range<usize>,
    hoff_range: Range<usize>,
  },
  FlipX{
    coin_flip:  Range<usize>,
  },
  NormalizePixelwise{
    mean_buf:       DeviceBuffer<f32>,
    std_dev_buf:    DeviceBuffer<f32>,
  },
  SubtractElemwiseMean{
    add:        CudnnAddOp,
    mean_arr_h: Array3d<f32>,
    mean_array: DeviceBuffer<f32>,
  },
}

pub struct Data3dOperator {
  batch_cap:    usize,
  config:       Data3dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_buf_h:     Vec<u8>,
  in_buf:       DeviceBuffer<u8>,
  tmp_buf:      DeviceBuffer<f32>,
  out_buf:      SharedDeviceBuf<f32>,

  rng:          Xorshiftplus128Rng,
  preprocs:     Vec<Data3dPreprocOperator>,
}

impl Data3dOperator {
  pub fn new(batch_size: usize, config: Data3dOperatorConfig, context: Rc<DeviceContext>) -> Data3dOperator {
    let ctx = &(*context).as_ref();
    let in_dims = config.in_dims;
    let (in_width, in_height, in_channels) = in_dims;
    let in_frame_len = in_dims.len();
    let out_dims = config.get_out_dims();
    let out_frame_len = out_dims.len();
    let mut preprocs = Vec::with_capacity(config.preprocs.len());
    for preproc_cfg in config.preprocs.iter() {
      match preproc_cfg {
        &Data3dPreproc::AddPixelwiseColorNoise{..} => {
          // FIXME(20160422)
          unimplemented!();
        }
        &Data3dPreproc::AddPixelwisePCALightingNoise{ref singular_vecs, ref singular_vals, std_dev} => {
          // FIXME(20160422)
          let mut svecs_buf = DeviceBuffer::zeros(9, ctx);
          let svecs_buf_h = vec![
            singular_vecs[0][0],
            singular_vecs[0][1],
            singular_vecs[0][2],
            singular_vecs[1][0],
            singular_vecs[1][1],
            singular_vecs[1][2],
            singular_vecs[2][0],
            singular_vecs[2][1],
            singular_vecs[2][2],
          ];
          svecs_buf.as_ref_mut(ctx).sync_load(&svecs_buf_h);
          let mut svals_buf = DeviceBuffer::zeros(3, ctx);
          svals_buf.as_ref_mut(ctx).sync_load(&singular_vals);
          let alphas_buf = DeviceBuffer::zeros(3 * batch_size, ctx);
          preprocs.push(Data3dPreprocOperator::AddPixelwisePCALightingNoise{
            dist:       GaussianDist{mean: 0.0, std: std_dev},
            svecs_buf:  svecs_buf,
            svals_buf:  svals_buf,
            alphas_buf: alphas_buf,
          });
        }
        &Data3dPreproc::Crop{crop_width, crop_height} => {
          // FIXME(20160407): this formulation is only valid for a single
          // crop, as it references `in_dims`.
          let max_offset_w = in_width - crop_width;
          let max_offset_h = in_height - crop_height;
          preprocs.push(Data3dPreprocOperator::Crop{
            transform:  CudnnTransformOp::new(
                CudnnTensorDesc::<f32>::create_4d_strided(
                    crop_width, crop_height, in_channels, batch_size,
                    1, in_width, in_width * in_height, in_width * in_height * in_channels,
                ).unwrap(),
                CudnnTensorDesc::<f32>::create_4d(
                    crop_width, crop_height, in_channels, batch_size,
                ).unwrap(),
            ),
            woff_range: Range::new(0, max_offset_w + 1),
            hoff_range: Range::new(0, max_offset_h + 1),
          });
        }
        &Data3dPreproc::FlipX => {
          preprocs.push(Data3dPreprocOperator::FlipX{
            coin_flip:  Range::new(0, 2),
          });
        }
        &Data3dPreproc::NormalizePixelwise{..} => {
          // FIXME(20160423)
          unimplemented!();
        }
        &Data3dPreproc::SubtractElemwiseMean{ref mean_path} => {
          let mut mean_file = match File::open(mean_path) {
            Ok(file) => file,
            Err(e) => panic!("failed to open mean path: {:?}", e),
          };
          let mean_arr_h: Array3d<f32> = match Array3d::deserialize(&mut mean_file) {
            Ok(arr) => arr,
            Err(_) => panic!("failed to deserialize mean array"),
          };
          let mut mean_array = DeviceBuffer::zeros(in_frame_len, ctx);
          mean_array.as_ref_mut(ctx).sync_load(mean_arr_h.as_slice());
          preprocs.push(Data3dPreprocOperator::SubtractElemwiseMean{
            add:        CudnnAddOp::new(
                CudnnTensorDesc::<f32>::create_4d(
                    in_width, in_height, in_channels, 1,
                ).unwrap(),
                CudnnTensorDesc::<f32>::create_4d(
                    in_width, in_height, in_channels, batch_size,
                ).unwrap(),
            ),
            mean_arr_h: mean_arr_h,
            mean_array: mean_array,
          });
        }
      }
    }
    Data3dOperator{
      batch_cap:    batch_size,
      config:       config,
      context:      context.clone(),
      in_buf_h:     repeat(0).take(batch_size * in_frame_len).collect(),
      in_buf:       DeviceBuffer::zeros(batch_size * in_frame_len, ctx),
      // FIXME(20160407): assuming that `in_frame_len` is as large as any
      // intermediate frame_len.
      tmp_buf:      DeviceBuffer::zeros(batch_size * in_frame_len, ctx),
      //tmp1_buf:     DeviceBuffer::zeros(batch_size * in_frame_len, ctx),
      //tmp2_buf:     DeviceBuffer::zeros(batch_size * in_frame_len, ctx),
      // FIXME(20160421): `out_buf` should have `out_frame_len` instead,
      // and a second temporary buffer should be used for intermediate values...
      // but in practice, if only pointers are used, then this does not cause
      // any problem to appear in downstream operators.
      out_buf:      Rc::new(RefCell::new(DeviceBuffer::zeros(batch_size * in_frame_len, ctx))),
      //out_buf:      Rc::new(RefCell::new(DeviceBuffer::zeros(batch_size * out_frame_len, ctx))),
      rng:          Xorshiftplus128Rng::new(&mut thread_rng()),
      preprocs:     preprocs,
    }
  }
}

impl Operator for Data3dOperator {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_buf.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    //let length = self.config.in_dims.len();
    let in_dims = self.config.in_dims;
    let (in_width, in_height, in_channels) = in_dims;
    let out_dims = self.config.get_out_dims();
    let (out_width, out_height, out_channels) = out_dims;
    let mut out_buf = self.out_buf.borrow_mut();
    let num_preprocs = self.config.preprocs.len();
    {
      let mut dst_buf = if num_preprocs % 2 == 0 {
        out_buf.as_ref_mut(ctx)
      } else {
        self.tmp_buf.as_ref_mut(ctx)
      };
      let in_buf = self.in_buf.as_ref(ctx);
      if self.config.normalize {
        in_buf.cast_bytes_normalized(&mut dst_buf);
      } else {
        in_buf.cast_bytes(&mut dst_buf);
      }
    }
    for (r, preproc) in self.config.preprocs.iter().enumerate() {
      let (src_buf, mut target_buf) = if (num_preprocs - r - 1) % 2 == 0 {
        (self.tmp_buf.as_ref(ctx), out_buf.as_ref_mut(ctx))
      } else {
        (out_buf.as_ref(ctx), self.tmp_buf.as_ref_mut(ctx))
      };
      match (preproc, &mut self.preprocs[r]) {
        ( &Data3dPreproc::Crop{crop_width, crop_height},
          &mut Data3dPreprocOperator::Crop{ref transform, ref woff_range, ref hoff_range},
        ) => {
          // FIXME(20160407): this formulation is only valid for a single
          // crop, as it references `in_dims`.
          /*let offset_w = self.rng.gen_range(0, in_width - crop_width);
          let offset_h = self.rng.gen_range(0, in_height - crop_height);*/
          let (offset_w, offset_h) = match phase {
            OpPhase::Inference => {
              let offset_w = (in_width - crop_width) / 2;
              let offset_h = (in_height - crop_height) / 2;
              (offset_w, offset_h)
            }
            OpPhase::Training{..} => {
              let offset_w = woff_range.ind_sample(&mut self.rng);
              let offset_h = hoff_range.ind_sample(&mut self.rng);
              (offset_w, offset_h)
            }
          };
          let buf_offset = offset_w + offset_h * in_width;
          unsafe { transform.transform(
              1.0, src_buf.as_ptr().offset(buf_offset as isize),
              0.0, target_buf.as_mut_ptr(),
              &*ctx.get_dnn(),
          ) }.unwrap();
        }

        ( &Data3dPreproc::AddPixelwiseColorNoise{..},
          &mut Data3dPreprocOperator::AddPixelwiseColorNoise,
        ) => {
          match phase {
            OpPhase::Inference => {
              src_buf.send(&mut target_buf);
            }
            OpPhase::Training{..} => {
              // FIXME(20160423)
              unimplemented!();
            }
          }
        }

        ( &Data3dPreproc::AddPixelwisePCALightingNoise{..},
          &mut Data3dPreprocOperator::AddPixelwisePCALightingNoise{ref dist, ref mut alphas_buf, ref mut svals_buf, ref mut svecs_buf},
        ) => {
          match phase {
            OpPhase::Inference => {
              src_buf.send(&mut target_buf);
            }
            OpPhase::Training{..} => {
              // FIXME(20160423)
              assert_eq!(3, self.config.in_dims.2);
              alphas_buf.as_ref_mut(ctx).sample(dist);
              unsafe { rembrandt_kernel_batch_map_preproc_pca3_noise(
                  src_buf.as_ptr(),
                  (out_width * out_height) as i32,
                  batch_size as i32,
                  alphas_buf.as_ref(ctx).as_ptr(),
                  svals_buf.as_ref(ctx).as_ptr(),
                  svecs_buf.as_ref(ctx).as_ptr(),
                  target_buf.as_mut_ptr(),
                  ctx.stream.ptr,
              ) };
            }
          }
        }

        ( &Data3dPreproc::SubtractElemwiseMean{..},
          &mut Data3dPreprocOperator::SubtractElemwiseMean{ref mut add, ref mut mean_array, ..},
        ) => {
          let alpha = if self.config.normalize {
            -1.0 / 255.0
          } else {
            -1.0
          };
          src_buf.send(&mut target_buf);
          add.set_batch_size(batch_size);
          unsafe { add.forward(
              alpha,
              mean_array.as_ref(ctx).as_ptr(),
              1.0,
              target_buf.as_mut_ptr(),
              &*ctx.get_dnn(),
          ) }.unwrap();
        }

        ( &Data3dPreproc::FlipX,
          &mut Data3dPreprocOperator::FlipX{ref coin_flip},
        ) => {
          match phase {
            OpPhase::Inference => {
              src_buf.send(&mut target_buf);
            }
            OpPhase::Training{..} => {
              match coin_flip.ind_sample(&mut self.rng) {
                0 => {
                  src_buf.send(&mut target_buf);
                }
                1 => {
                  // FIXME(20160504): this version of flip comes _before_ crop.
                  assert!(in_width <= 256);
                  unsafe { rembrandt_kernel_batch_blockmap256_flip(
                      src_buf.as_ptr(),
                      in_width as i32,
                      (in_height * in_channels * batch_size) as i32,
                      target_buf.as_mut_ptr(),
                      ctx.stream.ptr,
                  ) };
                }
                _ => unreachable!(),
              }
            }
          }
        }

        _ => unimplemented!(),
      }
    }
  }

  fn backward(&mut self, _batch_size: usize) {
    // Do nothing.
  }

  /*fn sync_grads(&mut self) {
    // Do nothing.
  }

  fn sync_params(&mut self) {
    // Do nothing.
  }

  fn reset_grads(&mut self, _scale: f32) {
    // Do nothing.
  }*/
}

impl InputOperator for Data3dOperator {
  fn downcast(&self) -> &Operator {
    self
  }

  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    assert!(batch_idx < self.batch_cap);
    let frame_len = self.config.in_dims.len();
    &mut self.in_buf_h[batch_idx * frame_len .. (batch_idx + 1) * frame_len]
  }

  fn load_frames(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    {
      // FIXME(20160329): does not use `batch_size` at all!
      //loop {
      //}
      let in_buf_h = &self.in_buf_h;
      let mut in_buf = self.in_buf.as_ref_mut(ctx);
      in_buf.sync_load(in_buf_h);
    }
  }

  fn stage_shape(&mut self, batch_idx: usize, shape: (usize, usize, usize)) {
    // Do nothing, but check that the input shape is correct.
    assert!(batch_idx < self.batch_cap);
    assert_eq!(shape, self.config.in_dims);
  }

  /*fn load_shapes(&mut self, batch_size: usize) {
    // Do nothing.
    assert!(batch_size <= self.batch_cap);
  }*/
}

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
  fn params_len(&self) -> usize {
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

#[derive(Clone, Copy, Debug)]
pub enum Conv2dFwdBackend {
  CudnnFastest,
  CudnnImplicitPrecompGemm,
  CudnnFftTiling,
}

#[derive(Clone, Copy, Debug)]
pub enum Conv2dBwdBackend {
  CudnnFastest,
  CudnnDeterministic,
}

#[derive(Clone, Copy, Debug)]
pub struct Conv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub conv_size:    usize,
  pub conv_stride:  usize,
  pub conv_pad:     usize,
  pub out_channels: usize,
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
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

pub struct Conv2dOperator {
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

  backward:     Option<Conv2dBwdOperator>,
  hv_backward:  Option<Conv2dHvBwdOperator>,
}

struct Conv2dBwdOperator {
  grad_weights: DeviceArray2d<f32>,
  grad_bias:    DeviceArray2d<f32>,
  acc_grad_weights: DeviceArray2d<f32>,
  acc_grad_bias:    DeviceArray2d<f32>,
  save_weights: DeviceArray2d<f32>,
  save_bias:    DeviceArray2d<f32>,

  conv_bwd_w:   CudnnConvBwdFilterOp,
  conv_bwd_d:   CudnnConvBwdDataOp,
}

struct Conv2dHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
  dir_bias:     DeviceArray2d<f32>,
}

impl Conv2dOperator {
  pub fn new(batch_size: usize, capability: OpCapability, params_offset: usize, config: Conv2dOperatorConfig, prev_op: Option<&Operator>, /*comm_worker: Option<Rc<RefCell<Comm>>>,*/ context: Rc<DeviceContext>) -> Conv2dOperator {
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
        acc_grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        acc_grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        save_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        save_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        conv_bwd_w:   conv_bwd_w,
        conv_bwd_d:   conv_bwd_d,
        //comm_worker:  comm_worker.unwrap(),
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
      params_off:   params_offset,
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

impl Operator for Conv2dOperator {
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
        panic!("Conv2dOperator: params init explicitly disabled");
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

  fn decode_params(&mut self, blob: &[u8]) -> usize {
    let Conv2dOperatorConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("Conv2dOperator failed to deserialize weights!");
    let load_bias = Array2d::deserialize(&mut reader)
      .ok().expect("Conv2dOperator failed to deserialize bias!");
    assert_eq!((conv_size * conv_size * in_channels, out_channels), load_weights.as_view().bound());
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
    let Conv2dOperatorConfig{in_dims, out_channels, conv_size, ..} = self.config;
    let (_, _, in_channels) = in_dims;
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    let mut reader = Cursor::new(blob);

    let load_update_weights = Array2d::deserialize(&mut reader)
      .ok().expect("AffineOperator failed to deserialize weights!");
    let load_update_bias = Array2d::deserialize(&mut reader)
      .ok().expect("AffineOperator failed to deserialize bias!");
    assert_eq!((conv_size * conv_size * in_channels, out_channels), load_update_weights.as_view().bound());
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
    // FIXME(20160423): This code is broken!
    unimplemented!();
  }

  fn sync_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights); //, ctx);
    comm_worker.store(self.params_off, &mut self.bias); //, ctx);
    // FIXME(20160423): This code is broken!
    unimplemented!();
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
