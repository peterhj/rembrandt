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
use array_cuda::device::linalg::{VectorExt, BlasMatrixExt, BlasVectorExt, Transpose};
use array_cuda::device::memory::{DeviceBufferInitExt, DeviceBuffer, DeviceBufferRef, DeviceBufferRefMut};
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
use std::cell::{RefCell, Ref, RefMut};
use std::cmp::{max, min};
use std::collections::{Bound, BTreeMap};
use std::fs::{File};
use std::io::{Cursor};
use std::iter::{repeat};
use std::marker::{PhantomData};
//use std::ops::{Deref};
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
  fn read<'a>(&'a mut self, offset: usize, dst: &mut DeviceBufferRefMut<'a, f32>) -> usize;
  fn accumulate_read<'a>(&'a mut self, offset: usize, alpha: f32, beta: f32, dst: &mut DeviceBufferRefMut<'a, f32>) -> usize { unimplemented!(); }
}

pub trait OpWrite {
  fn write<'a>(&'a mut self, offset: usize, src: &DeviceBufferRef<'a, f32>) -> usize;
  fn accumulate_write<'a>(&'a mut self, offset: usize, alpha: f32, beta: f32, src: &DeviceBufferRef<'a, f32>) -> usize { unimplemented!(); }
}

pub trait OpCursorInner {
  type Extra;

  fn extra(&self) -> Self::Extra;
}

pub struct OpCursor<T> where T: OpCursorInner {
  pub inner:    T,
  pub extra:    T::Extra,
}

impl<T> OpCursor<T> where T: OpCursorInner {
  pub fn new(inner: T) -> OpCursor<T> {
    let extra = inner.extra();
    OpCursor{
      inner:    inner,
      extra:    extra,
    }
  }
}

impl OpCursorInner for DeviceBuffer<f32> {
  type Extra = ();

  fn extra(&self) -> () {
    ()
  }
}

impl OpRead for OpCursor<DeviceBuffer<f32>> {
  fn read<'ctx>(&'ctx mut self, offset: usize, dst: &mut DeviceBufferRefMut<'ctx, f32>) -> usize {
    let buf_len = dst.len();
    dst.copy(&self.inner.as_ref_range(offset, offset + buf_len, dst.ctx));
    buf_len
  }

  fn accumulate_read<'a>(&'a mut self, offset: usize, alpha: f32, beta: f32, dst: &mut DeviceBufferRefMut<'a, f32>) -> usize {
    let buf_len = dst.len();
    let src = self.inner.as_ref_range(offset, offset + buf_len, dst.ctx);
    //dst.vector_scale(beta);
    dst.vector_add(alpha, &src, beta);
    buf_len
  }
}

impl OpWrite for OpCursor<DeviceBuffer<f32>> {
  fn write<'a>(&'a mut self, offset: usize, src: &DeviceBufferRef<'a, f32>) -> usize {
    let buf_len = src.len();
    self.inner.as_ref_mut_range(offset, offset + buf_len, src.ctx).copy(src);
    buf_len
  }

  fn accumulate_write<'a>(&'a mut self, offset: usize, alpha: f32, beta: f32, src: &DeviceBufferRef<'a, f32>) -> usize {
    let buf_len = src.len();
    let mut dst = self.inner.as_ref_mut_range(offset, offset + buf_len, src.ctx);
    //dst.vector_scale(beta);
    dst.vector_add(alpha, &src, beta);
    buf_len
  }
}

pub struct ExtentMap {
  offset_map:   BTreeMap<usize, (usize, usize, usize)>,
}

impl ExtentMap {
  pub fn new(part_lens: Vec<usize>) -> ExtentMap {
    let num_parts = part_lens.len();
    let mut offset_map = BTreeMap::new();
    let mut offset = 0;
    for part_idx in 0 .. num_parts {
      let part_len = part_lens[part_idx];
      offset_map.insert(offset, (offset, part_idx, part_len));
      offset_map.insert(offset + part_len - 1, (offset, part_idx, part_len));
      offset += part_len;
    }
    //println!("DEBUG: extent map: part lens: {:?}", part_lens);
    //println!("DEBUG: extent map: offsets: {:?}", offset_map);
    //println!("DEBUG: extent map: offset: {:?}", offset);
    ExtentMap{offset_map: offset_map}
  }

  pub fn foreach_extent<F>(&self, lower: usize, upper: usize, mut f: F) where F: FnMut(usize, (usize, usize), (usize, usize)) {
    let buf_len = upper - lower;
    let mut tmp_offset = lower;
    //println!("DEBUG: foreach extent: {} {} {}", lower, upper, buf_len);
    let mut prev_offset = None;
    for (_, &(part_offset, part_idx, part_len)) in self.offset_map.range(Bound::Included(&lower), Bound::Unbounded) {
      if tmp_offset >= upper {
        break;
      }
      match prev_offset {
        None => {}
        Some(prev_offset) => if prev_offset == part_offset {
          continue;
        }
      }
      assert!(part_offset <= tmp_offset);
      assert!(tmp_offset <= part_offset + part_len);
      let start_i = tmp_offset - part_offset;
      let end_i = min(part_len, start_i + min(part_len, buf_len - (tmp_offset - lower)));
      assert!(0 <= start_i);
      assert!(start_i <= end_i);
      assert!(end_i <= part_len);
      let tmp_part_len = end_i - start_i;
      //println!("DEBUG:   extent: {} {} {} {}", tmp_offset, start_i, end_i, tmp_part_len);
      f(part_idx, (start_i, end_i), (tmp_offset - lower, tmp_offset - lower + tmp_part_len));
      tmp_offset += tmp_part_len;
      prev_offset = Some(part_offset)
    }
    //println!("DEBUG:   offset: {}", tmp_offset);
    assert_eq!(upper, tmp_offset);
  }
}

impl OpCursorInner for Vec<DeviceBuffer<f32>> {
  type Extra = ExtentMap;

  fn extra(&self) -> ExtentMap {
    let part_lens: Vec<_> = self.iter().map(|buf| buf.len()).collect();
    ExtentMap::new(part_lens)
  }
}

impl OpRead for OpCursor<Vec<DeviceBuffer<f32>>> {
  fn read<'ctx>(&'ctx mut self, offset: usize, dst: &mut DeviceBufferRefMut<'ctx, f32>) -> usize {
    let buf_len = dst.len();
    let &mut OpCursor{ref extra, ref mut inner} = self;
    extra.foreach_extent(offset, offset + buf_len, |part_idx, (part_lower, part_upper), (whole_lower, whole_upper)| {
      dst.mut_range(whole_lower, whole_upper)
        .copy(&inner[part_idx].as_ref_range(part_lower, part_upper, dst.ctx));
    });
    buf_len
  }
}

impl OpWrite for OpCursor<Vec<DeviceBuffer<f32>>> {
  fn write<'ctx>(&'ctx mut self, offset: usize, src: &DeviceBufferRef<'ctx, f32>) -> usize {
    let buf_len = src.len();
    let &mut OpCursor{ref extra, ref mut inner} = self;
    extra.foreach_extent(offset, offset + buf_len, |part_idx, (part_lower, part_upper), (whole_lower, whole_upper)| {
      inner[part_idx].as_ref_mut_range(part_lower, part_upper, src.ctx)
        .copy(&src.range(whole_lower, whole_upper));
    });
    buf_len
  }
}

pub struct CowInput<T> {
  inner:    Rc<CowIoInner<T>>,
}

impl<T> CowInput<T> {
  pub fn new(input: T, num_outputs: usize) -> CowInput<T> {
    let mut outputs = Vec::with_capacity(num_outputs);
    for _ in 0 .. num_outputs {
      outputs.push(RefCell::new(CowData::Passthrough));
    }
    CowInput{
      inner:    Rc::new(CowIoInner{
        input:      RefCell::new(input),
        outputs:    outputs,
      }),
    }
  }

  pub fn output(&self, out_idx: usize) -> CowOutput<T> {
    assert!(out_idx < self.inner.outputs.len());
    CowOutput{
      inner:    self.inner.clone(),
      out_idx:  out_idx,
    }
  }

  pub fn borrow(&self) -> Ref<T> {
    self.inner.input.borrow()
  }

  pub fn borrow_mut(&self) -> RefMut<T> {
    self.inner.input.borrow_mut()
  }

  pub fn foreach_owned<F>(&self, mut map: F)
  where F: FnMut(&T, &mut T) {
    let input = self.inner.input.borrow();
    for output in self.inner.outputs.iter() {
      let output = output.borrow_mut();
      if output.is_owned() {
        let mut output = RefMut::map(output, |r| {
          match r {
            &mut CowData::Passthrough => panic!(),
            &mut CowData::Owned(ref mut owned) => owned,
          }
        });
        map(&*input, &mut *output);
      }
    }
  }
}

pub struct CowOutput<T> {
  inner:    Rc<CowIoInner<T>>,
  out_idx:  usize,
}

impl<T> CowOutput<T> {
  pub fn borrow(&self) -> Ref<T> {
    let output = self.inner.outputs[self.out_idx].borrow();
    if !output.is_owned() {
      self.inner.input.borrow()
    } else {
      Ref::map(output, |r| {
        match r {
          &CowData::Passthrough => panic!(),
          &CowData::Owned(ref owned) => owned,
        }
      })
    }
  }

  pub fn borrow_mut<F>(&self, construct: F) -> RefMut<T>
  where F: FnOnce(&T) -> T {
    let mut output = self.inner.outputs[self.out_idx].borrow_mut();
    if !output.is_owned() {
      *output = CowData::Owned(construct(&*self.inner.input.borrow()));
    }
    RefMut::map(output, |r| {
      match r {
        &mut CowData::Passthrough => panic!(),
        &mut CowData::Owned(ref mut owned) => owned,
      }
    })
  }
}

enum CowData<T> {
  Passthrough,
  Owned(T),
}

impl<T> CowData<T> {
  pub fn is_owned(&self) -> bool {
    match self {
      &CowData::Passthrough => false,
      &CowData::Owned(_) => true,
    }
  }
}

//impl<T> Deref for CowData<T> {
//}

struct CowIoInner<T> {
  input:    RefCell<T>,
  outputs:  Vec<RefCell<CowData<T>>>,
}

pub trait OperatorState {
}

pub struct AcyclicOperatorState {
}

impl OperatorState for AcyclicOperatorState {
}

impl AcyclicOperatorState {
  pub fn act(&self, _arm: usize) -> SharedDeviceBuf<f32> { unimplemented!(); }
  pub fn delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> { unimplemented!(); }
  pub fn r_act(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> { unimplemented!(); }
  pub fn r_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> { unimplemented!(); }
}

pub struct SequenceOperatorState {
}

impl OperatorState for SequenceOperatorState {
}

impl SequenceOperatorState {
  pub fn act(&self, _timestep: usize, _arm: usize) -> SharedDeviceBuf<f32> { unimplemented!(); }
  pub fn delta(&self, _timestep: usize, _arm: usize) -> Option<SharedDeviceBuf<f32>> { unimplemented!(); }
  //pub fn r_act(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> { unimplemented!(); }
  //pub fn r_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> { unimplemented!(); }
}

pub trait Operator {
  fn upcast_input(&mut self) -> &mut InputOperator { unreachable!(); }
  fn upcast_loss(&mut self) -> &mut LossOperator { unreachable!(); }

  //fn reset_batch(&self) { unimplemented!(); }
  #[deprecated] fn batch_size(&self) -> usize { unimplemented!(); }
  fn params_len(&self) -> usize { 0 }
  fn get_output_act(&self, _arm: usize) -> SharedDeviceBuf<f32> { unimplemented!(); }
  fn get_output_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> { unimplemented!(); }
  fn get_output_r_act(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> { unimplemented!(); }
  fn get_output_r_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> { unimplemented!(); }

  fn init_param(&mut self, _shared_seed: [u64; 2]) {}
  fn decode_param(&mut self, _blob: &[u8]) -> usize { 0 }
  fn encode_param(&mut self, _blob: &mut Vec<u8>) {}
  fn decode_state(&mut self, _blob: &[u8]) -> usize { 0 }
  fn encode_state(&mut self, _blob: &mut Vec<u8>) {}
  fn read_param(&mut self, _offset: usize, _reader: &mut OpRead) -> usize { 0 }
  fn write_param(&mut self, _offset: usize, _writer: &mut OpWrite) -> usize { 0 }
  fn forward(&mut self, batch_size: usize, phase: OpPhase);

  // Requires `Backward` capability.
  fn reset(&mut self) {}
  fn reset_grad(&mut self) { unimplemented!(); }
  fn read_grad(&mut self, _offset: usize, _reader: &mut OpRead) -> usize { 0 }
  fn write_grad(&mut self, _offset: usize, _writer: &mut OpWrite) -> usize { 0 }
  fn accumulate_grad_(&mut self, _offset: usize, _alpha: f32, _mu: f32, _grad_acc_writer: &mut OpWrite) -> usize { 0 }
  fn step(&mut self, _offset: usize, _step_size: f32, _grad_acc_reader: &mut OpRead) -> usize { 0 }
  fn backward(&mut self, batch_size: usize);
  fn regularize(&mut self, _reg: Regularization) {}

  fn reset_stats(&mut self) {}
  fn estimate_stats(&mut self, _acc_sample_size: usize, _batch_size: usize) {}
  fn finalize_stats(&mut self, _sample_size: usize) {}

  fn accumulate_grad(&mut self, _scale: f32, _momentum: f32) {}
  fn update_param(&mut self, _scale: f32) {}
  fn update_param2(&mut self, _grad_scale: f32, _update_scale: f32) {}
  #[deprecated] fn save_params(&mut self) {}
  #[deprecated] fn restore_params(&mut self) {}
  //fn set_grads_with_params_diff(&mut self) {}
  #[deprecated] fn stage_grads(&mut self, _offset: usize, _comm_worker: &mut CommWorker) -> usize { 0 }
  #[deprecated] fn merge_grads(&mut self, _offset: usize, _comm_worker: &mut CommWorker) -> usize { 0 }
  #[deprecated] fn stage_params(&mut self, _offset: usize, _comm_worker: &mut CommWorker) -> usize { 0 }
  #[deprecated] fn merge_params(&mut self, _offset: usize, _comm_worker: &mut CommWorker) -> usize { 0 }

  // Requires `RForward` capability.
  fn read_direction(&mut self, _offset: usize, _reader: &mut OpRead) -> usize { 0 }
  fn write_direction(&mut self, _offset: usize, _writer: &mut OpWrite) -> usize { 0 }
  fn r_forward(&mut self, _batch_size: usize) { unimplemented!(); }

  // Requires `RBackward` capability.
  fn reset_r_grad(&mut self) { unimplemented!(); }
  fn r_backward(&mut self, _batch_size: usize) { unimplemented!(); }
}

pub trait InputOperator: Operator {
  fn downcast(&self) -> &Operator { unimplemented!(); }
  fn stage_shape(&mut self, batch_idx: usize, shape: (usize, usize, usize));
  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8];
  fn load_frames(&mut self, batch_size: usize);
  fn preload_frame(&mut self, _batch_idx: usize) { unimplemented!(); }
  fn wait_preload_frames(&mut self, _batch_size: usize) { unimplemented!(); }
}

pub trait LossOperator: Operator {
  fn downcast(&self) -> &Operator { unimplemented!(); }

  fn stage_label(&mut self, batch_idx: usize, label: &SampleLabel);
  fn load_labels(&mut self, batch_size: usize);
  fn stage_weight(&mut self, batch_idx: usize, weight: f32);
  fn load_weights(&mut self, batch_size: usize);
  fn stage_category_weight(&mut self, _batch_idx: usize, _category: i32, _cat_weight: f32) {}
  fn load_category_weights(&mut self, _batch_size: usize) {}
  //fn stage_r_weight(&mut self, batch_idx: usize, weight: f32);
  //fn load_r_weights(&mut self, batch_size: usize);

  fn reset_targets(&mut self, _batch_size: usize) { unimplemented!(); }
  fn set_targets_with_r_loss(&mut self, _batch_size: usize) { unimplemented!(); }

  fn store_loss(&mut self, batch_size: usize) -> f32;
  fn store_output_values(&mut self, batch_size: usize);
  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32>;
  fn store_output_categories(&mut self, batch_size: usize);
  //fn get_output_categories(&self, batch_size: usize) -> &Array2d<i32>;
  fn get_output_categories(&mut self, batch_size: usize) -> &[i32];
  fn accuracy_count(&mut self, batch_size: usize) -> usize;
  //fn reset_loss(&mut self);

  /*fn forward_loss(&mut self, batch_size: usize);
  fn backward_loss(&mut self, batch_size: usize);
  fn r_forward_loss(&mut self, batch_size: usize);
  fn r_backward_loss(&mut self, batch_size: usize);*/
}

pub trait CompleteOperator: InputOperator + LossOperator {
}

pub enum OperatorVariant {
  Hidden(Box<Operator>),
  Input(Box<Operator>),
  Loss(Box<Operator>),
  //Split(Box<Operator>),
  //Join(Box<Operator>),
}

pub enum OperatorNode {
  Hidden(Box<Operator>),
  Input(Box<InputOperator>),
  Loss(Box<LossOperator>),
  //Split(Box<Operator>),
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
  CopySplit(SplitOperatorConfig),
  AddJoin(JoinOperatorConfig),
  //_Dummy(PhantomData<Comm>),
}

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

  pub fn build_variant(&self, batch_size: usize, capability: OpCapability, prev_ops: Vec<(usize, &Operator)>, context: Rc<DeviceContext>) -> OperatorVariant {
    match self {
      &OperatorConfig::Affine(ref cfg) => {
        assert_eq!(1, prev_ops.len());
        assert_eq!(0, prev_ops[0].0);
        let prev_op = Some(prev_ops[0].1);
        OperatorVariant::Hidden(Box::new(AffineOperator::new(batch_size, capability, *cfg, prev_op, context)))
      }
      &OperatorConfig::Conv2d(ref cfg) => {
        assert_eq!(1, prev_ops.len());
        assert_eq!(0, prev_ops[0].0);
        let prev_op = Some(prev_ops[0].1);
        OperatorVariant::Hidden(Box::new(Conv2dOperator::new(batch_size, capability, 0, *cfg, prev_op, context)))
      }
      &OperatorConfig::BNormConv2d(ref cfg) => {
        assert_eq!(1, prev_ops.len());
        OperatorVariant::Hidden(Box::new(BNormConv2dOperator::new(batch_size, capability, *cfg, prev_ops[0].0, prev_ops[0].1, context)))
      }
      &OperatorConfig::StackResConv2d(ref cfg) => {
        //unimplemented!();
        assert_eq!(1, prev_ops.len());
        assert_eq!(0, prev_ops[0].0);
        let prev_op = Some(prev_ops[0].1);
        OperatorVariant::Hidden(Box::new(StackResConv2dOperator::new(batch_size, capability, 0, *cfg, prev_op, context)))
      }
      &OperatorConfig::ProjStackResConv2d(ref cfg) => {
        //unimplemented!();
        assert_eq!(1, prev_ops.len());
        assert_eq!(0, prev_ops[0].0);
        let prev_op = Some(prev_ops[0].1);
        OperatorVariant::Hidden(Box::new(ProjStackResConv2dOperator::new(batch_size, capability, 0, *cfg, prev_op, context)))
      }
      &OperatorConfig::Pool2d(ref cfg) => {
        assert_eq!(1, prev_ops.len());
        assert_eq!(0, prev_ops[0].0);
        let prev_op = Some(prev_ops[0].1);
        OperatorVariant::Hidden(Box::new(Pool2dOperator::new(batch_size, capability, *cfg, prev_op, context)))
      }
      &OperatorConfig::Dropout(ref cfg) => {
        assert_eq!(1, prev_ops.len());
        assert_eq!(0, prev_ops[0].0);
        let prev_op = Some(prev_ops[0].1);
        OperatorVariant::Hidden(Box::new(DropoutOperator::new(batch_size, *cfg, prev_op, context)))
      }
      &OperatorConfig::Data3d(ref cfg) => {
        assert_eq!(0, prev_ops.len());
        OperatorVariant::Input(Box::new(Data3dOperator::new(batch_size, cfg.clone(), context)))
      }
      &OperatorConfig::VarData3d(ref cfg) => {
        assert_eq!(0, prev_ops.len());
        OperatorVariant::Input(Box::new(VarData3dOperator::new(batch_size, capability, cfg.clone(), context)))
      }
      &OperatorConfig::SoftmaxKLLoss(ref cfg) => {
        assert_eq!(1, prev_ops.len());
        assert_eq!(0, prev_ops[0].0);
        let prev_op = Some(prev_ops[0].1);
        OperatorVariant::Loss(Box::new(SoftmaxKLLossOperator::new(batch_size, capability, *cfg, prev_op, context)))
      }
      &OperatorConfig::CopySplit(ref cfg) => {
        assert_eq!(1, prev_ops.len());
        OperatorVariant::Hidden(Box::new(CopySplitOperator::new(batch_size, capability, *cfg, prev_ops[0].0, prev_ops[0].1, context)))
      }
      &OperatorConfig::AddJoin(ref cfg) => {
        OperatorVariant::Hidden(Box::new(AddJoinOperator::new(batch_size, capability, *cfg, prev_ops, context)))
      }
      //_ => unreachable!(),
    }
  }

  pub fn build_node(&self, batch_size: usize, capability: OpCapability, params_offset: Option<usize>, prev_op: Option<&Operator>, /*comm_worker: Option<Rc<RefCell<Comm>>>,*/ context: Rc<DeviceContext>) -> OperatorNode {
    match self {
      &OperatorConfig::Affine(ref cfg) => {
        OperatorNode::Hidden(Box::new(AffineOperator::new(batch_size, capability, /*params_offset.unwrap(),*/ *cfg, prev_op, /*comm_worker,*/ context)))
      }
      &OperatorConfig::Conv2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(Conv2dOperator::new(batch_size, capability, params_offset.unwrap(), *cfg, prev_op, /*comm_worker,*/ context)))
      }
      &OperatorConfig::BNormConv2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(BNormConv2dOperator::new(batch_size, capability, *cfg, 0, prev_op.unwrap(), /*comm_worker,*/ context)))
      }
      &OperatorConfig::StackResConv2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(StackResConv2dOperator::new(batch_size, capability, params_offset.unwrap(), *cfg, prev_op, /*comm_worker,*/ context)))
      }
      &OperatorConfig::ProjStackResConv2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(ProjStackResConv2dOperator::new(batch_size, capability, params_offset.unwrap(), *cfg, prev_op, /*comm_worker,*/ context)))
      }
      &OperatorConfig::Pool2d(ref cfg) => {
        OperatorNode::Hidden(Box::new(Pool2dOperator::new(batch_size, capability, *cfg, prev_op, context)))
      }
      &OperatorConfig::Dropout(ref cfg) => {
        OperatorNode::Hidden(Box::new(DropoutOperator::new(batch_size, *cfg, prev_op, context)))
      }
      &OperatorConfig::Data3d(ref cfg) => {
        OperatorNode::Input(Box::new(Data3dOperator::new(batch_size, cfg.clone(), context)))
      }
      &OperatorConfig::VarData3d(ref cfg) => {
        OperatorNode::Input(Box::new(VarData3dOperator::new(batch_size, capability, cfg.clone(), context)))
      }
      &OperatorConfig::SoftmaxKLLoss(ref cfg) => {
        OperatorNode::Loss(Box::new(SoftmaxKLLossOperator::new(batch_size, capability, *cfg, prev_op, context)))
      }
      &OperatorConfig::CopySplit(ref cfg) => {
        unimplemented!();
      }
      &OperatorConfig::AddJoin(ref cfg) => {
        unimplemented!();
      }
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

/*pub struct SplitOperator {
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
      shared_act:   prev_op.unwrap().get_output_act(),
      in_delta:     prev_op.unwrap().get_output_delta(),
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

  /*fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.shared_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    if self.in_delta.is_some() {
      assert!(self.out_deltas.len() >= 1);
      Some(self.out_deltas[self.out_deltas.len()-1].clone())
    } else {
      None
    }
  }*/

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
}*/

#[derive(Clone, Copy, Debug)]
pub struct SplitOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub num_out_arms: usize,
}

pub struct CopySplitOperator {
  //num_out_arms: usize,
  batch_cap:    usize,
  config:       SplitOperatorConfig,
  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  //out_acts:     Vec<SharedDeviceBuf<f32>>,

  backward:     Option<CopySplitBwdOperator>,
  r_forward:    Option<CopySplitRFwdOperator>,
}

struct CopySplitBwdOperator {
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_deltas:   Vec<Option<SharedDeviceBuf<f32>>>,
}

struct CopySplitRFwdOperator {
  in_r_act:     SharedDeviceBuf<f32>,
  //out_r_acts:   Vec<SharedDeviceBuf<f32>>,
}

impl CopySplitOperator {
  pub fn new(batch_size: usize, capability: OpCapability, config: SplitOperatorConfig, prev_arm: usize, prev_op: &Operator, context: Rc<DeviceContext>) -> CopySplitOperator {
    let num_out_arms = config.num_out_arms;
    let out_length = config.in_dims.len();

    let ctx = &(*context).as_ref();

    //assert_eq!(1, prev_ops.len());
    let in_act = prev_op.get_output_act(prev_arm);

    let backward = if capability.backward_enabled() {
      let in_delta = prev_op.get_output_delta(prev_arm);
      let mut out_deltas = Vec::with_capacity(num_out_arms);
      for arm in 0 .. num_out_arms {
        if in_delta.is_some() {
          out_deltas.push(Some(Rc::new(RefCell::new(DeviceBuffer::zeros(out_length * batch_size, ctx)))));
        } else {
          out_deltas.push(None);
        }
      }
      Some(CopySplitBwdOperator{
        in_delta:   in_delta,
        out_deltas: out_deltas,
      })
    } else {
      None
    };

    let r_forward = if capability.r_forward_enabled() {
      let in_r_act = prev_op.get_output_r_act(prev_arm).unwrap();
      Some(CopySplitRFwdOperator{
        in_r_act:   in_r_act,
      })
    } else {
      None
    };

    CopySplitOperator{
      //num_out_arms: num_out_arms,
      batch_cap:    batch_size,
      config:       config,
      context:      context.clone(),
      in_act:       in_act,
      backward:     backward,
      r_forward:    r_forward,
    }
  }
}

impl Operator for CopySplitOperator {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_act(&self, _arm: usize) -> SharedDeviceBuf<f32> {
    assert!(_arm < self.config.num_out_arms);
    self.in_act.clone()
  }

  fn get_output_delta(&self, arm: usize) -> Option<SharedDeviceBuf<f32>> {
    self.backward.as_ref().and_then(|bwd| bwd.out_deltas[arm].clone())
  }

  fn get_output_r_act(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    assert!(_arm < self.config.num_out_arms);
    self.r_forward.as_ref().map(|r_fwd| r_fwd.in_r_act.clone())
  }

  fn get_output_r_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    unimplemented!();
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    // Do nothing.
  }

  fn backward(&mut self, batch_size: usize) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    if let Some(ref mut in_delta) = backward.in_delta {
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);
      in_delta.copy(&backward.out_deltas[0].as_mut().unwrap().borrow_mut().as_ref(ctx));
      for arm in 1 .. self.config.num_out_arms {
        in_delta.vector_add(1.0, &backward.out_deltas[arm].as_mut().unwrap().borrow_mut().as_ref(ctx), 1.0);
      }
    }
  }

  fn r_forward(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    // Do nothing.
  }
}

#[derive(Clone, Copy, Debug)]
pub struct JoinOperatorConfig {
  pub num_in_arms:  usize,
  pub out_dims:     (usize, usize, usize),
  pub act_func:     ActivationFunction,
}

pub struct AddJoinOperator {
  //num_in_arms:  usize,
  batch_cap:    usize,
  config:       JoinOperatorConfig,
  context:      Rc<DeviceContext>,

  in_acts:      Vec<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,

  backward:     Option<AddJoinBwdOperator>,
  r_forward:    Option<AddJoinRFwdOperator>,
}

struct AddJoinBwdOperator {
  in_deltas:    Vec<Option<SharedDeviceBuf<f32>>>,
  out_delta:    SharedDeviceBuf<f32>,
}

struct AddJoinRFwdOperator {
  in_r_acts:    Vec<SharedDeviceBuf<f32>>,
  out_r_act:    SharedDeviceBuf<f32>,
}

impl AddJoinOperator {
  pub fn new(batch_size: usize, capability: OpCapability, config: JoinOperatorConfig, prev_ops: Vec<(usize, &Operator)>, context: Rc<DeviceContext>) -> AddJoinOperator {
    let num_in_arms = prev_ops.len();
    assert_eq!(num_in_arms, config.num_in_arms);
    let out_length = config.out_dims.len();

    let ctx = &(*context).as_ref();

    let mut in_acts = Vec::with_capacity(num_in_arms);
    for arm in 0 .. num_in_arms {
      let prev_arm = prev_ops[arm].0;
      let prev_op = prev_ops[arm].1;
      in_acts.push(prev_op.get_output_act(prev_arm));
    }
    let out_act = Rc::new(RefCell::new(DeviceBuffer::zeros(out_length * batch_size, ctx)));

    let backward = if capability.backward_enabled() {
      let mut in_deltas = Vec::with_capacity(num_in_arms);
      for arm in 0 .. num_in_arms {
        let prev_arm = prev_ops[arm].0;
        let prev_op = prev_ops[arm].1;
        in_deltas.push(prev_op.get_output_delta(prev_arm));
      }
      let out_delta = Rc::new(RefCell::new(DeviceBuffer::zeros(out_length * batch_size, ctx)));
      Some(AddJoinBwdOperator{
        in_deltas:  in_deltas,
        out_delta:  out_delta,
      })
    } else {
      None
    };

    let r_forward = if capability.r_forward_enabled() {
      let mut in_r_acts = Vec::with_capacity(num_in_arms);
      for arm in 0 .. num_in_arms {
        let prev_arm = prev_ops[arm].0;
        let prev_op = prev_ops[arm].1;
        in_r_acts.push(prev_op.get_output_r_act(prev_arm).unwrap());
      }
      let out_r_act = Rc::new(RefCell::new(DeviceBuffer::zeros(out_length * batch_size, ctx)));
      Some(AddJoinRFwdOperator{
        in_r_acts:  in_r_acts,
        out_r_act:  out_r_act,
      })
    } else {
      None
    };

    AddJoinOperator{
      //num_in_arms:  num_in_arms,
      batch_cap:    batch_size,
      config:       config,
      context:      context.clone(),
      in_acts:      in_acts,
      out_act:      out_act,
      backward:     backward,
      r_forward:    r_forward,
    }
  }
}

impl Operator for AddJoinOperator {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_act(&self, _arm: usize) -> SharedDeviceBuf<f32> {
    assert_eq!(0, _arm);
    self.out_act.clone()
  }

  fn get_output_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    assert_eq!(0, _arm);
    self.backward.as_ref().map(|bwd| bwd.out_delta.clone())
  }

  fn get_output_r_act(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    /*assert!(self.r_forward.is_some());
    assert_eq!(0, _arm);
    Some(self.r_forward.as_ref().unwrap().in_r_act.clone())*/
    unimplemented!();
  }

  fn get_output_r_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    unimplemented!();
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    let out_length = self.config.out_dims.len();
    let mut out_act = self.out_act.borrow_mut().as_ref_mut(ctx);

    out_act.copy(&self.in_acts[0].borrow_mut().as_ref(ctx));
    for arm in 1 .. self.config.num_in_arms {
      out_act.vector_add(1.0, &self.in_acts[arm].borrow_mut().as_ref(ctx), 1.0);
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
    let ctx = &(*self.context).as_ref();
    let out_length = self.config.out_dims.len();
    let mut backward = self.backward.as_mut().unwrap();

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        let out_act = self.out_act.borrow_mut().as_ref(ctx);
        let mut out_delta = backward.out_delta.borrow_mut().as_ref_mut(ctx);
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

    let out_delta = backward.out_delta.borrow_mut().as_ref(ctx);
    for arm in 0 .. self.config.num_in_arms {
      if let Some(ref mut in_delta) = backward.in_deltas[arm] {
        in_delta.borrow_mut().as_ref_mut(ctx).copy(&out_delta);
      }
    }
  }

  fn r_forward(&mut self, batch_size: usize) {
    assert!(self.r_forward.is_some());
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    let mut r_forward = self.r_forward.as_mut().unwrap();
    let mut out_r_act = r_forward.out_r_act.borrow_mut().as_ref_mut(ctx);

    // FIXME(20160602): activation function.
    unimplemented!();

    out_r_act.copy(&r_forward.in_r_acts[0].borrow_mut().as_ref(ctx));
    for arm in 1 .. self.config.num_in_arms {
      out_r_act.vector_add(1.0, &r_forward.in_r_acts[arm].borrow_mut().as_ref(ctx), 1.0);
    }
  }
}

pub struct CatJoinOperator;

pub struct BatchJoinOperator;

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
      in_act:       prev_op.unwrap().get_output_act(0),
      in_delta:     prev_op.unwrap().get_output_delta(0),
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

  fn get_output_act(&self, _arm: usize) -> SharedDeviceBuf<f32> {
    assert_eq!(0, _arm);
    self.out_act.clone()
  }

  fn get_output_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    assert_eq!(0, _arm);
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
}
