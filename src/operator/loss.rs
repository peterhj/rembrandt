use data_new::{SampleLabel};
use operator::{Operator, SharedDeviceBuf, OpPhase};

use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceContext};
use array_cuda::device::memory::{DeviceZeroExt, DeviceBuffer};
use array_new::{
  Array, AsyncArray, ArrayView, ArrayViewMut, ArrayZeroExt, NdArraySerialize,
  Shape, Array2d,
};
use cuda_dnn::v4::{
  CudnnSoftmaxOp,
  CudnnTensorDesc, //CudnnFilterDesc, CudnnConvDesc,
};
use rembrandt_kernels::ffi::*;

use std::cell::{RefCell};
use std::rc::{Rc};

pub trait LossOperator: Operator {
  fn downcast(&self) -> &Operator;
  fn stage_label(&mut self, batch_idx: usize, label: &SampleLabel);
  fn load_labels(&mut self, batch_size: usize);
  fn stage_weight(&mut self, batch_idx: usize, weight: f32);
  fn load_weights(&mut self, batch_size: usize);
  fn stage_category_weight(&mut self, _batch_idx: usize, _category: i32, _cat_weight: f32) {}
  fn load_category_weights(&mut self, _batch_size: usize) {}
  fn store_output_values(&mut self, batch_size: usize);
  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32>;
  fn store_output_categories(&mut self, batch_size: usize);
  fn get_output_categories(&self, batch_size: usize) -> &Array2d<i32>;
  fn accuracy_count(&self, batch_size: usize) -> usize;
  //fn reset_loss(&mut self);
}

#[derive(Clone, Copy)]
pub struct CategoricalLossConfig {
  pub num_categories:   usize,
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
        CudnnTensorDesc::<f32>::create_4d(1, 1, loss_config.num_categories, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, loss_config.num_categories, batch_size).unwrap(),
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
      out_values:   DeviceArray2d::<f32>::zeros((loss_config.num_categories, batch_size), ctx),
      out_values_h: Array2d::zeros((loss_config.num_categories, batch_size)),
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
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
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
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.label_cats.as_view(ctx).as_ptr(),
        self.weights.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }

  //fn update_params(&mut self, _step_size: f32, _l2_reg_coef: f32) {
  fn update_params(&mut self, _momentum: f32, _nesterov: bool) {
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
    // FIXME(20160330)
    unimplemented!();
  }

  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32> {
    &self.out_values_h
  }

  fn store_output_categories(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    assert!(self.loss_config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_argmax(
        self.out_values.as_view(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
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

  fn accuracy_count(&self, batch_size: usize) -> usize {
    assert!(batch_size <= self.batch_cap);
    let mut correct_count = 0;
    //print!("DEBUG: accuracy count: ");
    for (&y_label, &y_hat) in self.label_cats_h.as_slice().iter()
      .zip(self.out_cats_h.as_slice().iter())
      .take(batch_size)
    {
      //print!("({} {}) ", y_label, y_hat);
      if y_label == y_hat {
        correct_count += 1;
      }
    }
    //println!("");
    correct_count
  }
}

pub struct MarginalizedSoftmaxIndLossOperator {
  batch_cap:    usize,
  loss_config:  CategoricalLossConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     SharedDeviceBuf<f32>,

  /*label_cats:   DeviceArray2d<i32>,
  label_cats_h: Array2d<i32>,*/
  weights:      DeviceArray2d<f32>,
  weights_h:    Array2d<f32>,
  c_weights:    DeviceArray2d<f32>,
  c_weights_h:  Array2d<f32>,

  out_values:   DeviceArray2d<f32>,
  out_values_h: Array2d<f32>,
  /*max_value:    DeviceArray2d<f32>,
  out_cats:     DeviceArray2d<i32>,
  out_cats_h:   Array2d<i32>,*/
  /*out_loss1:    DeviceArray2d<f32>,
  out_loss:     DeviceBuffer<f32>,
  out_loss_h:   Vec<f32>,*/

  softmax:      CudnnSoftmaxOp,
}

impl MarginalizedSoftmaxIndLossOperator {
}

impl Operator for MarginalizedSoftmaxIndLossOperator {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
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
    unsafe { rembrandt_kernel_batch_map_marginalized_softmax_ind_backward(
        self.out_values.as_view(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.weights.as_view(ctx).as_ptr(),
        self.c_weights.as_view(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }

  //fn update_params(&mut self, _step_size: f32, _l2_reg_coef: f32) {
  fn update_params(&mut self, _momentum: f32, _nesterov: bool) {
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

impl LossOperator for MarginalizedSoftmaxIndLossOperator {
  fn downcast(&self) -> &Operator {
    self as &Operator
  }

  fn stage_label(&mut self, batch_idx: usize, label: &SampleLabel) {
    unimplemented!();
  }

  fn load_labels(&mut self, batch_size: usize) {
    unimplemented!();
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

  fn stage_category_weight(&mut self, batch_idx: usize, category: i32, cat_weight: f32) {
    assert!(batch_idx < self.batch_cap);
    assert!(category >= 0);
    assert!(category < self.loss_config.num_categories as i32);
    self.c_weights_h.as_mut_slice()[batch_idx * self.loss_config.num_categories + category as usize] = cat_weight;
  }

  fn load_category_weights(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    self.c_weights.as_view_mut(ctx)
      .sync_load(&self.c_weights_h.as_view());
  }

  fn store_output_values(&mut self, batch_size: usize) {
    unimplemented!();
  }

  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32> {
    unimplemented!();
  }

  fn store_output_categories(&mut self, batch_size: usize) {
    unimplemented!();
  }

  fn get_output_categories(&self, batch_size: usize) -> &Array2d<i32> {
    unimplemented!();
  }

  fn accuracy_count(&self, batch_size: usize) -> usize {
    unimplemented!();
  }
}
