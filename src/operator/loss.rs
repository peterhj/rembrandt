use data_new::{SampleLabel};
use operator::{Operator, LossOperator, SharedDeviceBuf, OpCapability, OpPhase};

use array::{
  Array, AsyncArray, ArrayView, ArrayViewMut, ArrayZeroExt, NdArraySerialize,
  Shape, Array2d,
};
use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::context::{DeviceContext};
use array_cuda::device::ext::{DeviceNumExt};
use array_cuda::device::memory::{DeviceBufferInitExt, DeviceBuffer};
use cuda_dnn::v4::{
  CudnnSoftmaxOp,
  CudnnTensorDesc, //CudnnFilterDesc, CudnnConvDesc,
};
use rembrandt_kernels::ffi::*;

use std::cell::{RefCell};
use std::iter::{repeat};
use std::rc::{Rc};

#[derive(Clone, Copy, Debug)]
pub struct CategoricalLossConfig {
  pub num_categories:   usize,
}

pub struct SoftmaxKLLossOperator {
  batch_cap:    usize,
  loss_config:  CategoricalLossConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     SharedDeviceBuf<f32>,
  logit:        DeviceBuffer<f32>,
  logit_max:    DeviceBuffer<f32>,
  factor:       DeviceBuffer<f32>,
  factor_sum:   DeviceBuffer<f32>,
  out_act:      DeviceBuffer<f32>,
  out_loss:     DeviceBuffer<f32>,

  label_cats:   DeviceBuffer<i32>,
  label_cats_h: Vec<i32>,
  weights:      DeviceBuffer<f32>,
  weights_h:    Vec<f32>,
  targets:      DeviceBuffer<f32>,
  r_weights:    DeviceBuffer<f32>,
  //r_weights_h:  Vec<f32>,

  out_cats:     DeviceBuffer<i32>,
  out_cats_h:   Vec<i32>,
  out_loss0:    DeviceBuffer<f32>,
  out_loss_h:   Vec<f32>,

  //softmax:      CudnnSoftmaxOp,

  backward:     Option<SoftmaxKLLossBwdOperator>,
  r_forward:    Option<SoftmaxKLLossRFwdOperator>,
  //r_backward:   Option<SoftmaxKLLossRBwdOperator>,
}

pub struct SoftmaxKLLossBwdOperator {
  //in_delta:     SharedDeviceBuf<f32>,
}

pub struct SoftmaxKLLossRFwdOperator {
  in_r_act:     SharedDeviceBuf<f32>,
  mix_in_r_act: DeviceBuffer<f32>,
  out_r_act:    DeviceBuffer<f32>,
  out_r_loss:   DeviceBuffer<f32>,
}

impl SoftmaxKLLossOperator {
  pub fn new(batch_size: usize, capability: OpCapability, loss_config: CategoricalLossConfig, prev_op: Option<&Operator>, context: Rc<DeviceContext>) -> SoftmaxKLLossOperator {
    /*let softmax = CudnnSoftmaxOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, loss_config.num_categories, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, loss_config.num_categories, batch_size).unwrap(),
    );*/

    let ctx = &(*context).as_ref();

    //let mut targets = DeviceBuffer::zeros(batch_size, ctx);
    //targets.as_ref_mut(ctx).set_constant(1.0);

    let backward = if capability.backward_enabled() {
      Some(SoftmaxKLLossBwdOperator{
      })
    } else {
      None
    };

    let r_forward = if capability.r_forward_enabled() {
      Some(SoftmaxKLLossRFwdOperator{
        in_r_act:       Rc::new(RefCell::new(DeviceBuffer::zeros(loss_config.num_categories * batch_size, ctx))),
        mix_in_r_act:   DeviceBuffer::zeros(loss_config.num_categories * batch_size, ctx),
        out_r_act:      DeviceBuffer::zeros(loss_config.num_categories * batch_size, ctx),
        out_r_loss:     DeviceBuffer::zeros(batch_size, ctx),
      })
    } else {
      None
    };

    SoftmaxKLLossOperator{
      batch_cap:    batch_size,
      loss_config:  loss_config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("SoftmaxKLLossOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas().unwrap(),
      logit:        DeviceBuffer::zeros(loss_config.num_categories * batch_size, ctx),
      logit_max:    DeviceBuffer::zeros(batch_size, ctx),
      factor:       DeviceBuffer::zeros(loss_config.num_categories * batch_size, ctx),
      factor_sum:   DeviceBuffer::zeros(batch_size, ctx),
      out_act:      DeviceBuffer::zeros(loss_config.num_categories * batch_size, ctx),
      out_loss:     DeviceBuffer::zeros(batch_size, ctx),
      label_cats:   DeviceBuffer::zeros(batch_size, ctx),
      label_cats_h: repeat(0).take(batch_size).collect(),
      weights:      DeviceBuffer::zeros(batch_size, ctx),
      weights_h:    repeat(0.0).take(batch_size).collect(),
      targets:      DeviceBuffer::ones(batch_size, ctx),
      r_weights:    DeviceBuffer::ones(batch_size, ctx),
      out_cats:     DeviceBuffer::zeros(batch_size, ctx),
      out_cats_h:   repeat(0).take(batch_size).collect(),
      out_loss0:    DeviceBuffer::zeros(1, ctx),
      out_loss_h:   vec![0.0],
      //softmax:      softmax,
      backward:     backward,
      r_forward:    r_forward,
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
    /*self.softmax.set_batch_size(batch_size).unwrap();
    unsafe { self.softmax.forward(
        self.in_act.borrow_mut().as_ref(ctx).as_ptr(),
        self.out_values.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) }.unwrap();*/
    assert!(self.loss_config.num_categories <= 1024);
    self.logit.as_ref_mut(ctx).copy(&self.in_act.borrow_mut().as_ref(ctx));
    unsafe { rembrandt_kernel_batch_blockreduce_argmax(
        self.logit.as_ref(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.logit_max.as_ref_mut(ctx).as_mut_ptr(),
        self.out_cats.as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_scalar_sub_map_batch_inplace(
        self.logit.as_ref_mut(ctx).as_mut_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.logit_max.as_ref(ctx).as_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_exp_map(
        self.logit.as_ref(ctx).as_ptr(),
        (self.loss_config.num_categories * batch_size) as i32,
        self.factor.as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_batch_blockreduce_sum(
        self.factor.as_ref(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.factor_sum.as_ref_mut(ctx).as_mut_ptr(),
        0.0,
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_scalar_div_map_batch(
        self.factor.as_ref(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.factor_sum.as_ref(ctx).as_ptr(),
        self.out_act.as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_softmax_kl_loss_fwd_batch(
        self.out_act.as_ref(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.label_cats.as_ref(ctx).as_ptr(),
        self.weights.as_ref(ctx).as_ptr(),
        self.targets.as_ref(ctx).as_ptr(),
        self.out_loss.as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }

  fn backward(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    /*unsafe { rembrandt_kernel_batch_map_softmax_kl_backward(
        self.out_values.as_view(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.label_cats.as_ref(ctx).as_ptr(),
        self.weights.as_ref(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };*/
    unsafe { rembrandt_kernel_softmax_kl_loss_bwd_batch(
        self.out_act.as_ref(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.label_cats.as_ref(ctx).as_ptr(),
        self.weights.as_ref(ctx).as_ptr(),
        self.targets.as_ref(ctx).as_ptr(),
        self.in_delta.borrow_mut().as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }

  fn r_forward(&mut self, batch_size: usize) {
    assert!(self.r_forward.is_some());
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    let mut r_forward = self.r_forward.as_mut().unwrap();
    unsafe { rembrandt_kernel_inner_prod_blockreduce_batch(
        r_forward.in_r_act.borrow_mut().as_ref(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.factor.as_ref(ctx).as_ptr(),
        0.0,
        r_forward.mix_in_r_act.as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_div_map_inplace(
        r_forward.mix_in_r_act.as_ref_mut(ctx).as_mut_ptr(),
        batch_size as i32,
        self.factor_sum.as_ref(ctx).as_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_softmax_r_fwd_batch(
        self.out_act.as_ref(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        r_forward.in_r_act.borrow_mut().as_ref(ctx).as_ptr(),
        r_forward.mix_in_r_act.as_ref(ctx).as_ptr(),
        r_forward.out_r_act.as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_softmax_kl_loss_r_fwd_batch(
        self.out_act.as_ref(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        r_forward.out_r_act.as_ref(ctx).as_ptr(),
        self.label_cats.as_ref(ctx).as_ptr(),
        self.r_weights.as_ref(ctx).as_ptr(),
        r_forward.out_r_loss.as_ref_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };
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
    self.label_cats.as_ref_mut(ctx)
      .sync_load(&self.label_cats_h);
  }

  fn stage_weight(&mut self, batch_idx: usize, weight: f32) {
    self.weights_h.as_mut_slice()[batch_idx] = weight;
  }

  fn load_weights(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    self.weights.as_ref_mut(ctx)
      .sync_load(&self.weights_h);
  }

  fn reset_targets(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    self.targets.as_ref_mut(ctx)
      .set_constant(1.0);
  }

  fn set_targets_with_r_loss(&mut self, batch_size: usize) {
    assert!(self.r_forward.is_some());
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    self.targets.as_ref_mut(ctx)
      .copy(&self.r_forward.as_mut().unwrap().out_r_loss.as_ref(ctx));
  }

  fn store_loss(&mut self, batch_size: usize) -> f32 {
    // FIXME(20160423)
    assert!(batch_size <= self.batch_cap);
    assert!(self.loss_config.num_categories <= 1024);
    let ctx = &(*self.context).as_ref();
    /*unsafe { rembrandt_kernel_batch_map_softmax_kl_loss1(
        self.out_values.as_view(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.label_cats.as_ref(ctx).as_ptr(),
        self.weights.as_ref(ctx).as_ptr(),
        0.0,
        self.out_loss1.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };*/
    assert!(batch_size <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_sum(
        self.out_loss.as_ref(ctx).as_ptr(),
        batch_size as i32,
        1,
        self.out_loss0.as_ref_mut(ctx).as_mut_ptr(),
        0.0,
        ctx.stream.ptr,
    ) };
    self.out_loss0.as_ref(ctx)
      .sync_store(&mut self.out_loss_h);
    self.out_loss_h[0]
  }

  fn store_output_values(&mut self, batch_size: usize) {
    // FIXME(20160330)
    unimplemented!();
  }

  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32> {
    // FIXME(20160527)
    //&self.out_values_h
    unimplemented!();
  }

  fn store_output_categories(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    /*assert!(self.loss_config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_blockreduce_argmax(
        self.out_act.as_ref(ctx).as_ptr(),
        self.loss_config.num_categories as i32,
        batch_size as i32,
        self.max_value.as_view_mut(ctx).as_mut_ptr(),
        self.out_cats.as_view_mut(ctx).as_mut_ptr(),
        ctx.stream.ptr,
    ) };*/
    self.out_cats.as_ref(ctx)
      .sync_store(&mut self.out_cats_h);
  }

  //fn get_output_categories(&self, batch_size: usize) -> &Array2d<i32> {
  fn get_output_categories(&self, batch_size: usize) -> &[i32] {
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

  fn store_loss(&mut self, batch_size: usize) -> f32 {
    unimplemented!();
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

  //fn get_output_categories(&self, batch_size: usize) -> &Array2d<i32> {
  fn get_output_categories(&self, batch_size: usize) -> &[i32] {
    unimplemented!();
  }

  fn accuracy_count(&self, batch_size: usize) -> usize {
    unimplemented!();
  }
}
