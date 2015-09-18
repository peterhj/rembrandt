use data::{SampleDatum, SampleLabel};

use array::{View, MutView, WithZeros, Array2d};
use async::{AsyncContext, AsyncLoad, AsyncStore, AsyncSend};
use async_cuda::array_num::{DeviceNumExt};
use async_cuda::array_types::{DeviceArray2d, DeviceBuf};
use async_cuda::context::{DeviceContext};
use linalg::blas::{BVector, BMatrix, Transpose};
use rembrandt_kernels::*;

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use std::cell::{RefCell};
use std::rc::{Rc};

pub trait Layer {
  fn output_activation(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>>;
  fn output_delta(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>>;

  fn initialize_params(&mut self, ctx: &DeviceContext) {}
  fn reset_gradients(&mut self, ctx: &DeviceContext) {}
  fn forward(&mut self, ctx: &DeviceContext) {}
  fn backward(&mut self, ctx: &DeviceContext) {}
  fn update_params(&mut self, learning_rate: f32, ctx: &DeviceContext) {}
}

/*pub enum LayerWrapper {
  Data(DataLayer),
  InnerProd(InnerProdLayer),
  SoftmaxLoss(SoftmaxLossLayer),
}

impl Layer for LayerWrapper {
}*/

pub struct DataLayer {
  pub in_bytes:   DeviceBuf<u8>,
  pub out_act:    Rc<RefCell<DeviceBuf<f32>>>,
  pub out_delta:  Rc<RefCell<DeviceBuf<f32>>>, // XXX: unused.
}

impl DataLayer {
  pub fn new(length: usize) -> DataLayer {
    DataLayer{
      in_bytes:   DeviceBuf::with_zeros(length),
      out_act:    Rc::new(RefCell::new(DeviceBuf::with_zeros(length))),
      out_delta:  Rc::new(RefCell::new(DeviceBuf::with_zeros(length))),
    }
  }

  pub fn load_sample(&mut self, datum: &SampleDatum, ctx: &DeviceContext) {
    ctx.synchronize();
    match datum {
      &SampleDatum::RgbPerChannelBytes(ref bytes) => {
        let bytes_view = bytes.as_view();
        let dims = bytes_view.get_bound();
        self.in_bytes.as_mut_view_3d(dims).sync_load(&bytes_view, ctx);
        unsafe { rembrandt_kernel_image_cast_to_float(
            dims.0 as i32, dims.1 as i32, dims.2 as i32,
            self.in_bytes.as_view().as_ptr(),
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }
  }
}

impl Layer for DataLayer {
  fn output_activation(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> {
    Some(self.out_delta.clone())
  }
}

pub struct InnerProdLayer {
  pub in_act:       Rc<RefCell<DeviceBuf<f32>>>,
  pub in_delta:     Rc<RefCell<DeviceBuf<f32>>>,
  pub out_act:      Rc<RefCell<DeviceBuf<f32>>>,
  pub out_delta:    Rc<RefCell<DeviceBuf<f32>>>,
  pub weights:      DeviceArray2d<f32>,
  pub bias:         DeviceArray2d<f32>,
  pub grad_weights: DeviceArray2d<f32>,
  pub grad_bias:    DeviceArray2d<f32>,
}

impl InnerProdLayer {
  pub fn new(prev_layer: Option<&Layer>, in_channels: usize, out_channels: usize) -> InnerProdLayer {
    InnerProdLayer{
      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta().unwrap(),
      out_act:      Rc::new(RefCell::new(DeviceBuf::with_zeros(out_channels))),
      out_delta:    Rc::new(RefCell::new(DeviceBuf::with_zeros(out_channels))),
      weights:      DeviceArray2d::with_zeros((in_channels, out_channels)),
      bias:         DeviceArray2d::with_zeros((1, out_channels)),
      grad_weights: DeviceArray2d::with_zeros((in_channels, out_channels)),
      grad_bias:    DeviceArray2d::with_zeros((1, out_channels)),
    }
  }
}

impl Layer for InnerProdLayer {
  fn output_activation(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> {
    Some(self.out_delta.clone())
  }

  fn initialize_params(&mut self, ctx: &DeviceContext) {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 0.001);
    let (in_channels, out_channels) = self.weights.as_view().get_bound();
    let mut init_weights = Array2d::with_zeros((in_channels, out_channels));
    {
      let mut init_weights_view = init_weights.as_mut_view();
      let _ = init_weights_view.as_mut_slice().iter_mut()
        .map(|w| *w = normal.ind_sample(&mut rng) as f32);
    }
    let init_bias = Array2d::with_zeros((1, out_channels));
    self.weights.as_mut_view().sync_load(&init_weights.as_view(), ctx);
    self.bias.as_mut_view().sync_load(&init_bias.as_view(), ctx);
  }

  fn reset_gradients(&mut self, ctx: &DeviceContext) {
    self.grad_weights.as_mut_view().set_scalar(0.0, ctx);
    self.grad_bias.as_mut_view().set_scalar(0.0, ctx);
  }

  fn forward(&mut self, ctx: &DeviceContext) {
    let (in_channels, out_channels) = self.weights.as_view().get_bound();
    {
      self.out_act.borrow_mut().as_mut_view_2d((1, out_channels)).matrix_prod(
          1.0,
          &self.in_act.borrow().as_view_2d((1, in_channels)), Transpose::N,
          &self.weights.as_view(), Transpose::N,
          0.0,
          ctx);
    }
    {
      self.out_act.borrow_mut().as_mut_view_2d((1, out_channels)).row_vector_sum(
          1.0,
          &self.bias.as_view(),
          ctx);
    }
  }

  fn backward(&mut self, ctx: &DeviceContext) {
    let (in_channels, out_channels) = self.weights.as_view().get_bound();
    {
      self.grad_weights.as_mut_view().matrix_prod(
          1.0,
          &self.in_act.borrow().as_view_2d((in_channels, 1)), Transpose::N,
          &self.out_delta.borrow().as_view_2d((1, out_channels)), Transpose::N,
          1.0,
          ctx);
      self.grad_bias.as_mut_view().row_vector_sum(
          1.0,
          &self.out_delta.borrow().as_view_2d((1, out_channels)),
          ctx);
    }

    // TODO: update input delta, if prev layer is not a backprop sink.
  }

  fn update_params(&mut self, learning_rate: f32, ctx: &DeviceContext) {
    self.weights.as_mut_view().matrix_sum(-learning_rate, &self.grad_weights.as_view(), ctx);
    self.bias.as_mut_view().matrix_sum(-learning_rate, &self.grad_bias.as_view(), ctx);
  }
}
  
pub struct SoftmaxLossLayer {
  pub in_act:         Rc<RefCell<DeviceBuf<f32>>>,
  pub in_delta:       Rc<RefCell<DeviceBuf<f32>>>,
  pub probabilities:  DeviceArray2d<f32>,
  pub normalization:  DeviceArray2d<f32>,
  pub category_guess: DeviceArray2d<i32>,
  pub category_true:  Option<i32>,
  pub num_categories: usize,
}

impl SoftmaxLossLayer {
  pub fn new(prev_layer: Option<&Layer>, num_categories: usize) -> SoftmaxLossLayer {
    SoftmaxLossLayer{
      in_act:         prev_layer.unwrap().output_activation().unwrap(),
      in_delta:       prev_layer.unwrap().output_delta().unwrap(),
      probabilities:  DeviceArray2d::with_zeros((num_categories, 1)),
      normalization:  DeviceArray2d::with_zeros((1, 1)),
      category_guess: DeviceArray2d::with_zeros((1, 1)),
      category_true:  None,
      num_categories: num_categories,
    }
  }

  pub fn load_sample(&mut self, maybe_label: Option<SampleLabel>, ctx: &DeviceContext) {
    self.category_true = maybe_label.map(|label| label.0);
  }

  pub fn get_guess(&self, ctx: &DeviceContext) -> i32 {
    let mut host_guess = Array2d::with_zeros((1, 1));
    self.category_guess.as_view().sync_store(&mut host_guess.as_mut_view(), &ctx);
    host_guess.as_view().as_slice()[0]
  }

  pub fn correct_guess(&self, ctx: &DeviceContext) -> bool {
    let guess = self.get_guess(ctx);
    //println!("DEBUG: truth: {} guess: {}", unwrap!(self.category_true), guess);
    unwrap!(self.category_true) == guess
  }
}

impl Layer for SoftmaxLossLayer {
  fn output_activation(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> { None }
  fn output_delta(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> { None }

  fn forward(&mut self, ctx: &DeviceContext) {
    {
      self.in_act.borrow().as_view_2d((self.num_categories, 1)).send(&mut self.probabilities.as_mut_view(), ctx);
    }
    {
      unsafe { rembrandt_kernel_blockreduce_argmax(
          self.num_categories as i32,
          self.probabilities.as_view().as_ptr(),
          self.normalization.as_mut_view().as_mut_ptr(),
          self.category_guess.as_mut_view().as_mut_ptr(),
          ctx.stream.ptr,
      ) };
    }

    {
      unsafe { rembrandt_kernel_map_subtract_scalar(
          self.probabilities.as_mut_view().as_mut_ptr(),
          self.num_categories as i32,
          self.normalization.as_view().as_ptr(),
          ctx.stream.ptr,
      ) };
    }

    {
      unsafe { rembrandt_kernel_map_exp(
          self.probabilities.as_mut_view().as_mut_ptr(),
          self.num_categories as i32,
          ctx.stream.ptr,
      ) };
    }

    {
      unsafe { rembrandt_kernel_blockreduce_sum(
          self.num_categories as i32,
          self.probabilities.as_view().as_ptr(),
          self.normalization.as_mut_view().as_mut_ptr(),
          ctx.stream.ptr,
      ) };
    }

    {
      unsafe { rembrandt_kernel_map_divide_scalar(
          self.probabilities.as_mut_view().as_mut_ptr(),
          self.num_categories as i32,
          self.normalization.as_view().as_ptr(),
          ctx.stream.ptr,
      ) };
    }
  }

  fn backward(&mut self, ctx: &DeviceContext) {
    {
      unsafe { rembrandt_kernel_map_softmax_cross_entropy_loss_backprop(
          self.probabilities.as_view().as_ptr(),
          self.num_categories as i32,
          unwrap!(self.category_true),
          self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
          ctx.stream.ptr,
      ) };
    }
  }
}
