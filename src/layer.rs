use data::{SampleDatum, SampleLabel};
use opt::{DescentConfig};

use array::{View, MutView, WithZeros, ArrayDeserialize, Array2d};
use async::{AsyncContext, AsyncLoad, AsyncStore, AsyncSend};
use async_cuda::array_num::{DeviceNumExt};
use async_cuda::array_types::{DeviceArray2d, DeviceBuf};
use async_cuda::context::{DeviceContext};
use linalg::blas::{BVector, BMatrix, Transpose};
use rembrandt_kernels::*;

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::fs::{File};
use std::path::{PathBuf};
use std::rc::{Rc};

#[derive(Clone, Copy)]
pub enum LayerInitialization {
  Zeros,
  Identity,
  Normal{std: f32},
}

#[derive(Clone, Copy)]
pub enum ActivationFunction {
  Identity,
  Relu,
  Sigmoid,
  Tanh,
}

pub trait Layer {
  fn output_activation(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>>;
  fn output_delta(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>>;

  fn initialize_params(&mut self, ctx: &DeviceContext) {}
  fn reset_gradients(&mut self, ctx: &DeviceContext) {}
  fn forward(&mut self, ctx: &DeviceContext) {}
  fn backward(&mut self, descent: &DescentConfig, ctx: &DeviceContext) {}
  fn descend(&mut self, descent: &DescentConfig, ctx: &DeviceContext) {}
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
        let length = bytes_view.len();
        self.in_bytes.as_mut_view_3d(dims).sync_load(&bytes_view, ctx);
        unsafe { rembrandt_kernel_image_cast_to_float(
            dims.0 as i32, dims.1 as i32, dims.2 as i32,
            self.in_bytes.as_view().as_ptr(),
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
        // FIXME(20150921): subtract out data-specific average.
        /*unsafe { rembrandt_kernel_map_add_constant_float(
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            length as i32,
            -33.318,
            ctx.stream.ptr,
        ) };*/
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
  pub act_fun:      ActivationFunction,
}

impl InnerProdLayer {
  pub fn new(prev_layer: Option<&Layer>, in_channels: usize, out_channels: usize, act_fun: ActivationFunction) -> InnerProdLayer {
    InnerProdLayer{
      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta().unwrap(),
      out_act:      Rc::new(RefCell::new(DeviceBuf::with_zeros(out_channels))),
      out_delta:    Rc::new(RefCell::new(DeviceBuf::with_zeros(out_channels))),
      weights:      DeviceArray2d::with_zeros((in_channels, out_channels)),
      bias:         DeviceArray2d::with_zeros((1, out_channels)),
      grad_weights: DeviceArray2d::with_zeros((in_channels, out_channels)),
      grad_bias:    DeviceArray2d::with_zeros((1, out_channels)),
      act_fun:      act_fun,
    }
  }

  pub fn load_params(&mut self, prefix: &str, ctx: &DeviceContext) {
    let (in_channels, out_channels) = self.weights.as_view().get_bound();
    let weights_path = PathBuf::from(&format!("{}_weights.ndarray", prefix));
    let bias_path = PathBuf::from(&format!("{}_bias.ndarray", prefix));
    let mut weights_file = File::open(&weights_path)
      .ok().expect("failed to open weights!");
    let mut bias_file = File::open(&bias_path)
      .ok().expect("failed to open bias!");
    let init_weights = Array2d::deserialize(&mut weights_file)
      .ok().expect("failed to deserialize array!");
    let init_bias = Array2d::deserialize(&mut bias_file)
      .ok().expect("failed to deserialize array!");
    assert_eq!((in_channels, out_channels), init_weights.as_view().get_bound());
    assert_eq!((1, out_channels), init_bias.as_view().get_bound());
    self.weights.as_mut_view().sync_load(&init_weights.as_view(), ctx);
    self.bias.as_mut_view().sync_load(&init_bias.as_view(), ctx);
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
    let (in_channels, out_channels) = self.weights.as_view().get_bound();
    let mut rng = thread_rng();
    //let xavier_bound = 2.4495 / ((in_channels + out_channels) as f32).sqrt();
    //let dist = Range::new(-xavier_bound, xavier_bound);
    let dist = Normal::new(0.0, 0.01);
    let mut init_weights = Array2d::with_zeros((in_channels, out_channels));
    for w in init_weights.as_mut_view().as_mut_slice().iter_mut() {
      *w = dist.ind_sample(&mut rng) as f32;
    }
    let init_bias = Array2d::with_zeros((1, out_channels));
    self.weights.as_mut_view().sync_load(&init_weights.as_view(), ctx);
    self.bias.as_mut_view().sync_load(&init_bias.as_view(), ctx);
  }

  fn reset_gradients(&mut self, ctx: &DeviceContext) {
    //self.grad_weights.as_mut_view().set_scalar(0.0, ctx);
    //self.grad_bias.as_mut_view().set_scalar(0.0, ctx);
    self.grad_weights.as_mut_view().matrix_scale(0.0, ctx);
    self.grad_bias.as_mut_view().row_vector_scale(0.0, ctx);
  }

  fn forward(&mut self, ctx: &DeviceContext) {
    let (in_channels, out_channels) = self.weights.as_view().get_bound();
    {
      self.out_act.borrow_mut().as_mut_view_2d((1, out_channels)).matrix_prod(
          1.0,
          &self.in_act.borrow().as_view_2d((in_channels, 1)), Transpose::T,
          &self.weights.as_view(), Transpose::N,
          0.0,
          ctx);
      self.out_act.borrow_mut().as_mut_view_2d((1, out_channels)).row_vector_sum(
          1.0,
          &self.bias.as_view(),
          ctx);
      // TODO(20150918): update output activation w/ activation function.
      match self.act_fun {
        ActivationFunction::Identity => {}
        ActivationFunction::Relu => {
          unsafe { rembrandt_kernel_map_relu_activation(
              out_channels as i32,
              self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
        ActivationFunction::Sigmoid => {
          unsafe { rembrandt_kernel_map_sigmoid_activation(
              out_channels as i32,
              self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
        ActivationFunction::Tanh => {
          unsafe { rembrandt_kernel_map_tanh_activation(
              out_channels as i32,
              self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
      }
    }
  }

  fn backward(&mut self, descent: &DescentConfig, ctx: &DeviceContext) {
    let (in_channels, out_channels) = self.weights.as_view().get_bound();
    {
      // TODO(20150918): update output delta w/ activation derivative.
      match self.act_fun {
        ActivationFunction::Identity => {}
        ActivationFunction::Relu => {
          unsafe { rembrandt_kernel_map_relu_activation_backprop(
              out_channels as i32,
              self.out_act.borrow().as_view().as_ptr(),
              self.out_delta.borrow_mut().as_mut_view().as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
        ActivationFunction::Sigmoid => {
          unsafe { rembrandt_kernel_map_sigmoid_activation_backprop(
              out_channels as i32,
              self.out_act.borrow().as_view().as_ptr(),
              self.out_delta.borrow_mut().as_mut_view().as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
        ActivationFunction::Tanh => {
          unsafe { rembrandt_kernel_map_tanh_activation_backprop(
              out_channels as i32,
              self.out_act.borrow().as_view().as_ptr(),
              self.out_delta.borrow_mut().as_mut_view().as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
      }
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
      // TODO: update input delta, if prev layer is not a backprop sink.
      self.in_delta.borrow_mut().as_mut_view_2d((in_channels, 1)).matrix_prod(
          1.0,
          &self.weights.as_view(), Transpose::N,
          &self.out_delta.borrow().as_view_2d((out_channels, 1)), Transpose::N,
          0.0,
          ctx);
    }
  }

  fn descend(&mut self, descent: &DescentConfig, ctx: &DeviceContext) {
    self.weights.as_mut_view().matrix_sum(-descent.step_size, &self.grad_weights.as_view(), ctx);
    self.bias.as_mut_view().matrix_sum(-descent.step_size, &self.grad_bias.as_view(), ctx);
  }
}

pub struct ConvLayer;

impl ConvLayer {
}

impl Layer for ConvLayer {
  // FIXME(20150920)
  fn output_activation(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> { None }
  fn output_delta(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> { None }
}

pub struct SoftmaxLossLayer {
  pub in_act:         Rc<RefCell<DeviceBuf<f32>>>,
  pub in_delta:       Rc<RefCell<DeviceBuf<f32>>>,
  pub probabilities:  DeviceArray2d<f32>,
  pub normalization:  DeviceArray2d<f32>,
  pub tmp_max:        DeviceArray2d<f32>,
  pub category_guess: DeviceArray2d<i32>,
  pub category_truth: Option<i32>,
  pub num_categories: usize,
}

impl SoftmaxLossLayer {
  pub fn new(prev_layer: Option<&Layer>, num_categories: usize) -> SoftmaxLossLayer {
    SoftmaxLossLayer{
      in_act:         prev_layer.unwrap().output_activation().unwrap(),
      in_delta:       prev_layer.unwrap().output_delta().unwrap(),
      probabilities:  DeviceArray2d::with_zeros((num_categories, 1)),
      normalization:  DeviceArray2d::with_zeros((1, 1)),
      tmp_max:        DeviceArray2d::with_zeros((1, 1)),
      category_guess: DeviceArray2d::with_zeros((1, 1)),
      category_truth: None,
      num_categories: num_categories,
    }
  }

  pub fn load_sample(&mut self, maybe_label: Option<SampleLabel>, ctx: &DeviceContext) {
    self.category_truth = maybe_label.map(|label| label.0);
  }

  pub fn get_guess(&self, ctx: &DeviceContext) -> i32 {
    let mut host_guess = Array2d::with_zeros((1, 1));
    self.category_guess.as_view().sync_store(&mut host_guess.as_mut_view(), &ctx);
    host_guess.as_view().as_slice()[0]
  }

  pub fn correct_guess(&self, ctx: &DeviceContext) -> bool {
    let guess = self.get_guess(ctx);
    //println!("DEBUG: truth: {} guess: {}", unwrap!(self.category_truth), guess);
    unwrap!(self.category_truth) == guess
  }
}

impl Layer for SoftmaxLossLayer {
  fn output_activation(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> { None }
  fn output_delta(&self) -> Option<Rc<RefCell<DeviceBuf<f32>>>> { None }

  fn forward(&mut self, ctx: &DeviceContext) {
    ctx.synchronize();
    self.in_act.borrow().as_view_2d((self.num_categories, 1)).send(&mut self.probabilities.as_mut_view(), ctx);
    ctx.synchronize();
    unsafe { rembrandt_kernel_blockreduce_argmax(
        self.num_categories as i32,
        self.probabilities.as_view().as_ptr(),
        self.tmp_max.as_mut_view().as_mut_ptr(),
        self.category_guess.as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_map_subtract_scalar(
        self.probabilities.as_mut_view().as_mut_ptr(),
        self.num_categories as i32,
        self.tmp_max.as_view().as_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_map_exp(
        self.probabilities.as_mut_view().as_mut_ptr(),
        self.num_categories as i32,
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_blockreduce_sum(
        self.num_categories as i32,
        self.probabilities.as_view().as_ptr(),
        self.normalization.as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    unsafe { rembrandt_kernel_map_divide_scalar(
        self.probabilities.as_mut_view().as_mut_ptr(),
        self.num_categories as i32,
        self.normalization.as_view().as_ptr(),
        ctx.stream.ptr,
    ) };
  }

  fn backward(&mut self, descent: &DescentConfig, ctx: &DeviceContext) {
    //self.probabilities.as_view().send(&mut self.in_delta.borrow_mut().as_mut_view_2d((self.num_categories, 1)), ctx);
    //ctx.synchronize();
    unsafe { rembrandt_kernel_map_softmax_cross_entropy_loss_backprop(
        self.probabilities.as_view().as_ptr(),
        self.num_categories as i32,
        unwrap!(self.category_truth),
        self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.in_delta.borrow_mut().as_mut_view_2d((1, self.num_categories)).row_vector_scale(
        1.0 / (descent.minibatch_size as f32),
        ctx,
    );
  }
}
