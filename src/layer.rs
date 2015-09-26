use data::{SampleDatum, SampleLabel};
use opt::{DescentConfig, DescentSchedule};

use array::{View, MutView, WithZeros, ArrayDeserialize, Array2d};
use async::{AsyncContext, AsyncLoad, AsyncStore, AsyncSend};
use async_cuda::array_num::{DeviceNumExt};
use async_cuda::array_rand::{DeviceRandExt};
use async_cuda::array_types::{DeviceArray2d, DeviceBuf};
use async_cuda::context::{DeviceContext};
use linalg::blas::{BVector, BMatrix, Transpose};
use rembrandt_kernels::*;

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{max};
use std::fs::{File};
use std::path::{PathBuf};
use std::rc::{Rc};

#[derive(Clone, Copy)]
pub enum LayerInitialization {
  Zeros,
  Normal{std: f32},
  Glorot,
}

#[derive(Clone, Copy)]
pub enum ActivationFunction {
  Identity,
  Rect,
  Sigmoid,
  Tanh,
}

#[derive(Clone, Copy)]
pub enum PoolKind {
  Max,
  Average,
}

pub type SharedDeviceBuf<T> = Rc<RefCell<DeviceBuf<T>>>;

pub trait LayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize);
  fn get_out_dims(&self) -> (usize, usize, usize);
}

pub trait Layer {
  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>>;
  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>>;

  fn initialize_params(&mut self, ctx: &DeviceContext) {}
  fn reset_gradients(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {}
  fn forward(&mut self, ctx: &DeviceContext) {}
  fn backward(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {}
  fn descend(&mut self, descent: &DescentSchedule, t: usize, ctx: &DeviceContext) {}
}

pub trait LossLayer {
  // TODO
}

#[derive(Clone, Copy)]
pub struct DataLayerConfig {
  pub raw_width:    usize,
  pub raw_height:   usize,
  pub crop_width:   usize,
  pub crop_height:  usize,
  pub channels:     usize,
}

impl LayerConfig for DataLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (self.raw_width, self.raw_height, self.channels)
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    (self.crop_width, self.crop_height, self.channels)
  }
}

pub struct DataLayer {
  pub in_bytes:   DeviceBuf<u8>,
  pub raw_image:  DeviceBuf<f32>,
  pub crop_image: SharedDeviceBuf<f32>,
  pub config:     DataLayerConfig,
}

impl DataLayer {
  pub fn new(config: DataLayerConfig) -> DataLayer {
    let raw_length = config.raw_width * config.raw_height * config.channels;
    let crop_length = config.crop_width * config.crop_height * config.channels;
    DataLayer{
      in_bytes:   DeviceBuf::with_zeros(raw_length),
      raw_image:  DeviceBuf::with_zeros(raw_length),
      crop_image: Rc::new(RefCell::new(DeviceBuf::with_zeros(crop_length))),
      config:     config,
    }
  }

  pub fn load_sample(&mut self, datum: &SampleDatum, ctx: &DeviceContext) {
    let DataLayerConfig{
      raw_width, raw_height,
      crop_width, crop_height,
      channels} = self.config;
    match datum {
      &SampleDatum::RgbPerChannelBytes(ref bytes) => {
        let bytes_view = bytes.as_view();
        let dims = bytes_view.get_bound();
        let length = bytes_view.len();
        self.in_bytes.as_mut_view_3d(dims).sync_load(&bytes_view, ctx);
        unsafe { rembrandt_kernel_image_cast_to_float(
            dims.0 as i32, dims.1 as i32, dims.2 as i32,
            self.in_bytes.as_view().as_ptr(),
            //self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            self.raw_image.as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
        // FIXME(20150921): subtract out data-specific average.
        /*unsafe { rembrandt_kernel_map_add_constant_float(
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            length as i32,
            -33.318,
            ctx.stream.ptr,
        ) };*/
        let off_w = (raw_width - crop_width) / 2;
        let off_h = (raw_height - crop_height) / 2;
        for c in (0 .. channels) {
          let (_, back_raw_view) = self.raw_image.as_view().split_at(c * raw_width * raw_height);
          let back_raw_view_2d = back_raw_view.as_view_2d((raw_width, raw_height));
          let cropped_raw_view = back_raw_view_2d.view((off_w, off_h), (off_w + crop_width, off_h + crop_height));
          let mut crop_image = self.crop_image.borrow_mut();
          let mut crop_view = crop_image.as_mut_view();
          let (_, mut back_crop_view) = crop_view.split_at(c * crop_width * crop_height);
          let mut shaped_crop_view = back_crop_view.as_mut_view_2d((crop_width, crop_height));
          cropped_raw_view.send(&mut shaped_crop_view, ctx);
        }
      }
      _ => unimplemented!(),
    }
  }
}

impl Layer for DataLayer {
  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.crop_image.clone())
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }
}

#[derive(Clone, Copy)]
pub struct FullyConnLayerConfig {
  pub in_channels:  usize,
  pub out_channels: usize,
  pub act_fun:      ActivationFunction,
}

impl LayerConfig for FullyConnLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (1, 1, self.in_channels)
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    (1, 1, self.out_channels)
  }
}

pub struct FullyConnLayer {
  pub in_act:       SharedDeviceBuf<f32>,
  pub in_delta:     Option<SharedDeviceBuf<f32>>,
  pub out_act:      SharedDeviceBuf<f32>,
  pub out_delta:    SharedDeviceBuf<f32>,
  pub weights:      DeviceArray2d<f32>,
  pub bias:         DeviceArray2d<f32>,
  pub grad_weights: DeviceArray2d<f32>,
  pub grad_bias:    DeviceArray2d<f32>,
  pub config:       FullyConnLayerConfig,
}

impl FullyConnLayer {
  pub fn new(prev_layer: Option<&Layer>, config: FullyConnLayerConfig) -> FullyConnLayer {
    FullyConnLayer{
      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta(),
      out_act:      Rc::new(RefCell::new(DeviceBuf::with_zeros(config.out_channels))),
      out_delta:    Rc::new(RefCell::new(DeviceBuf::with_zeros(config.out_channels))),
      weights:      DeviceArray2d::with_zeros((config.in_channels, config.out_channels)),
      bias:         DeviceArray2d::with_zeros((1, config.out_channels)),
      grad_weights: DeviceArray2d::with_zeros((config.in_channels, config.out_channels)),
      grad_bias:    DeviceArray2d::with_zeros((1, config.out_channels)),
      config:       config,
    }
  }

  pub fn load_params(&mut self, prefix: &str, ctx: &DeviceContext) {
    let FullyConnLayerConfig{in_channels, out_channels, ..} = self.config;
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

impl Layer for FullyConnLayer {
  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
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

  fn reset_gradients(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    self.grad_weights.as_mut_view().matrix_scale(descent.momentum(), ctx);
    self.grad_bias.as_mut_view().row_vector_scale(descent.momentum(), ctx);
  }

  fn forward(&mut self, ctx: &DeviceContext) {
    let (in_channels, out_channels) = self.weights.as_view().get_bound();
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
    match self.config.act_fun {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
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

  fn backward(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    let (in_channels, out_channels) = self.weights.as_view().get_bound();
    match self.config.act_fun {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
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
    if let Some(ref mut in_delta) = self.in_delta {
      in_delta.borrow_mut().as_mut_view_2d((in_channels, 1)).matrix_prod(
          1.0,
          &self.weights.as_view(), Transpose::N,
          &self.out_delta.borrow().as_view_2d((1, out_channels)), Transpose::T,
          0.0,
          ctx);
    }
  }

  fn descend(&mut self, descent: &DescentSchedule, t: usize, ctx: &DeviceContext) {
    self.weights.as_mut_view().matrix_sum(-descent.step_size(t), &self.grad_weights.as_view(), ctx);
    self.bias.as_mut_view().matrix_sum(-descent.step_size(t), &self.grad_bias.as_view(), ctx);
  }
}

#[derive(Clone, Copy)]
pub struct Conv2dLayerConfig {
  pub in_width:     usize,
  pub in_height:    usize,
  pub in_channels:  usize,
  pub conv_size:    usize,
  pub conv_stride:  usize,
  pub conv_pad:     usize,
  pub out_channels: usize,
  pub act_fun:      ActivationFunction,
}

impl LayerConfig for Conv2dLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (self.in_width, self.in_height, self.in_channels)
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    // XXX(20150926): Note that pool layer uses ceil((L+2*pad-size)/stride)
    // whereas conv layer uses floor((L+2*pad-size)/stride).
    let out_width = max(0, (self.in_width + 2 * self.conv_pad - self.conv_size) as isize) as usize / self.conv_stride + 1;
    let out_height = max(0, (self.in_height + 2 * self.conv_pad - self.conv_size) as isize) as usize / self.conv_stride + 1;
    (out_width, out_height, self.out_channels)
  }
}

pub struct Conv2dLayer {
  pub in_act:       SharedDeviceBuf<f32>,
  pub in_delta:     Option<SharedDeviceBuf<f32>>,
  pub in_col:       DeviceArray2d<f32>,
  pub grad_col:     DeviceArray2d<f32>,
  pub unit:         DeviceArray2d<f32>,
  pub out_act:      SharedDeviceBuf<f32>,
  pub out_delta:    SharedDeviceBuf<f32>,
  pub weights:      DeviceArray2d<f32>,
  pub bias:         DeviceArray2d<f32>,
  pub grad_weights: DeviceArray2d<f32>,
  pub grad_bias:    DeviceArray2d<f32>,
  pub config:       Conv2dLayerConfig,
}

impl Conv2dLayer {
  pub fn new(prev_layer: Option<&Layer>, config: Conv2dLayerConfig, ctx: &DeviceContext) -> Conv2dLayer {
    let Conv2dLayerConfig{
      in_width, in_height, in_channels,
      conv_size, conv_stride, conv_pad,
      out_channels, ..} = config;
    let (out_width, out_height, _) = config.get_out_dims();
    let out_length = out_width * out_height * out_channels;
    let mut unit = DeviceArray2d::with_zeros((out_width * out_height, 1));
    // FIXME: or .col_vector_scale().
    unit.as_mut_view().set_constant(1.0, ctx);
    Conv2dLayer{
      in_act:       prev_layer.unwrap().output_activation().unwrap(),
      in_delta:     prev_layer.unwrap().output_delta(),
      in_col:       DeviceArray2d::with_zeros((out_width * out_height, conv_size * conv_size * in_channels)),
      grad_col:     DeviceArray2d::with_zeros((out_width * out_height, conv_size * conv_size * in_channels)),
      unit:         unit,
      out_act:      Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length))),
      out_delta:    Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length))),
      weights:      DeviceArray2d::with_zeros((conv_size * conv_size * in_channels, out_channels)),
      bias:         DeviceArray2d::with_zeros((1, out_channels)),
      grad_weights: DeviceArray2d::with_zeros((conv_size * conv_size * in_channels, out_channels)),
      grad_bias:    DeviceArray2d::with_zeros((1, out_channels)),
      config:       config,
    }
  }
}

impl Layer for Conv2dLayer {
  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn initialize_params(&mut self, ctx: &DeviceContext) {
    let Conv2dLayerConfig{in_channels, conv_size, out_channels, ..} = self.config;
    let mut rng = thread_rng();
    let dist = Normal::new(0.0, 0.01);
    let mut init_weights = Array2d::with_zeros((conv_size * conv_size * in_channels, out_channels));
    for w in init_weights.as_mut_view().as_mut_slice().iter_mut() {
      *w = dist.ind_sample(&mut rng) as f32;
    }
    let init_bias = Array2d::with_zeros((1, out_channels));
    self.weights.as_mut_view().sync_load(&init_weights.as_view(), ctx);
    self.bias.as_mut_view().sync_load(&init_bias.as_view(), ctx);
  }

  fn reset_gradients(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    self.grad_weights.as_mut_view().matrix_scale(descent.momentum(), ctx);
    self.grad_bias.as_mut_view().row_vector_scale(descent.momentum(), ctx);
  }

  fn forward(&mut self, ctx: &DeviceContext) {
    let Conv2dLayerConfig{
      in_width, in_height, in_channels,
      conv_size, conv_stride, conv_pad,
      out_channels, ..} = self.config;
    let (out_width, out_height, _) = self.config.get_out_dims();
    let out_length = out_width * out_height * out_channels;
    unsafe { rembrandt_kernel_image_im2col(
        self.in_act.borrow().as_view().as_ptr(),
        in_width as i32, in_height as i32, in_channels as i32,
        conv_size as i32, conv_stride as i32, conv_pad as i32,
        self.in_col.as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.out_act.borrow_mut().as_mut_view_2d((out_width * out_height, out_channels)).matrix_prod(
        1.0,
        &self.in_col.as_view(), Transpose::N,
        &self.weights.as_view(), Transpose::N,
        0.0,
        ctx,
    );
    self.out_act.borrow_mut().as_mut_view_2d((out_width * out_height, out_channels)).matrix_prod(
        1.0,
        &self.unit.as_view(), Transpose::N,
        &self.bias.as_view(), Transpose::N,
        1.0,
        ctx,
    );
    match self.config.act_fun {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_map_relu_activation(
            out_length as i32,
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      ActivationFunction::Sigmoid => {
        unsafe { rembrandt_kernel_map_sigmoid_activation(
            out_length as i32,
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      ActivationFunction::Tanh => {
        unsafe { rembrandt_kernel_map_tanh_activation(
            out_length as i32,
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
    }
  }

  fn backward(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    let Conv2dLayerConfig{
      in_width, in_height, in_channels,
      conv_size, conv_stride, conv_pad,
      out_channels, ..} = self.config;
    let (out_width, out_height, _) = self.config.get_out_dims();
    let out_length = out_width * out_height * out_channels;
    match self.config.act_fun {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_map_relu_activation_backprop(
            out_length as i32,
            self.out_act.borrow().as_view().as_ptr(),
            self.out_delta.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      ActivationFunction::Sigmoid => {
        unsafe { rembrandt_kernel_map_sigmoid_activation_backprop(
            out_length as i32,
            self.out_act.borrow().as_view().as_ptr(),
            self.out_delta.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      ActivationFunction::Tanh => {
        unsafe { rembrandt_kernel_map_tanh_activation_backprop(
            out_length as i32,
            self.out_act.borrow().as_view().as_ptr(),
            self.out_delta.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
    }
    self.grad_weights.as_mut_view().matrix_prod(
        1.0,
        &self.in_col.as_view(), Transpose::T,
        &self.out_delta.borrow().as_view_2d((out_width * out_height, out_channels)), Transpose::N,
        1.0,
        ctx,
    );
    self.grad_bias.as_mut_view().matrix_prod(
        1.0,
        &self.unit.as_view(), Transpose::T,
        &self.out_delta.borrow().as_view_2d((out_width * out_height, out_channels)), Transpose::N,
        1.0,
        ctx,
    );
    if let Some(ref mut in_delta) = self.in_delta {
      self.grad_col.as_mut_view().matrix_prod(
          1.0,
          &self.out_delta.borrow().as_view_2d((out_width * out_height, out_channels)), Transpose::N,
          &self.weights.as_view(), Transpose::T,
          0.0,
          ctx,
      );
      unsafe { rembrandt_kernel_image_col2im(
          self.grad_col.as_view().as_ptr(),
          in_width as i32, in_height as i32, in_channels as i32,
          conv_size as i32, conv_stride as i32, conv_pad as i32,
          in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
          ctx.stream.ptr,
      ) };
    }
  }

  fn descend(&mut self, descent: &DescentSchedule, t: usize, ctx: &DeviceContext) {
    self.weights.as_mut_view().matrix_sum(-descent.step_size(t), &self.grad_weights.as_view(), ctx);
    self.bias.as_mut_view().matrix_sum(-descent.step_size(t), &self.grad_bias.as_view(), ctx);
  }
}

#[derive(Clone, Copy)]
pub struct PoolLayerConfig {
  pub in_width:     usize,
  pub in_height:    usize,
  pub channels:     usize,
  pub pool_size:    usize,
  pub pool_stride:  usize,
  pub pool_pad:     usize,
  pub pool_kind:    PoolKind,
}

impl LayerConfig for PoolLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (self.in_width, self.in_height, self.channels)
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    // XXX(20150926): Note that pool layer uses ceil((L+2*pad-size)/stride)
    // whereas conv layer uses floor((L+2*pad-size)/stride).
    let out_width = max(0, (self.in_width + 2 * self.pool_pad - self.pool_size + self.pool_stride - 1) as isize) as usize / self.pool_stride + 1;
    let out_height = max(0, (self.in_height + 2 * self.pool_pad - self.pool_size + self.pool_stride - 1) as isize) as usize / self.pool_stride + 1;
    (out_width, out_height, self.channels)
  }
}

pub struct PoolLayer {
  pub in_act:     SharedDeviceBuf<f32>,
  pub in_delta:   SharedDeviceBuf<f32>,
  pub in_mask:    DeviceBuf<i32>,
  pub out_act:    SharedDeviceBuf<f32>,
  pub out_delta:  SharedDeviceBuf<f32>,
  pub config:     PoolLayerConfig,
}

impl PoolLayer {
  pub fn new(prev_layer: Option<&Layer>, config: PoolLayerConfig) -> PoolLayer {
    let PoolLayerConfig{in_width, in_height, channels, ..} = config;
    let (out_width, out_height, _) = config.get_out_dims();
    let in_length = in_width * in_height * channels;
    let out_length = out_width * out_height * channels;
    PoolLayer{
      in_act:     prev_layer.unwrap().output_activation().unwrap(),
      in_delta:   prev_layer.unwrap().output_delta().unwrap(),
      in_mask:    DeviceBuf::with_zeros(in_length),
      out_act:    Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length))),
      out_delta:  Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length))),
      config:     config,
    }
  }
}

impl Layer for PoolLayer {
  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn forward(&mut self, ctx: &DeviceContext) {
    let PoolLayerConfig{
      in_width, in_height, channels,
      pool_size, pool_stride, pool_pad, ..} = self.config;
    match self.config.pool_kind {
      PoolKind::Max => {
        unsafe { rembrandt_kernel_image_max_pool(
            self.in_act.borrow().as_view().as_ptr(),
            in_width as i32, in_height as i32, channels as i32,
            pool_size as i32, pool_stride as i32, pool_pad as i32,
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            self.in_mask.as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      PoolKind::Average => {
        // TODO(20150924)
        unimplemented!();
      }
    }
  }

  fn backward(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    let PoolLayerConfig{
      in_width, in_height, channels,
      pool_size, pool_stride, pool_pad, ..} = self.config;
    match self.config.pool_kind {
      PoolKind::Max => {
        unsafe { rembrandt_kernel_image_max_pool_backward(
            self.out_delta.borrow().as_view().as_ptr(),
            self.in_mask.as_view().as_ptr(),
            in_width as i32, in_height as i32, channels as i32,
            pool_size as i32, pool_stride as i32, pool_pad as i32,
            self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      PoolKind::Average => {
        // TODO(20150924)
        unimplemented!();
      }
    }
  }
}

#[derive(Clone, Copy)]
pub struct DropoutLayerConfig {
  pub channels:   usize,
  pub drop_ratio: f32,
}

impl LayerConfig for DropoutLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (1, 1, self.channels)
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    (1, 1, self.channels)
  }
}

pub struct DropoutLayer {
  pub in_act:     SharedDeviceBuf<f32>,
  pub in_delta:   SharedDeviceBuf<f32>,
  pub in_rand:    DeviceBuf<f32>,
  pub in_mask:    DeviceBuf<i32>,
  pub out_act:    SharedDeviceBuf<f32>,
  pub out_delta:  SharedDeviceBuf<f32>,
  pub config:     DropoutLayerConfig,
}

impl DropoutLayer {
  pub fn new(prev_layer: Option<&Layer>, config: DropoutLayerConfig) -> DropoutLayer {
    let in_act = prev_layer.unwrap().output_activation().unwrap();
    let in_delta = prev_layer.unwrap().output_delta().unwrap();
    let act_length = in_act.borrow().as_view().len();
    DropoutLayer{
      in_act:     in_act,
      in_delta:   in_delta,
      in_rand:    DeviceBuf::with_zeros(act_length),
      in_mask:    DeviceBuf::with_zeros(act_length),
      out_act:    Rc::new(RefCell::new(DeviceBuf::with_zeros(act_length))),
      out_delta:  Rc::new(RefCell::new(DeviceBuf::with_zeros(act_length))),
      config:     config,
    }
  }
}

impl Layer for DropoutLayer {
  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn forward(&mut self, ctx: &DeviceContext) {
    let act_length = self.in_act.borrow().as_view().len();
    self.in_rand.as_mut_view().sample_uniform(ctx);
    unsafe { rembrandt_kernel_map_dropout(
        self.in_act.borrow().as_view().as_ptr(),
        act_length as i32,
        self.config.drop_ratio, 1.0,
        self.in_rand.as_view().as_ptr(),
        self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
        self.in_mask.as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }

  fn backward(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    let act_length = self.in_act.borrow().as_view().len();
    unsafe { rembrandt_kernel_map_dropout_backprop(
        self.out_delta.borrow().as_view().as_ptr(),
        act_length as i32,
        self.config.drop_ratio, 1.0,
        self.in_mask.as_view().as_ptr(),
        self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

#[derive(Clone, Copy)]
pub struct SoftmaxLossLayerConfig {
  pub num_categories: usize,
}

impl LayerConfig for SoftmaxLossLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (1, 1, self.num_categories)
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    (1, 1, self.num_categories)
  }
}

pub struct SoftmaxLossLayer {
  pub in_act:         SharedDeviceBuf<f32>,
  pub in_delta:       SharedDeviceBuf<f32>,
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
  fn output_activation(&self) -> Option<SharedDeviceBuf<f32>> { None }
  fn output_delta(&self) -> Option<SharedDeviceBuf<f32>> { None }

  fn forward(&mut self, ctx: &DeviceContext) {
    //ctx.synchronize();
    self.in_act.borrow().as_view_2d((self.num_categories, 1)).send(&mut self.probabilities.as_mut_view(), ctx);
    //ctx.synchronize();
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

  fn backward(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    unsafe { rembrandt_kernel_map_softmax_cross_entropy_loss_backprop(
        self.probabilities.as_view().as_ptr(),
        self.num_categories as i32,
        unwrap!(self.category_truth),
        self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.in_delta.borrow_mut().as_mut_view_2d((1, self.num_categories)).row_vector_scale(
        1.0 / (descent.minibatch_size() as f32),
        ctx,
    );
  }
}

impl LossLayer for SoftmaxLossLayer {
  // TODO
}
