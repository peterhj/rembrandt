use data::{SampleDatum, SampleLabel};
use opt::{DescentConfig, DescentSchedule, OptPhase};

use array::{View, MutView, WithZeros, ArrayDeserialize, Array2d, Array3d};
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
use std::collections::{BTreeMap};
use std::fmt::{Debug};
use std::fs::{File};
use std::iter::{repeat};
use std::path::{PathBuf};
use std::rc::{Rc};
use std::slice::bytes::{copy_memory};

#[derive(Clone, Copy, Debug)]
pub enum ParamsInitialization {
  Zeros,
  Normal{mean: f32, std: f32},
  Glorot,
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationFunction {
  Identity,
  Rect,
  Sigmoid,
  Tanh,
}

#[derive(Clone, Copy, Debug)]
pub enum PoolKind {
  Max,
  Average,
}

pub type SharedDeviceBuf<T> = Rc<RefCell<DeviceBuf<T>>>;

pub trait LayerConfig: Debug {
  fn get_in_dims(&self) -> (usize, usize, usize);
  fn get_out_dims(&self) -> (usize, usize, usize);
}

pub trait Layer {
  fn get_id(&self) -> usize;
  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>>;
  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>>;

  fn initialize_params(&mut self, ctx: &DeviceContext) {}
  //fn load(&mut self, phase: OptPhase, datum: &SampleDatum, maybe_label: Option<SampleLabel>, batch_idx: usize, ctx: &DeviceContext) {}
  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {}
  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {}
  fn descend(&mut self, descent: &DescentSchedule, t: usize, ctx: &DeviceContext) {}
  fn reset_gradients(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {}

  fn preload_frame(&mut self, batch_idx: usize, frame: &Array3d<u8>, ctx: &DeviceContext) {}
  fn load_frames(&mut self, batch_size: usize, ctx: &DeviceContext) {}
  fn preload_label(&mut self, batch_idx: usize, label: i32, ctx: &DeviceContext) {}
  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceContext) {}
  fn preload_mask(&mut self, batch_idx: usize, mask: &Array2d<u8>, ctx: &DeviceContext) {}
  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceContext) {}
  fn predict_labels(&mut self, batch_size: usize, ctx: &DeviceContext) -> &[i32]
  { unimplemented!(); }
  fn count_accuracy(&mut self, batch_size: usize, ctx: &DeviceContext) -> usize
  { unimplemented!(); }
  fn predict_probs(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32>
  { unimplemented!(); }
  fn predict_cdfs(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32>
  { unimplemented!(); }
}

#[derive(Clone, Copy, Debug)]
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
  pub in_buf_host:  Vec<u8>,
  pub in_buf:       DeviceBuf<u8>,
  pub out_buf:      SharedDeviceBuf<f32>,
  pub layer_id:   usize,
  pub config:     DataLayerConfig,
  pub batch_lim:  usize,
}

impl DataLayer {
  pub fn new(layer_id: usize, config: DataLayerConfig, batch_size: usize) -> DataLayer {
    let raw_length = config.raw_width * config.raw_height * config.channels * batch_size;
    let crop_length = config.crop_width * config.crop_height * config.channels * batch_size;
    DataLayer{
      in_bytes:   DeviceBuf::with_zeros(raw_length),
      raw_image:  DeviceBuf::with_zeros(raw_length),
      crop_image: Rc::new(RefCell::new(DeviceBuf::with_zeros(crop_length))),
      in_buf_host:  repeat(0).take(raw_length).collect(),
      in_buf:       DeviceBuf::with_zeros(raw_length),
      out_buf:      Rc::new(RefCell::new(DeviceBuf::with_zeros(crop_length))),
      layer_id:   layer_id,
      config:     config,
      batch_lim:  batch_size,
    }
  }
}

impl Layer for DataLayer {
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    //Some(self.crop_image.clone())
    Some(self.out_buf.clone())
  }

  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    None
  }

  /*fn load(&mut self, phase: OptPhase, datum: &SampleDatum, maybe_label: Option<SampleLabel>, batch_idx: usize, ctx: &DeviceContext) {
    assert!(batch_idx + 1 <= self.batch_lim);
    let mut rng = thread_rng();
    let DataLayerConfig{
      raw_width, raw_height,
      crop_width, crop_height,
      channels} = self.config;
    match datum {
      &SampleDatum::RgbPerChannelBytes(ref bytes) => {
        let bytes_view = bytes.as_view();
        let dims = bytes_view.get_bound();
        let length = bytes_view.len();
        {
          let (_, mut in_bytes_view) = self.in_bytes.as_mut_view().split_at(batch_idx * length, length);
          let mut in_bytes_view3d = in_bytes_view.as_mut_view_3d(dims);
          in_bytes_view3d.sync_load(&bytes_view, ctx);
        }
        {
          let &mut DataLayer{ref in_bytes, ref mut raw_image, ..} = self;
          let (_, in_bytes_view) = in_bytes.as_view().split_at(batch_idx * length, length);
          let (_, mut raw_image_view) = raw_image.as_mut_view().split_at(batch_idx * length, length);
          unsafe { rembrandt_kernel_image_cast_to_float(
              dims.0 as i32, dims.1 as i32, dims.2 as i32,
              in_bytes_view.as_ptr(),
              raw_image_view.as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }

        // TODO(20150921): Mean subtract.
        /*unsafe { rembrandt_kernel_map_add_constant_float(
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            length as i32,
            -33.318,
            ctx.stream.ptr,
        ) };*/

        // Crop.
        if raw_width == crop_width && raw_height == crop_height {
          let &mut DataLayer{ref raw_image, ref mut crop_image, ..} = self;
          let (_, raw_image_view) = raw_image.as_view().split_at(batch_idx * length, length);
          let mut crop_image = crop_image.borrow_mut();
          let (_, mut crop_image_view) = crop_image.as_mut_view().split_at(batch_idx * length, length);
          raw_image_view.send(&mut crop_image_view, ctx);
        } else {
          // TODO(20151016): crop w/ batches.
          unimplemented!();
          let (off_w, off_h) = match phase {
            OptPhase::Training    => {
              let range_w = Range::new(0, raw_width - crop_width);
              let range_h = Range::new(0, raw_height - crop_height);
              (range_w.ind_sample(&mut rng), range_h.ind_sample(&mut rng))
            }
            // XXX: Always center crop during evaluation.
            OptPhase::Evaluation  => ((raw_width - crop_width) / 2, (raw_height - crop_height) / 2),
          };
          for c in (0 .. channels) {
            let (_, back_raw_view) = self.raw_image.as_view().split_at(c * raw_width * raw_height, raw_width * raw_height);
            let back_raw_view_2d = back_raw_view.as_view_2d((raw_width, raw_height));
            let cropped_raw_view = back_raw_view_2d.view((off_w, off_h), (off_w + crop_width, off_h + crop_height));
            let mut crop_image = self.crop_image.borrow_mut();
            let mut crop_view = crop_image.as_mut_view();
            let (_, mut back_crop_view) = crop_view.split_at(c * crop_width * crop_height, crop_width * crop_height);
            let mut shaped_crop_view = back_crop_view.as_mut_view_2d((crop_width, crop_height));
            cropped_raw_view.send(&mut shaped_crop_view, ctx);
          }
        }

        // Mirror.
        // TODO
      }
      _ => unimplemented!(),
    }
  }*/

  fn preload_frame(&mut self, batch_idx: usize, frame: &Array3d<u8>, ctx: &DeviceContext) {
    assert!(batch_idx < self.batch_lim);
    let DataLayerConfig{
      raw_width, raw_height, channels, ..} = self.config;
    let raw_length = raw_width * raw_height * channels;
    let frame = frame.as_slice();
    assert_eq!(raw_length, frame.len());
    copy_memory(frame, &mut self.in_buf_host[batch_idx * raw_length .. (batch_idx + 1) * raw_length]);
  }

  fn load_frames(&mut self, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);
    let DataLayerConfig{
      raw_width, raw_height, channels, ..} = self.config;
    let raw_length = raw_width * raw_height * channels;
    {
      let in_buf_host = &self.in_buf_host;
      let mut in_view = self.in_buf.as_mut_view();
      in_view.sync_load(in_buf_host, ctx);
    }
    unsafe { rembrandt_kernel_image_cast_to_float(
        (raw_length * batch_size) as i32, 1, 1,
        self.in_buf.as_view().as_ptr(),
        self.out_buf.borrow_mut().as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
  }
}

#[derive(Clone, Copy, Debug)]
pub struct SplitLayerConfig {
  pub width:    usize,
  pub height:   usize,
  pub channels: usize,
  pub fan_out:  usize,
}

impl LayerConfig for SplitLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (self.width, self.height, self.channels)
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    (self.width, self.height, self.channels)
  }
}

pub struct SplitLayer {
  pub in_act:     SharedDeviceBuf<f32>,
  pub in_delta:   SharedDeviceBuf<f32>,
  pub out_acts:   BTreeMap<usize, SharedDeviceBuf<f32>>,
  pub out_deltas: BTreeMap<usize, SharedDeviceBuf<f32>>,
  pub layer_id:   usize,
  pub config:     SplitLayerConfig,
}

impl SplitLayer {
  // TODO
}

impl Layer for SplitLayer {
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_acts[&out_layer_id].clone())
  }

  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_deltas[&out_layer_id].clone())
  }

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
    let in_act = self.in_act.borrow();
    let in_act_view = in_act.as_view();
    for (_, out_act) in self.out_acts.iter_mut() {
      let mut out_act = out_act.borrow_mut();
      let mut out_act_view = out_act.as_mut_view();
      in_act_view.send(&mut out_act_view, ctx);
    }
  }

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
    let dims = self.config.get_in_dims();
    let length = dims.0 * dims.1 * dims.2;
    for (idx, (_, out_delta)) in self.out_deltas.iter().enumerate() {
      let out_delta = out_delta.borrow();
      let out_delta_view = out_delta.as_view_2d((1, length));
      let mut in_delta = self.in_delta.borrow_mut();
      let mut in_delta_view = in_delta.as_mut_view_2d((1, length));
      if idx == 0 {
        out_delta_view.send(&mut in_delta_view, ctx);
      } else {
        in_delta_view.row_vector_sum(
            1.0,
            &out_delta_view,
            ctx,
        );
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct JoinLayerConfig {
  pub width:        usize,
  pub height:       usize,
  pub in_channels:  usize,
  pub out_channels: usize,
  pub fan_in:       usize,
}

impl LayerConfig for JoinLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (self.width, self.height, self.in_channels)
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    (self.width, self.height, self.out_channels)
  }
}

pub struct JoinLayer {
  pub in_acts:    BTreeMap<usize, SharedDeviceBuf<f32>>,
  pub in_deltas:  BTreeMap<usize, SharedDeviceBuf<f32>>,
  pub out_act:    SharedDeviceBuf<f32>,
  pub out_delta:  SharedDeviceBuf<f32>,
  pub layer_id:   usize,
  pub config:     JoinLayerConfig,
}

// TODO

#[derive(Clone, Copy, Debug)]
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
  pub layer_id:     usize,
  pub config:       FullyConnLayerConfig,
}

impl FullyConnLayer {
  pub fn new(layer_id: usize, config: FullyConnLayerConfig, batch_size: usize, prev_layer: Option<&Layer>) -> FullyConnLayer {
    FullyConnLayer{
      in_act:       prev_layer.unwrap().output_activation(-1).unwrap(),
      in_delta:     prev_layer.unwrap().output_delta(-1),
      out_act:      Rc::new(RefCell::new(DeviceBuf::with_zeros(config.out_channels))),
      out_delta:    Rc::new(RefCell::new(DeviceBuf::with_zeros(config.out_channels))),
      weights:      DeviceArray2d::with_zeros((config.in_channels, config.out_channels)),
      bias:         DeviceArray2d::with_zeros((1, config.out_channels)),
      grad_weights: DeviceArray2d::with_zeros((config.in_channels, config.out_channels)),
      grad_bias:    DeviceArray2d::with_zeros((1, config.out_channels)),
      layer_id:     layer_id,
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
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
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

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
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

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
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
    self.grad_weights.as_mut_view().matrix_sum(descent.l2_reg_coef(), &self.weights.as_view(), ctx);
    self.grad_bias.as_mut_view().matrix_sum(descent.l2_reg_coef(), &self.bias.as_view(), ctx);
    self.weights.as_mut_view().matrix_sum(-descent.step_size(t), &self.grad_weights.as_view(), ctx);
    self.bias.as_mut_view().matrix_sum(-descent.step_size(t), &self.grad_bias.as_view(), ctx);
  }
}

#[derive(Clone, Copy, Debug)]
pub struct Conv2dLayerConfig {
  pub in_width:     usize,
  pub in_height:    usize,
  pub in_channels:  usize,
  pub conv_size:    usize,
  pub conv_stride:  usize,
  pub conv_pad:     usize,
  pub out_channels: usize,
  pub act_fun:      ActivationFunction,
  pub init_weights: ParamsInitialization,
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
  pub work:         DeviceBuf<f32>,
  pub out_act:      SharedDeviceBuf<f32>,
  pub out_delta:    SharedDeviceBuf<f32>,
  pub weights:      DeviceArray2d<f32>,
  pub bias:         DeviceArray2d<f32>,
  pub grad_weights: DeviceArray2d<f32>,
  pub grad_bias:    DeviceArray2d<f32>,
  pub layer_id:     usize,
  pub config:       Conv2dLayerConfig,
  pub batch_lim:    usize,
}

impl Conv2dLayer {
  pub fn new(layer_id: usize, config: Conv2dLayerConfig, batch_size: usize, prev_layer: Option<&Layer>, ctx: &DeviceContext) -> Conv2dLayer {
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
      in_act:       prev_layer.unwrap().output_activation(-1).unwrap(),
      in_delta:     prev_layer.unwrap().output_delta(-1),
      // XXX(20151016): note that im2col buffers are NOT duplicated for batch.
      in_col:       DeviceArray2d::with_zeros((out_width * out_height, conv_size * conv_size * in_channels)),
      grad_col:     DeviceArray2d::with_zeros((out_width * out_height, conv_size * conv_size * in_channels)),
      unit:         unit,
      work:         DeviceBuf::with_zeros(1024), // FIXME(20151002): for cuDNN work space.
      out_act:      Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length * batch_size))),
      out_delta:    Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length * batch_size))),
      weights:      DeviceArray2d::with_zeros((conv_size * conv_size * in_channels, out_channels)),
      bias:         DeviceArray2d::with_zeros((1, out_channels)),
      grad_weights: DeviceArray2d::with_zeros((conv_size * conv_size * in_channels, out_channels)),
      grad_bias:    DeviceArray2d::with_zeros((1, out_channels)),
      layer_id:     layer_id,
      config:       config,
      batch_lim:    batch_size,
    }
  }
}

impl Layer for Conv2dLayer {
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn initialize_params(&mut self, ctx: &DeviceContext) {
    let Conv2dLayerConfig{in_channels, conv_size, out_channels, ..} = self.config;
    let mut rng = thread_rng();
    let mut init_weights = Array2d::with_zeros((conv_size * conv_size * in_channels, out_channels));
    match self.config.init_weights {
      ParamsInitialization::Normal{mean, std} => {
        let dist = Normal::new(mean as f64, std as f64);
        for w in init_weights.as_mut_view().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      _ => unimplemented!(),
    }
    let init_bias = Array2d::with_zeros((1, out_channels));
    self.weights.as_mut_view().sync_load(&init_weights.as_view(), ctx);
    self.bias.as_mut_view().sync_load(&init_bias.as_view(), ctx);
  }

  fn reset_gradients(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    self.grad_weights.as_mut_view().matrix_scale(descent.momentum(), ctx);
    self.grad_bias.as_mut_view().row_vector_scale(descent.momentum(), ctx);
  }

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size <= self.batch_lim);
    let Conv2dLayerConfig{
      in_width, in_height, in_channels,
      conv_size, conv_stride, conv_pad,
      out_channels, ..} = self.config;
    let (out_width, out_height, _) = self.config.get_out_dims();
    let in_length = in_width * in_height * in_channels;
    let out_length = out_width * out_height * out_channels;
    for batch_idx in (0 .. batch_size) {
      let &mut Conv2dLayer{
        ref in_act, ref mut in_col, ref mut out_act,
        ref weights, ref bias, ref unit, ..} = self;
      let in_act = in_act.borrow();
      let mut out_act = out_act.borrow_mut();
      let (_, in_act_view) = in_act.as_view().split_at(batch_idx * in_length, in_length);
      let (_, mut out_act_view) = out_act.as_mut_view().split_at(batch_idx * out_length, out_length);
      // FIXME(20151016): 1x1 conv optimization.
      unsafe { rembrandt_kernel_image_im2col(
          in_act_view.as_ptr(),
          in_width as i32, in_height as i32, in_channels as i32,
          conv_size as i32, conv_stride as i32, conv_pad as i32,
          in_col.as_mut_view().as_mut_ptr(),
          ctx.stream.ptr,
      ) };
      out_act_view.as_mut_view_2d((out_width * out_height, out_channels)).matrix_prod(
          1.0,
          &in_col.as_view(), Transpose::N,
          &weights.as_view(), Transpose::N,
          0.0,
          ctx,
      );
      out_act_view.as_mut_view_2d((out_width * out_height, out_channels)).matrix_prod(
          1.0,
          &unit.as_view(), Transpose::N,
          &bias.as_view(), Transpose::N,
          1.0,
          ctx,
      );
      match self.config.act_fun {
        ActivationFunction::Identity => {}
        ActivationFunction::Rect => {
          unsafe { rembrandt_kernel_map_relu_activation(
              out_length as i32,
              out_act_view.as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
        ActivationFunction::Sigmoid => {
          unsafe { rembrandt_kernel_map_sigmoid_activation(
              out_length as i32,
              out_act_view.as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
        ActivationFunction::Tanh => {
          unsafe { rembrandt_kernel_map_tanh_activation(
              out_length as i32,
              out_act_view.as_mut_ptr(),
              ctx.stream.ptr,
          ) };
        }
      }
    }
  }

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
    let Conv2dLayerConfig{
      in_width, in_height, in_channels,
      conv_size, conv_stride, conv_pad,
      out_channels, ..} = self.config;
    let (out_width, out_height, _) = self.config.get_out_dims();
    let in_length = in_width * in_height * in_channels;
    let out_length = out_width * out_height * out_channels;
    for batch_idx in (0 .. batch_size) {
      let &mut Conv2dLayer{
        ref in_act, ref mut in_delta, ref out_act, ref out_delta,
        ref mut grad_weights, ref mut grad_bias,
        ref mut in_col, ref mut grad_col,
        ref weights, ref bias, ref unit, ..} = self;
      let in_act = in_act.borrow();
      let out_act = out_act.borrow();
      let (_, in_act_view) = in_act.as_view().split_at(batch_idx * in_length, in_length);
      let (_, out_act_view) = out_act.as_view().split_at(batch_idx * out_length, out_length);

      {
        let mut out_delta = out_delta.borrow_mut();
        let (_, mut out_delta_view) = out_delta.as_mut_view().split_at(batch_idx * out_length, out_length);
        match self.config.act_fun {
          ActivationFunction::Identity => {}
          ActivationFunction::Rect => {
            unsafe { rembrandt_kernel_map_relu_activation_backprop(
                out_length as i32,
                out_act_view.as_ptr(),
                out_delta_view.as_mut_ptr(),
                ctx.stream.ptr,
            ) };
          }
          ActivationFunction::Sigmoid => {
            unsafe { rembrandt_kernel_map_sigmoid_activation_backprop(
                out_length as i32,
                out_act_view.as_ptr(),
                out_delta_view.as_mut_ptr(),
                ctx.stream.ptr,
            ) };
          }
          ActivationFunction::Tanh => {
            unsafe { rembrandt_kernel_map_tanh_activation_backprop(
                out_length as i32,
                out_act_view.as_ptr(),
                out_delta_view.as_mut_ptr(),
                ctx.stream.ptr,
            ) };
          }
        }
      }

      let out_delta = out_delta.borrow();
      let (_, out_delta_view) = out_delta.as_view().split_at(batch_idx * out_length, out_length);
      // XXX: Recomputing im2col.
      // FIXME(20151016): 1x1 conv optimization.
      unsafe { rembrandt_kernel_image_im2col(
          in_act_view.as_ptr(),
          in_width as i32, in_height as i32, in_channels as i32,
          conv_size as i32, conv_stride as i32, conv_pad as i32,
          in_col.as_mut_view().as_mut_ptr(),
          ctx.stream.ptr,
      ) };
      grad_weights.as_mut_view().matrix_prod(
          1.0,
          &in_col.as_view(), Transpose::T,
          &out_delta_view.as_view_2d((out_width * out_height, out_channels)), Transpose::N,
          1.0,
          ctx,
      );
      grad_bias.as_mut_view().matrix_prod(
          1.0,
          &unit.as_view(), Transpose::T,
          &out_delta_view.as_view_2d((out_width * out_height, out_channels)), Transpose::N,
          1.0,
          ctx,
      );
      if let &mut Some(ref mut in_delta) = in_delta {
        let mut in_delta = in_delta.borrow_mut();
        let (_, mut in_delta_view) = in_delta.as_mut_view().split_at(batch_idx * in_length, in_length);
        grad_col.as_mut_view().matrix_prod(
            1.0,
            &out_delta_view.as_view_2d((out_width * out_height, out_channels)), Transpose::N,
            &weights.as_view(), Transpose::T,
            0.0,
            ctx,
        );
        // FIXME(20151016): 1x1 conv optimization.
        unsafe { rembrandt_kernel_image_col2im(
            grad_col.as_view().as_ptr(),
            in_width as i32, in_height as i32, in_channels as i32,
            conv_size as i32, conv_stride as i32, conv_pad as i32,
            in_delta_view.as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
    }
  }

  fn descend(&mut self, descent: &DescentSchedule, t: usize, ctx: &DeviceContext) {
    self.grad_weights.as_mut_view().matrix_sum(descent.l2_reg_coef(), &self.weights.as_view(), ctx);
    self.grad_bias.as_mut_view().matrix_sum(descent.l2_reg_coef(), &self.bias.as_view(), ctx);
    self.weights.as_mut_view().matrix_sum(-descent.step_size(t), &self.grad_weights.as_view(), ctx);
    self.bias.as_mut_view().matrix_sum(-descent.step_size(t), &self.grad_bias.as_view(), ctx);
  }
}

#[derive(Clone, Copy, Debug)]
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
  pub layer_id:   usize,
  pub config:     PoolLayerConfig,
}

impl PoolLayer {
  pub fn new(layer_id: usize, config: PoolLayerConfig, batch_size: usize, prev_layer: Option<&Layer>) -> PoolLayer {
    let PoolLayerConfig{in_width, in_height, channels, ..} = config;
    let (out_width, out_height, _) = config.get_out_dims();
    let in_length = in_width * in_height * channels;
    let out_length = out_width * out_height * channels;
    PoolLayer{
      in_act:     prev_layer.unwrap().output_activation(-1).unwrap(),
      in_delta:   prev_layer.unwrap().output_delta(-1).unwrap(),
      in_mask:    DeviceBuf::with_zeros(in_length),
      out_act:    Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length))),
      out_delta:  Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length))),
      layer_id:   layer_id,
      config:     config,
    }
  }
}

impl Layer for PoolLayer {
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
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
        unsafe { rembrandt_kernel_image_average_pool(
            self.in_act.borrow().as_view().as_ptr(),
            in_width as i32, in_height as i32, channels as i32,
            pool_size as i32, pool_stride as i32, pool_pad as i32,
            self.out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
    }
  }

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
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
        unsafe { rembrandt_kernel_image_average_pool_backward(
            self.out_delta.borrow().as_view().as_ptr(),
            in_width as i32, in_height as i32, channels as i32,
            pool_size as i32, pool_stride as i32, pool_pad as i32,
            self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub struct LocalResNormLayerConfig {
  pub channels:   usize,
  pub cross_size: usize,
}

impl LayerConfig for LocalResNormLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (0, 0, 0) // FIXME
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    (0, 0, 0) // FIXME
  }
}

pub struct LocalResNormLayer {
  pub layer_id:   usize,
  pub config:     LocalResNormLayerConfig,
}

impl LocalResNormLayer {
  pub fn new(layer_id: usize, config: LocalResNormLayerConfig, batch_size: usize, prev_layer: Option<&Layer>) -> LocalResNormLayer {
    // TODO
    LocalResNormLayer{
      layer_id: layer_id,
      config:   config,
    }
  }
}

impl Layer for LocalResNormLayer {
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    None // FIXME
    //Some(self.out_act.clone())
  }

  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    None // FIXME
    //Some(self.out_delta.clone())
  }

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
    // TODO
  }

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
    // TODO
  }
}

#[derive(Clone, Copy, Debug)]
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
  pub layer_id:   usize,
  pub config:     DropoutLayerConfig,
}

impl DropoutLayer {
  pub fn new(layer_id: usize, config: DropoutLayerConfig, batch_size: usize, prev_layer: Option<&Layer>) -> DropoutLayer {
    let in_act = prev_layer.unwrap().output_activation(-1).unwrap();
    let in_delta = prev_layer.unwrap().output_delta(-1).unwrap();
    let act_length = in_act.borrow().as_view().len();
    DropoutLayer{
      in_act:     in_act,
      in_delta:   in_delta,
      in_rand:    DeviceBuf::with_zeros(act_length),
      in_mask:    DeviceBuf::with_zeros(act_length),
      out_act:    Rc::new(RefCell::new(DeviceBuf::with_zeros(act_length))),
      out_delta:  Rc::new(RefCell::new(DeviceBuf::with_zeros(act_length))),
      layer_id:   layer_id,
      config:     config,
    }
  }
}

impl Layer for DropoutLayer {
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
    match phase {
      OptPhase::Training => {
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
      OptPhase::Evaluation => {
        self.in_act.borrow().as_view().send(&mut self.out_act.borrow_mut().as_mut_view(), ctx);
      }
    }
  }

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
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

#[derive(Clone, Copy, Debug)]
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
  pub category_truth: Vec<i32>,

  pub pred_labels:      DeviceArray2d<i32>,
  pub pred_labels_host: Array2d<i32>,
  pub true_labels:      DeviceArray2d<i32>,
  pub true_labels_host: Array2d<i32>,

  pub layer_id:       usize,
  pub num_categories: usize,
  pub batch_lim:      usize,
}

impl SoftmaxLossLayer {
  pub fn new(layer_id: usize, num_categories: usize, batch_size: usize, prev_layer: Option<&Layer>) -> SoftmaxLossLayer {
    SoftmaxLossLayer{
      in_act:         prev_layer.unwrap().output_activation(-1).unwrap(),
      in_delta:       prev_layer.unwrap().output_delta(-1).unwrap(),
      probabilities:  DeviceArray2d::with_zeros((num_categories, batch_size)),
      normalization:  DeviceArray2d::with_zeros((1, batch_size)),
      tmp_max:        DeviceArray2d::with_zeros((1, batch_size)),

      category_guess: DeviceArray2d::with_zeros((1, batch_size)),
      category_truth: Vec::new(),

      pred_labels:      DeviceArray2d::with_zeros((1, batch_size)),
      pred_labels_host: Array2d::with_zeros((1, batch_size)),
      true_labels:      DeviceArray2d::with_zeros((1, batch_size)),
      true_labels_host: Array2d::with_zeros((1, batch_size)),

      layer_id:       layer_id,
      num_categories: num_categories,
      batch_lim:      batch_size,
    }
  }

  /*pub fn get_guess_one(&self, ctx: &DeviceContext) -> i32 {
    let mut host_guess = Array2d::with_zeros((1, 1));
    self.category_guess.as_view().sync_store(&mut host_guess.as_mut_view(), &ctx);
    host_guess.as_view().as_slice()[0]
  }

  pub fn correct_guess_one(&self, ctx: &DeviceContext) -> bool {
    let guess = self.get_guess(ctx);
    //println!("DEBUG: truth: {} guess: {}", unwrap!(self.category_truth), guess);
    unwrap!(self.category_truth) == guess
  }*/

  /*pub fn get_guess(&self, batch_size: usize, ctx: &DeviceContext) -> Vec<i32> {
    assert!(batch_size <= self.batch_lim);
    let mut host_guess = Array2d::with_zeros((1, batch_size));
    self.category_guess.as_view().view((0, 0), (1, batch_size))
      .sync_store(&mut host_guess.as_mut_view(), &ctx);
    host_guess.into_data()
  }

  pub fn correct_guess(&self, batch_size: usize, ctx: &DeviceContext) -> usize {
    assert!(batch_size <= self.batch_lim);
    let guess = self.get_guess(batch_size, ctx);
    assert_eq!(guess.len(), self.category_truth.len());
    let mut num_correct = 0;
    for (&y_truth, &y_hat) in self.category_truth.iter().zip(guess.iter()) {
      if y_truth == y_hat {
        num_correct += 1;
      }
    }
    num_correct
  }*/
}

impl Layer for SoftmaxLossLayer {
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> { None }
  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> { None }

  /*fn load(&mut self, phase: OptPhase, datum: &SampleDatum, maybe_label: Option<SampleLabel>, batch_idx: usize, ctx: &DeviceContext) {
    //self.category_truth = maybe_label.map(|label| label.0);
    if batch_idx == 0 {
      self.category_truth.clear();
    }
    if let Some(SampleLabel(label)) = maybe_label {
      assert_eq!(batch_idx, self.category_truth.len());
      self.category_truth.push(label);
    }
  }*/

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size <= self.batch_lim);
    self.in_act.borrow().as_view_2d((self.num_categories, batch_size))
      .send(&mut self.probabilities.as_mut_view().mut_view((0, 0), (self.num_categories, batch_size)), ctx);
    for batch_idx in (0 .. batch_size) {
      let &mut SoftmaxLossLayer{
        ref mut probabilities, ref mut normalization,
        ref mut tmp_max, ref mut category_guess,
        ref mut pred_labels, ..} = self;
      unsafe { rembrandt_kernel_blockreduce_argmax(
          self.num_categories as i32,
          probabilities.as_view()
            .view((0, batch_idx), (self.num_categories, batch_idx + 1)).as_ptr(),
          tmp_max.as_mut_view()
            .mut_view((0, batch_idx), (1, batch_idx + 1)).as_mut_ptr(),
          //category_guess.as_mut_view()
          pred_labels.as_mut_view()
            .mut_view((0, batch_idx), (1, batch_idx + 1)).as_mut_ptr(),
          ctx.stream.ptr,
      ) };
      unsafe { rembrandt_kernel_map_subtract_scalar(
          probabilities.as_mut_view()
            .mut_view((0, batch_idx), (self.num_categories, batch_idx + 1)).as_mut_ptr(),
          self.num_categories as i32,
          tmp_max.as_view()
            .view((0, batch_idx), (1, batch_idx + 1)).as_ptr(),
          ctx.stream.ptr,
      ) };
      unsafe { rembrandt_kernel_map_exp(
          probabilities.as_mut_view()
            .mut_view((0, batch_idx), (self.num_categories, batch_idx + 1)).as_mut_ptr(),
          self.num_categories as i32,
          ctx.stream.ptr,
      ) };
      unsafe { rembrandt_kernel_blockreduce_sum(
          self.num_categories as i32,
          probabilities.as_view()
            .view((0, batch_idx), (self.num_categories, batch_idx + 1)).as_ptr(),
          normalization.as_mut_view()
            .mut_view((0, batch_idx), (1, batch_idx + 1)).as_mut_ptr(),
          ctx.stream.ptr,
      ) };
      unsafe { rembrandt_kernel_map_divide_scalar(
          probabilities.as_mut_view()
            .mut_view((0, batch_idx), (self.num_categories, batch_idx + 1)).as_mut_ptr(),
          self.num_categories as i32,
          normalization.as_view()
            .view((0, batch_idx), (1, batch_idx + 1)).as_ptr(),
          ctx.stream.ptr,
      ) };
    }
  }

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size <= self.batch_lim);
    for batch_idx in (0 .. batch_size) {
      let &mut SoftmaxLossLayer{
        ref mut in_delta, ref probabilities, ..} = self;
      let mut in_delta = in_delta.borrow_mut();
      let (_, mut in_delta_view) = in_delta.as_mut_view().split_at(self.num_categories * batch_idx, self.num_categories);
      unsafe { rembrandt_kernel_map_softmax_cross_entropy_loss_backprop(
          probabilities.as_view()
            .view((0, batch_idx), (self.num_categories, batch_idx + 1)).as_ptr(),
          self.num_categories as i32,
          //self.category_truth[batch_idx],
          self.true_labels_host.as_slice()[batch_idx],
          in_delta_view.as_mut_ptr(),
          ctx.stream.ptr,
      ) };
      in_delta_view.as_mut_view_2d((1, self.num_categories)).row_vector_scale(
          1.0 / (descent.minibatch_size() as f32),
          ctx,
      );
    }
  }

  fn preload_label(&mut self, batch_idx: usize, label: i32, ctx: &DeviceContext) {
    assert!(batch_idx < self.batch_lim);
    self.true_labels_host.as_mut_slice()[batch_idx] = label;
  }

  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);
    self.true_labels.as_mut_view().sync_load(&self.true_labels_host.as_view(), ctx);
  }

  fn preload_mask(&mut self, batch_idx: usize, mask: &Array2d<u8>, ctx: &DeviceContext) {}

  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceContext) {}

  fn predict_labels(&mut self, batch_size: usize, ctx: &DeviceContext) -> &[i32] {
    assert!(batch_size == self.batch_lim);
    self.pred_labels.as_view().sync_store(&mut self.pred_labels_host.as_mut_view(), ctx);
    self.pred_labels_host.as_slice()
  }

  fn count_accuracy(&mut self, batch_size: usize, ctx: &DeviceContext) -> usize {
    assert!(batch_size == self.batch_lim);
    let mut num_correct = 0;
    for (&y_truth, &y_hat) in self.true_labels_host.as_slice().iter().zip(self.pred_labels_host.as_slice().iter()) {
      if y_truth == y_hat {
        num_correct += 1;
      }
    }
    num_correct
  }

  fn predict_probs(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32>
  { unimplemented!(); }

  fn predict_cdfs(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32>
  { unimplemented!(); }
}

/*pub struct SoftmaxPolicyGradientLossLayer {
  pub in_act:         SharedDeviceBuf<f32>,
  pub in_delta:       SharedDeviceBuf<f32>,
  pub probabilities:  DeviceArray2d<f32>,
  pub normalization:  DeviceArray2d<f32>,
  pub tmp_max:        DeviceArray2d<f32>,
  pub action:         DeviceArray2d<i32>,
  pub layer_id:       usize,
  pub num_categories: usize,
}

impl Layer for SoftmaxPolicyGradientLossLayer {
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> { None }
  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> { None }

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
    self.in_act.borrow().as_view_2d((self.num_categories, 1)).send(&mut self.probabilities.as_mut_view(), ctx);
    unsafe { rembrandt_kernel_blockreduce_argmax(
        self.num_categories as i32,
        self.probabilities.as_view().as_ptr(),
        self.tmp_max.as_mut_view().as_mut_ptr(),
        self.action.as_mut_view().as_mut_ptr(),
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

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
    // FIXME(20151005): read label from device side.
    unimplemented!();
    /*unsafe { rembrandt_kernel_map_softmax_cross_entropy_loss_dev_backprop(
        self.probabilities.as_view().as_ptr(),
        self.num_categories as i32,
        self.action.as_view().as_ptr(),
        self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.in_delta.borrow_mut().as_mut_view_2d((1, self.num_categories)).row_vector_scale(
        1.0 / (descent.minibatch_size() as f32),
        ctx,
    );*/
  }
}*/
