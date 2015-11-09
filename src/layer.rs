use data::{SampleDatum, SampleLabel, NewSampleLabel};
use opt::{DescentConfig, DescentSchedule, OptPhase};

use array::{
  View, MutView, WithZeros, ArrayDeserialize, ArraySerialize,
  NdArrayFormat, Array2d, Array3d,
};
use async::{AsyncContext, AsyncLoad, AsyncStore, AsyncSend};
use async_cuda::array_num::{DeviceNumExt};
use async_cuda::array_rand::{DeviceRandExt};
use async_cuda::array_types::{DeviceArray2d, DeviceBuf};
use async_cuda::context::{DeviceContext};
use cuda_dnn::{
  CudnnConvFwdOp, CudnnConvBwdFilterOp, CudnnConvBwdDataOp,
  CudnnAddOp, CudnnActKind, CudnnActOp, CudnnSoftmaxOp,
  CudnnTensorDesc, CudnnFilterDesc, CudnnConvDesc,
};
use linalg::blas::{BVector, BMatrix, Transpose};
use rembrandt_kernels::*;
use rembrandt_kernels::ffi::*;

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::cell::{RefCell};
use std::cmp::{max};
use std::collections::{BTreeMap};
use std::fmt::{Debug};
use std::fs::{File};
use std::io::{Cursor};
use std::iter::{repeat};
use std::path::{PathBuf};
use std::ptr::{null_mut};
use std::rc::{Rc};
use std::slice::bytes::{copy_memory};

#[derive(Clone, Copy, Debug)]
pub enum ParamsInitialization {
  None,
  Zeros,
  Uniform{half_range: f32},
  Normal{mean: f32, std: f32},
  Glorot,
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationFunction {
  Identity,
  Rect,
  Sigmoid,
  Tanh,
  BoundedRect,  // a (failed) experiment at extracting more sparsity.
}

#[derive(Clone, Copy, Debug)]
pub enum PoolKind {
  Max,
  Average,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct LayerStats {
  pub num_nonzero_acts: i64,
  pub num_total_acts:   i64,
  pub num_nonzero_grad: i64,
  pub num_total_grad:   i64,
}

impl LayerStats {
  /*pub fn new() -> LayerStats {
    LayerStats{
    }
  }*/

  pub fn act_sparseness(&self) -> f64 {
    1.0 - ((self.num_nonzero_acts as f64) / (self.num_total_acts as f64))
  }

  pub fn grad_sparseness(&self) -> f64 {
    1.0 - ((self.num_nonzero_grad as f64) / (self.num_total_grad as f64))
  }
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
  fn initialize_gradients(&mut self, ctx: &DeviceContext) {}
  fn load_params(&mut self, blob: &[u8], ctx: &DeviceContext) -> usize { 0 }
  fn save_params(&self, ctx: &DeviceContext) -> Vec<u8> { Vec::new() }

  //fn load(&mut self, phase: OptPhase, datum: &SampleDatum, maybe_label: Option<SampleLabel>, batch_idx: usize, ctx: &DeviceContext) {}
  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {}
  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {}
  fn descend(&mut self, descent: &DescentSchedule, t: usize, ctx: &DeviceContext) {}
  fn reset_gradients(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {}
  fn reset_objective(&mut self, ctx: &DeviceContext) {}

  fn reset_stats(&mut self) { unimplemented!(); }
  fn stats(&self) -> &LayerStats { unimplemented!(); }

  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8] { unimplemented!(); }
  fn preload_frame(&mut self, batch_idx: usize, frame: &Array3d<u8>, ctx: &DeviceContext) {}
  fn preload_frame_permute(&mut self, batch_idx: usize, frame: &Array3d<u8>, permute_idx: usize, ctx: &DeviceContext) {}
  fn load_frames(&mut self, batch_size: usize, ctx: &DeviceContext) {}

  fn preload_mask(&mut self, batch_idx: usize, mask: &Array2d<f32>, ctx: &DeviceContext) {}
  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceContext) {}

  fn preload_label(&mut self, batch_idx: usize, label: i32, ctx: &DeviceContext) {}
  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceContext) {}

  fn preload_new_label(&mut self, batch_idx: usize, label: NewSampleLabel, ctx: &DeviceContext) {}
  fn load_new_labels(&mut self, batch_size: usize, ctx: &DeviceContext) {}

  fn store_labels(&mut self, batch_size: usize, ctx: &DeviceContext) { unimplemented!(); }
  fn predict_labels(&mut self, batch_size: usize, ctx: &DeviceContext) -> &[i32] { unimplemented!(); }
  fn count_accuracy(&mut self, batch_size: usize, ctx: &DeviceContext) -> usize { unimplemented!(); }

  fn store_ranked_labels(&mut self, batch_size: usize, ctx: &DeviceContext) { unimplemented!(); }
  fn predict_ranked_labels(&mut self, batch_size: usize) -> &[i32] { unimplemented!(); }

  fn store_loss(&mut self, batch_size: usize, ctx: &DeviceContext) { unimplemented!(); }
  fn predict_loss(&mut self, batch_size: usize, ctx: &DeviceContext) -> f32 { unimplemented!(); }

  fn store_probs(&mut self, batch_size: usize, ctx: &DeviceContext) { unimplemented!(); }
  fn predict_probs(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32> { unimplemented!(); }

  fn store_cdfs(&mut self, batch_size: usize, ctx: &DeviceContext) { unimplemented!(); }
  fn predict_cdfs(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32> { unimplemented!(); }

  /*fn store_boltzmann_q(&mut self, batch_size: usize, beta: f32, ctx: &DeviceContext) { unimplemented!(); }
  fn predict_boltzmann_q(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32> { unimplemented!(); }*/
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
  //pub in_bytes:   DeviceBuf<u8>,
  //pub raw_image:  DeviceBuf<f32>,
  //pub crop_image: SharedDeviceBuf<f32>,
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
      //in_bytes:   DeviceBuf::with_zeros(raw_length),
      //raw_image:  DeviceBuf::with_zeros(raw_length),
      //crop_image: Rc::new(RefCell::new(DeviceBuf::with_zeros(crop_length))),
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

  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    assert!(batch_idx < self.batch_lim);
    let DataLayerConfig{
      raw_width, raw_height, channels, ..} = self.config;
    let raw_length = raw_width * raw_height * channels;
    &mut self.in_buf_host[batch_idx * raw_length .. (batch_idx + 1) * raw_length]
  }

  fn preload_frame(&mut self, batch_idx: usize, frame: &Array3d<u8>, ctx: &DeviceContext) {
    assert!(batch_idx < self.batch_lim);
    let DataLayerConfig{
      raw_width, raw_height, channels, ..} = self.config;
    let raw_length = raw_width * raw_height * channels;
    let frame = frame.as_slice();
    assert_eq!(raw_length, frame.len());
    copy_memory(frame, &mut self.in_buf_host[batch_idx * raw_length .. (batch_idx + 1) * raw_length]);
  }

  fn preload_frame_permute(&mut self, batch_idx: usize, frame: &Array3d<u8>, permute_idx: usize, ctx: &DeviceContext) {
    assert!(batch_idx < self.batch_lim);
    let DataLayerConfig{
      raw_width, raw_height, channels, ..} = self.config;
    let raw_length = raw_width * raw_height * channels;
    let plane_len = raw_width * raw_height;
    let frame = frame.as_slice();
    assert_eq!(raw_length, frame.len());
    // FIXME(20151022): permute_idx has a special meaning at the moment.
    if permute_idx == 0 {
      copy_memory(frame, &mut self.in_buf_host[batch_idx * raw_length .. (batch_idx + 1) * raw_length]);
    } else if permute_idx == 1 {
      let batch_offset = batch_idx * raw_length;
      for c in (0 .. channels) {
        //let c_offset = c + 2 * ((c + 1) % 2) - 1;
        let c_offset = match (c % 2) {
          0 => c + 1,
          1 => c - 1,
          _ => unreachable!(),
        };
        copy_memory(
            &frame[c * plane_len .. (c + 1) * plane_len],
            &mut self.in_buf_host[batch_offset + (c_offset) * plane_len .. batch_offset + (c_offset + 1) * plane_len],
        );
      }
    } else {
      unreachable!();
    }
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

  pub fn load_params_old(&mut self, prefix: &str, ctx: &DeviceContext) {
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
      _ => { unimplemented!(); }
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
      _ => { unimplemented!(); }
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

// TODO(20151104): dense and low-rank dense layers (DenseLayer supports batching
// and replaces FullyConnLayer).

#[derive(Clone, Copy, Debug)]
pub struct DenseLayerConfig;

pub struct DenseLayer;

#[derive(Clone, Copy, Debug)]
pub struct LowRankDenseLayerConfig;

pub struct LowRankDenseLayer;

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
  pub out_act:      SharedDeviceBuf<f32>,
  pub out_delta:    SharedDeviceBuf<f32>,
  pub weights:      DeviceArray2d<f32>,
  pub bias:         DeviceArray2d<f32>,
  pub grad_weights: DeviceArray2d<f32>,
  pub grad_bias:    DeviceArray2d<f32>,

  //pub in_col:       DeviceArray2d<f32>,
  //pub grad_col:     DeviceArray2d<f32>,
  //pub unit:         DeviceArray2d<f32>,
  //pub work:         DeviceBuf<f32>,

  pub work_space:     DeviceBuf<u8>,
  pub conv_fwd_op:    CudnnConvFwdOp,
  pub conv_bwd_w_op:  CudnnConvBwdFilterOp,
  pub conv_bwd_d_op:  CudnnConvBwdDataOp,
  pub bias_op:        CudnnAddOp,
  pub act_op:         Option<CudnnActOp>,

  pub stats:          LayerStats,
  pub out_act_nz:     DeviceBuf<f32>,
  pub out_act_nz_tmp: DeviceBuf<f32>,
  pub out_act_nz_h:   Array2d<f32>,
  pub grad_nz:        DeviceBuf<f32>,
  pub grad_nz_tmp:    DeviceBuf<f32>,
  pub grad_nz_h:      Array2d<f32>,

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

    /*let mut unit = DeviceArray2d::with_zeros((out_width * out_height, 1));
    // FIXME: or .col_vector_scale().
    unit.as_mut_view().set_constant(1.0, ctx);*/

    let mut work_size = 0;
    let conv_fwd_op = CudnnConvFwdOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        &ctx.dnn,
    ).unwrap();
    work_size = max(work_size, conv_fwd_op.work_size);
    let conv_bwd_w_op = CudnnConvBwdFilterOp::create_fastest(
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        &ctx.dnn,
    ).unwrap();
    work_size = max(work_size, conv_bwd_w_op.work_size);
    let conv_bwd_d_op = CudnnConvBwdDataOp::create_fastest(
        CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        &ctx.dnn,
    ).unwrap();
    work_size = max(work_size, conv_bwd_d_op.work_size);

    let bias_op = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
    );
    let act_op = match config.act_fun {
      ActivationFunction::Identity |
      ActivationFunction::BoundedRect => None,
      act_fun => {
        let act_kind = match act_fun {
          ActivationFunction::Rect      => CudnnActKind::Relu,
          ActivationFunction::Sigmoid   => CudnnActKind::Sigmoid,
          ActivationFunction::Tanh      => CudnnActKind::Tanh,
          _ => unreachable!(),
        };
        Some(CudnnActOp::new(
            act_kind,
            CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
        ))
      }
    };

    Conv2dLayer{
      in_act:       prev_layer.unwrap().output_activation(-1).unwrap(),
      in_delta:     prev_layer.unwrap().output_delta(-1),
      out_act:      Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length * batch_size))),
      out_delta:    Rc::new(RefCell::new(DeviceBuf::with_zeros(out_length * batch_size))),
      weights:      DeviceArray2d::with_zeros((conv_size * conv_size * in_channels, out_channels)),
      bias:         DeviceArray2d::with_zeros((1, out_channels)),
      grad_weights: DeviceArray2d::with_zeros((conv_size * conv_size * in_channels, out_channels)),
      grad_bias:    DeviceArray2d::with_zeros((1, out_channels)),

      // XXX(20151016): note that im2col buffers are NOT duplicated for batch.
      //in_col:       DeviceArray2d::with_zeros((out_width * out_height, conv_size * conv_size * in_channels)),
      //grad_col:     DeviceArray2d::with_zeros((out_width * out_height, conv_size * conv_size * in_channels)),
      //unit:         unit,
      //work:         DeviceBuf::with_zeros(1024), // FIXME(20151002): for cuDNN work space.

      work_space:     DeviceBuf::with_zeros(work_size),
      conv_fwd_op:    conv_fwd_op,
      conv_bwd_w_op:  conv_bwd_w_op,
      conv_bwd_d_op:  conv_bwd_d_op,
      bias_op:        bias_op,
      act_op:         act_op,

      stats:          LayerStats::default(),
      out_act_nz:     DeviceBuf::with_zeros(1),
      out_act_nz_tmp: DeviceBuf::with_zeros(2 * ((out_length * batch_size)+1024-1)/1024),
      out_act_nz_h:   Array2d::with_zeros((1, 1)),
      grad_nz:        DeviceBuf::with_zeros(1),
      grad_nz_tmp:    DeviceBuf::with_zeros(2 * ((conv_size * conv_size * in_channels * out_channels)+1024-1)/1024),
      grad_nz_h:      Array2d::with_zeros((1, 1)),

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
      ParamsInitialization::None => {
        panic!("params initialization explicitly disabled!");
      }
      ParamsInitialization::Uniform{half_range} => {
        let dist = Range::new(-half_range as f64, half_range as f64);
        for w in init_weights.as_mut_view().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
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

  fn load_params(&mut self, blob: &[u8], ctx: &DeviceContext) -> usize {
    let Conv2dLayerConfig{in_channels, conv_size, out_channels, ..} = self.config;
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("Conv2dLayer failed to deserialize weights!");
    let load_bias = Array2d::deserialize(&mut reader)
      .ok().expect("Conv2dLayer failed to deserialize bias!");
    assert_eq!((conv_size * conv_size * in_channels, out_channels), load_weights.as_view().get_bound());
    assert_eq!((1, out_channels), load_bias.as_view().get_bound());
    self.weights.as_mut_view().sync_load(&load_weights.as_view(), ctx);
    self.bias.as_mut_view().sync_load(&load_bias.as_view(), ctx);
    let progress = reader.position() as usize;
    println!("DEBUG: conv2d layer: load params: read {}", progress);
    progress
  }

  fn save_params(&self, ctx: &DeviceContext) -> Vec<u8> {
    let mut blob = Vec::new();
    let mut save_weights = Array2d::with_zeros(self.weights.as_view().get_bound());
    let mut save_bias = Array2d::with_zeros(self.bias.as_view().get_bound());
    self.weights.as_view().sync_store(&mut save_weights.as_mut_view(), ctx);
    self.bias.as_view().sync_store(&mut save_bias.as_mut_view(), ctx);
    //(&save_weights.as_view() as &ArraySerialize<f32, (usize, usize), NdArrayFormat>)
    save_weights.as_view().serialize(&mut blob);
    //(&save_bias.as_view() as &ArraySerialize<f32, (usize, usize), NdArrayFormat>)
    save_bias.as_view().serialize(&mut blob);
    blob
  }

  fn reset_stats(&mut self) {
    self.stats = LayerStats::default();
  }

  fn stats(&self) -> &LayerStats {
    &self.stats
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

    let &mut Conv2dLayer{
      ref in_act, ref mut out_act,
      ref mut work_space,
      ref weights, ref bias,
      //ref mut in_col, ref unit,
      ref mut out_act_nz, ref mut out_act_nz_tmp, ref mut out_act_nz_h,
    ..} = self;

    // FIXME(20151019): if the batch_size is less than self.batch_cap, then need
    // to correct the batch size within the ops as well.
    unsafe { self.conv_fwd_op.forward(
        in_act.borrow().as_view().as_ptr(),
        weights.as_view().as_ptr(),
        out_act.borrow_mut().as_mut_view().as_mut_ptr(),
        work_space.as_mut_view().as_mut_ptr(),
        &ctx.dnn,
    ).unwrap() };
    unsafe { self.bias_op.forward(
        bias.as_view().as_ptr(),
        out_act.borrow_mut().as_mut_view().as_mut_ptr(),
        &ctx.dnn,
    ).unwrap() };
    match self.config.act_fun {
      ActivationFunction::Identity => {}
      ActivationFunction::Sigmoid |
      ActivationFunction::Tanh => {
        // FIXME(20151103): noticed that cudnn activations are slower
        // (for larger nets, effect size of about 2%).
        unsafe { self.act_op.as_ref().unwrap().forward_in_place(
            out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            &ctx.dnn,
        ).unwrap() };
      }
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_inplace(
            out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            out_length as i32,
            batch_size as i32,
            ctx.stream.ptr,
        ) };
      }
      ActivationFunction::BoundedRect => {
        unsafe { rembrandt_kernel_batch_map_bounded_rect_inplace(
            out_act.borrow_mut().as_mut_view().as_mut_ptr(),
            out_length as i32,
            batch_size as i32,
            ctx.stream.ptr,
        ) };
      }
    }

    // TODO(20151104): add flag for accumulating stats.
    if false {
      unsafe { rembrandt_kernel_reduce_count_nonzero(
          out_act.borrow().as_view().as_ptr(),
          (out_length * batch_size) as i32,
          out_act_nz_tmp.as_mut_view().as_mut_ptr(),
          out_act_nz.as_mut_view().as_mut_ptr(),
          ctx.stream.ptr,
      ) };
      out_act_nz.as_view_2d((1, 1)).sync_store(&mut out_act_nz_h.as_mut_view(), ctx);
      self.stats.num_nonzero_acts += out_act_nz_h.as_slice()[0] as i64;
      self.stats.num_total_acts += (out_length * batch_size) as i64;
    }

    /*for batch_idx in (0 .. batch_size) {
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
    }*/
  }

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
    let Conv2dLayerConfig{
      in_width, in_height, in_channels,
      conv_size, conv_stride, conv_pad,
      out_channels, ..} = self.config;
    let (out_width, out_height, _) = self.config.get_out_dims();
    let in_length = in_width * in_height * in_channels;
    let out_length = out_width * out_height * out_channels;

    let &mut Conv2dLayer{
      ref in_act, ref mut in_delta, ref out_act, ref out_delta,
      ref mut grad_weights, ref mut grad_bias,
      ref mut work_space,
      ref weights, ref bias,
      //ref mut in_col, ref mut grad_col, ref unit,
      ref mut grad_nz, ref mut grad_nz_tmp, ref mut grad_nz_h,
    ..} = self;

    match self.config.act_fun {
      ActivationFunction::Identity => {}
      ActivationFunction::Sigmoid |
      ActivationFunction::Tanh => {
        // FIXME(20151103): noticed that cudnn activations are slower
        // (for larger nets, effect size of about 2%).
        unsafe { self.act_op.as_ref().unwrap().backward_in_place(
            in_act.borrow().as_view().as_ptr(),
            out_act.borrow().as_view().as_ptr(),
            out_delta.borrow_mut().as_mut_view().as_mut_ptr(),
            &ctx.dnn,
        ).unwrap() };
      }
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_backprop_inplace(
            out_act.borrow().as_view().as_ptr(),
            out_length as i32,
            batch_size as i32,
            out_delta.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      ActivationFunction::BoundedRect => {
        unsafe { rembrandt_kernel_batch_map_bounded_rect_backprop_inplace(
            out_act.borrow().as_view().as_ptr(),
            out_length as i32,
            batch_size as i32,
            out_delta.borrow_mut().as_mut_view().as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
    }
    unsafe { self.conv_bwd_w_op.backward_filter(
        in_act.borrow().as_view().as_ptr(),
        out_delta.borrow().as_view().as_ptr(),
        grad_weights.as_mut_view().as_mut_ptr(),
        work_space.as_mut_view().as_mut_ptr(),
        &ctx.dnn,
    ).unwrap() };
    unsafe { self.conv_bwd_w_op.backward_bias(
        out_delta.borrow().as_view().as_ptr(),
        grad_bias.as_mut_view().as_mut_ptr(),
        &ctx.dnn,
    ).unwrap() };
    if let &mut Some(ref mut in_delta) = in_delta {
      unsafe { self.conv_bwd_d_op.backward_data(
          weights.as_view().as_ptr(),
          out_delta.borrow().as_view().as_ptr(),
          in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
          work_space.as_mut_view().as_mut_ptr(),
          &ctx.dnn,
      ).unwrap() };
    }

    // TODO(20151104): add flag for accumulating stats.
    if false {
      unsafe { rembrandt_kernel_reduce_count_sparse(
          grad_weights.as_view().as_ptr(),
          grad_weights.as_view().len() as i32,
          grad_nz_tmp.as_mut_view().as_mut_ptr(),
          grad_nz.as_mut_view().as_mut_ptr(),
          1.0e-4f32,
          ctx.stream.ptr,
      ) };
      grad_nz.as_view_2d((1, 1)).sync_store(&mut grad_nz_h.as_mut_view(), ctx);
      self.stats.num_nonzero_grad += grad_nz_h.as_slice()[0] as i64;
      self.stats.num_total_grad += grad_weights.as_view().len() as i64;
    }

    /*for batch_idx in (0 .. batch_size) {
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
    }*/
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
  pub do_mask:        bool,
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
  pub loss_accum:     DeviceArray2d<f32>,
  pub loss:           DeviceArray2d<f32>,
  pub loss_host:      Array2d<f32>,
  //pub normalization:  DeviceArray2d<f32>,
  pub tmp_max:        DeviceArray2d<f32>,

  //pub category_guess: DeviceArray2d<i32>,
  //pub category_truth: Vec<i32>,

  //pub zero_mask:        DeviceArray2d<f32>,
  //pub zero_mask_host:   Array2d<f32>,
  pub mask:             DeviceArray2d<f32>,
  pub mask_host:        Array2d<f32>,

  pub true_labels:      DeviceArray2d<i32>,
  pub true_labels_host: Array2d<i32>,
  pub pred_labels:      DeviceArray2d<i32>,
  pub pred_labels_host: Array2d<i32>,
  pub ranked_labels:    Vec<i32>,
  pub pred_probs_host:  Array2d<f32>,
  pub pred_cdfs:        DeviceArray2d<f32>,
  pub pred_cdfs_host:   Array2d<f32>,
  pub pred_qvals:       DeviceArray2d<f32>,
  pub pred_qvals_host:  Array2d<f32>,

  pub softmax_op:     CudnnSoftmaxOp,

  pub layer_id:       usize,
  pub config:         SoftmaxLossLayerConfig,
  pub batch_lim:      usize,
}

impl SoftmaxLossLayer {
  pub fn new(layer_id: usize, config: SoftmaxLossLayerConfig, batch_size: usize, prev_layer: Option<&Layer>) -> SoftmaxLossLayer {
    let softmax_op = CudnnSoftmaxOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, batch_size).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(1, 1, config.num_categories, batch_size).unwrap(),
    );
    SoftmaxLossLayer{
      in_act:         prev_layer.unwrap().output_activation(-1).unwrap(),
      in_delta:       prev_layer.unwrap().output_delta(-1).unwrap(),
      probabilities:  DeviceArray2d::with_zeros((config.num_categories, batch_size)),
      loss_accum:     DeviceArray2d::with_zeros((1, batch_size)),
      loss:           DeviceArray2d::with_zeros((1, 1)),
      loss_host:      Array2d::with_zeros((1, 1)),
      //normalization:  DeviceArray2d::with_zeros((1, batch_size)),
      tmp_max:        DeviceArray2d::with_zeros((1, batch_size)),

      //category_guess: DeviceArray2d::with_zeros((1, batch_size)),
      //category_truth: Vec::new(),

      //zero_mask:        DeviceArray2d::with_zeros((config.num_categories, batch_size)),
      //zero_mask_host:   Array2d::with_zeros((config.num_categories, batch_size)),
      mask:             DeviceArray2d::with_zeros((config.num_categories, batch_size)),
      mask_host:        Array2d::with_zeros((config.num_categories, batch_size)),

      true_labels:      DeviceArray2d::with_zeros((1, batch_size)),
      true_labels_host: Array2d::with_zeros((1, batch_size)),
      pred_labels:      DeviceArray2d::with_zeros((1, batch_size)),
      pred_labels_host: Array2d::with_zeros((1, batch_size)),
      ranked_labels:    repeat(0).take(config.num_categories * batch_size).collect(),
      pred_probs_host:  Array2d::with_zeros((config.num_categories, batch_size)),
      pred_cdfs:        DeviceArray2d::with_zeros((config.num_categories, batch_size)),
      pred_cdfs_host:   Array2d::with_zeros((config.num_categories, batch_size)),
      pred_qvals:       DeviceArray2d::with_zeros((config.num_categories, batch_size)),
      pred_qvals_host:  Array2d::with_zeros((config.num_categories, batch_size)),

      softmax_op:     softmax_op,

      layer_id:       layer_id,
      config:         config,
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

  fn reset_objective(&mut self, ctx: &DeviceContext) {
    self.loss_accum.as_mut_view().row_vector_scale(0.0, ctx);
    self.loss.as_mut_view().row_vector_scale(0.0, ctx);
  }

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);

    /*let &mut SoftmaxLossLayer{
      ref in_act, ref mut probabilities,
      ref mut tmp_max, 
      ref mut pred_labels,
      ref zero_mask,
      ..} = self;*/
    if self.config.do_mask {
      unsafe { rembrandt_kernel_batch_map_pos_mask_inplace(
          self.in_act.borrow_mut().as_mut_view().as_mut_ptr(),
          self.config.num_categories as i32,
          batch_size as i32,
          self.mask.as_view().as_ptr(),
          ctx.stream.ptr,
      ) };
    }
    unsafe { self.softmax_op.forward(
        self.in_act.borrow().as_view().as_ptr(),
        self.probabilities.as_mut_view().as_mut_ptr(),
        &ctx.dnn,
    ).unwrap() };

    /*let &mut SoftmaxLossLayer{
      ref mut probabilities,
      ref mut tmp_max,
      ref mut pred_labels, ..} = self;
    for batch_idx in (0 .. batch_size) {
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
    }*/

    /*self.in_act.borrow().as_view_2d((self.num_categories, batch_size))
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
    }*/
  }

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);

    unsafe { rembrandt_kernel_batch_map_softmax_cross_entropy_loss(
        self.probabilities.as_view().as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_labels.as_view().as_ptr(),
        self.loss_accum.as_mut_view().as_mut_ptr(),
        descent.minibatch_size() as f32,
        ctx.stream.ptr,
    ) };
    // TODO(20151108): reduce the minibatch loss.

    unsafe { rembrandt_kernel_batch_map_softmax_cross_entropy_loss_backprop(
        self.probabilities.as_view().as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.true_labels.as_view().as_ptr(),
        self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
        descent.minibatch_size() as f32,
        ctx.stream.ptr,
    ) };

    // XXX(20151028): Do not mask in backward pass!
    /*if self.config.do_mask {
      unsafe { rembrandt_kernel_batch_map_zero_mask_inplace(
          self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
          self.config.num_categories as i32,
          batch_size as i32,
          self.zero_mask.as_view().as_ptr(),
          ctx.stream.ptr,
      ) };
    }*/

    /*for batch_idx in (0 .. batch_size) {
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
    }*/
  }

  fn preload_label(&mut self, batch_idx: usize, label: i32, ctx: &DeviceContext) {
    assert!(batch_idx < self.batch_lim);
    self.true_labels_host.as_mut_slice()[batch_idx] = label;
  }

  fn load_labels(&mut self, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);
    self.true_labels.as_mut_view().sync_load(&self.true_labels_host.as_view(), ctx);
  }

  fn preload_mask(&mut self, batch_idx: usize, mask: &Array2d<f32>, ctx: &DeviceContext) {
    assert!(batch_idx < self.batch_lim);
    assert!(self.config.do_mask);
    self.mask_host.as_mut_slice()[batch_idx * self.config.num_categories .. (batch_idx + 1) * self.config.num_categories].clone_from_slice(mask.as_slice());
  }

  /*fn preload_mask_permute(&mut self, batch_idx: usize, mask: &Array2d<f32>, permute_idx: usize, ctx: &DeviceContext) {
    assert!(batch_idx < self.batch_lim);
    let raw_length = raw_width * raw_height * channels;
    let plane_len = raw_width * raw_height;
    let frame = mask.as_slice();
    assert_eq!(raw_length, frame.len());
    // FIXME(20151022): permute_idx has a special meaning at the moment.
    if permute_idx == 0 {
      copy_memory(frame, &mut self.mask_host[batch_idx * raw_length .. (batch_idx + 1) * raw_length]);
    } else if permute_idx == 1 {
      let batch_offset = batch_idx * raw_length;
      for c in (0 .. channels) {
        let c_offset = 2 * ((c + 1) % 2) - 1;
        copy_memory(
            &frame[c * plane_len .. (c + 1) * plane_len],
            &mut self.mask_host[batch_offset + (c + c_offset) * plane_len .. batch_offset + (c + 1 + c_offset) * plane_len],
        );
      }
    } else {
      unreachable!();
    }
  }*/

  fn load_masks(&mut self, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);
    assert!(self.config.do_mask);
    self.mask.as_mut_view().sync_load(&self.mask_host.as_view(), ctx);
  }

  fn store_labels(&mut self, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);
    assert!(self.config.num_categories <= 1024);
    /*println!("DEBUG: store_labels:");
    println!("DEBUG:   true labels:          {:?}", self.true_labels_host);
    println!("DEBUG:   pred labels (BEFORE): {:?}", self.pred_labels_host);*/
    unsafe { rembrandt_kernel_batch_blockreduce_argmax(
        self.probabilities.as_view().as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.tmp_max.as_mut_view().as_mut_ptr(),
        self.pred_labels.as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.pred_labels.as_view().sync_store(&mut self.pred_labels_host.as_mut_view(), ctx);
    //println!("DEBUG:   pred labels (AFTER):  {:?}", self.pred_labels_host);
  }

  fn predict_labels(&mut self, batch_size: usize, ctx: &DeviceContext) -> &[i32] {
    assert!(batch_size == self.batch_lim);
    self.pred_labels_host.as_slice()
  }

  fn count_accuracy(&mut self, batch_size: usize, ctx: &DeviceContext) -> usize {
    assert!(batch_size == self.batch_lim);
    /*println!("DEBUG: count_accuracy:");
    println!("DEBUG:   true labels: {:?}", self.true_labels_host);
    println!("DEBUG:   pred labels: {:?}", self.pred_labels_host);*/
    let mut num_correct = 0;
    for (&y_truth, &y_hat) in self.true_labels_host.as_slice().iter().zip(self.pred_labels_host.as_slice().iter()) {
      if y_truth == y_hat {
        num_correct += 1;
      }
    }
    num_correct
  }

  fn store_ranked_labels(&mut self, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);
    self.store_probs(batch_size, ctx);
  }

  fn predict_ranked_labels(&mut self, batch_size: usize) -> &[i32] {
    assert!(batch_size == self.batch_lim);
    let mut tmp_labels: Vec<(i32, i32)> = repeat((0, 0)).take(self.config.num_categories).collect();
    for batch_idx in (0 .. batch_size) {
      for j in (0 .. self.config.num_categories) {
        let prob = self.pred_probs_host.as_slice()[batch_idx * self.config.num_categories + j];
        // FIXME(20151108): a nasty hack casting f32 to i32 in order to call .sort();
        // one of the more annoying parts of rust.
        tmp_labels[j] = ((-prob * 1.0e6) as i32, j as i32);
      }
      tmp_labels.sort();
      for k in (0 .. self.config.num_categories) {
        self.ranked_labels[batch_idx * self.config.num_categories + k] = tmp_labels[k].1;
      }
    }
    &self.ranked_labels
  }

  fn store_loss(&mut self, batch_size: usize, ctx: &DeviceContext) {
    // TODO(20151108)
    unimplemented!();
  }

  fn predict_loss(&mut self, batch_size: usize, ctx: &DeviceContext) -> f32 {
    // TODO(20151108)
    unimplemented!();
  }

  fn store_probs(&mut self, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);
    self.probabilities.as_view().sync_store(&mut self.pred_probs_host.as_mut_view(), ctx);
  }

  fn predict_probs(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32> {
    assert!(batch_size == self.batch_lim);
    &self.pred_probs_host
  }

  fn store_cdfs(&mut self, batch_size: usize, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);
    assert!(self.config.num_categories <= 1024);
    unsafe { rembrandt_kernel_batch_blockscan_prefix_sum(
        self.probabilities.as_view().as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.pred_cdfs.as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.pred_cdfs.as_view().sync_store(&mut self.pred_cdfs_host.as_mut_view(), ctx);
  }

  fn predict_cdfs(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32> {
    assert!(batch_size == self.batch_lim);
    &self.pred_cdfs_host
  }

  /*fn store_boltzmann_q(&mut self, batch_size: usize, beta: f32, ctx: &DeviceContext) {
    assert!(batch_size == self.batch_lim);
    unsafe { rembrandt_kernel_batch_map_boltzmann_q_transform(
        self.probabilities.as_view().as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        beta,
        self.pred_qvals.as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    self.pred_qvals.as_view().sync_store(&mut self.pred_qvals_host.as_mut_view(), ctx);
  }

  fn predict_boltzmann_q(&mut self, batch_size: usize, ctx: &DeviceContext) -> &Array2d<f32> {
    assert!(batch_size == self.batch_lim);
    &self.pred_qvals_host
  }*/
}

#[derive(Clone, Copy, Debug)]
pub struct MultiBinLogisticLossLayerConfig {
  pub num_categories: usize,
  pub do_mask:        bool,
}

impl LayerConfig for MultiBinLogisticLossLayerConfig {
  fn get_in_dims(&self) -> (usize, usize, usize) {
    (1, 1, self.num_categories)
  }

  fn get_out_dims(&self) -> (usize, usize, usize) {
    (1, 1, self.num_categories)
  }
}

pub struct MultiBinLogisticLossLayer {
  pub in_act:       SharedDeviceBuf<f32>,
  pub in_delta:     SharedDeviceBuf<f32>,
  pub out_act:      DeviceBuf<f32>,
  pub cat_labels:   DeviceBuf<i32>,
  pub cat_labels_h: Vec<i32>,
  pub bin_labels:   DeviceBuf<i32>,
  pub bin_labels_h: Vec<i32>,
  pub mask:         DeviceBuf<f32>,
  pub mask_h:       Vec<f32>,

  pub layer_id: usize,
  pub config:   MultiBinLogisticLossLayerConfig,
}

impl MultiBinLogisticLossLayer {
  // TODO(20151101)
}

impl Layer for MultiBinLogisticLossLayer {
  fn get_id(&self) -> usize {
    self.layer_id
  }

  fn output_activation(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> { None }
  fn output_delta(&self, out_layer_id: usize) -> Option<SharedDeviceBuf<f32>> { None }

  fn forward(&mut self, phase: OptPhase, batch_size: usize, ctx: &DeviceContext) {
    // TODO(20151101)
    unsafe { rembrandt_kernel_batch_map_multi_bin_logistic(
        self.in_act.borrow().as_view().as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.out_act.as_mut_view().as_mut_ptr(),
        ctx.stream.ptr,
    ) };
    if self.config.do_mask {
      unsafe { rembrandt_kernel_batch_map_pos_mask_inplace(
          self.out_act.as_mut_view().as_mut_ptr(),
          self.config.num_categories as i32,
          batch_size as i32,
          self.mask.as_view().as_ptr(),
          ctx.stream.ptr,
      ) };
    }
  }

  fn backward(&mut self, descent: &DescentSchedule, batch_size: usize, ctx: &DeviceContext) {
    // TODO(20151101)
    unsafe { rembrandt_kernel_batch_map_multi_bin_logistic_xent_loss_backprop(
        self.out_act.as_view().as_ptr(),
        self.config.num_categories as i32,
        batch_size as i32,
        self.cat_labels.as_view().as_ptr(),
        self.bin_labels.as_view().as_ptr(),
        self.in_delta.borrow_mut().as_mut_view().as_mut_ptr(),
        null_mut(),
        descent.minibatch_size() as f32,
        ctx.stream.ptr,
    ) };
  }

  // TODO(20151101)
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
