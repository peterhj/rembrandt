use data_new::{SampleLabel};
use operator::{
  Operator, InputOperator, SharedDeviceBuf, OpPhase, OpCapability,
};
use operator::comm::{CommWorker};
use operator::conv::{
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
use array_cuda::device::linalg::{BlasMatrixExt, BlasVectorExt, Transpose};
use array_cuda::device::memory::{DeviceBufferInitExt, DeviceBuffer};
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
use std::cell::{RefCell};
use std::cmp::{max, min};
use std::fs::{File};
use std::io::{Cursor};
use std::iter::{repeat};
use std::marker::{PhantomData};
use std::path::{PathBuf};
use std::rc::{Rc};

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

  /*fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_buf.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    None
  }*/

  fn get_output_act(&self, _arm: usize) -> SharedDeviceBuf<f32> {
    assert_eq!(0, _arm);
    self.out_buf.clone()
  }

  fn get_output_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    assert_eq!(0, _arm);
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

#[derive(Clone, Debug)]
pub enum VarData3dPreprocConfig {
  /*AddPixelwiseColorNoise{
    brightness_hrange:  f32,
    contrast_hrange:    f32,
    saturation_hrange:  f32,
  },*/
  AddPixelwisePCALightingNoise{
    singular_vecs:  Vec<Vec<f32>>,
    singular_vals:  Vec<f32>,
    std_dev:    f32,
  },
  Crop{
    crop_width:     usize,
    crop_height:    usize,
  },
  FlipX,
  ScaleBicubic{
    scale_lower:    usize,
    scale_upper:    usize,
  },
}

#[derive(Clone, Debug)]
pub struct VarData3dOperatorConfig {
  //pub max_in_dims:  (usize, usize, usize),
  pub in_stride:    usize,
  pub out_dims:     (usize, usize, usize),
  pub normalize:    bool,
  pub preprocs:     Vec<VarData3dPreprocConfig>,
}

impl VarData3dOperatorConfig {
  pub fn get_out_dims(&self) -> (usize, usize, usize) {
    self.out_dims
    /*let mut out_dims = self.in_dims;
    for preproc in self.preprocs.iter() {
      match preproc {
        /*&VarData3dPreproc::AddPixelwiseColorNoise{..} => {
          // Do nothing.
        }*/
        &VarData3dPreproc::AddPixelwisePCALightingNoise{..} => {
          // Do nothing.
        }
        &VarData3dPreproc::Crop{crop_width, crop_height} => {
          assert!(crop_width <= out_dims.0);
          assert!(crop_height <= out_dims.1);
          out_dims = (crop_width, crop_height, out_dims.2);
          assert!(self.in_dims.len() >= out_dims.len());
        }
        &VarData3dPreproc::FlipX => {
          // Do nothing.
        }
        &VarData3dPreproc::ScaleBicubic{..} => {
          // Do nothing.
        }
      }
    }
    out_dims*/
  }
}

pub enum VarData3dPreprocState {
  //AddPixelwiseColorNoise,
  AddPixelwisePCALightingNoise{
    dist:       GaussianDist<f32>,
    svecs_buf:  DeviceBuffer<f32>,
    svals_buf:  DeviceBuffer<f32>,
    alphas_buf: DeviceBuffer<f32>,
  },
  Crop,
  FlipX{
    coin_flip:  Range<usize>,
  },
  ScaleBicubic,
}

pub struct VarData3dOperator {
  batch_cap:    usize,
  config:       VarData3dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_buf_h:     Vec<u8>,
  in_buf:       DeviceBuffer<u8>,

  in_shapes:    Vec<(usize, usize, usize)>,
  in_widths_h:  Vec<i32>,
  in_heights_h: Vec<i32>,
  in_widths:    DeviceBuffer<i32>,
  in_heights:   DeviceBuffer<i32>,
  in_x_offsets_h:   Vec<i32>,
  in_y_offsets_h:   Vec<i32>,
  in_x_offsets: DeviceBuffer<i32>,
  in_y_offsets: DeviceBuffer<i32>,

  tmp_buf:      DeviceBuffer<f32>,
  out_buf:      SharedDeviceBuf<f32>,

  rng:          Xorshiftplus128Rng,
  preprocs:     Vec<VarData3dPreprocState>,

  r_forward:    Option<VarData3dRFwdOperator>,
}

struct VarData3dRFwdOperator {
  out_r_buf:    SharedDeviceBuf<f32>,
}

impl VarData3dOperator {
  pub fn new(batch_size: usize, capability: OpCapability, config: VarData3dOperatorConfig, context: Rc<DeviceContext>) -> VarData3dOperator {
    let ctx = &(*context).as_ref();
    /*let in_dims = config.max_in_dims;
    let (in_width, in_height, in_channels) = in_dims;
    let in_frame_len = in_dims.len();*/
    let in_stride = config.in_stride;
    let out_dims = config.out_dims;
    let out_frame_len = out_dims.len();
    let max_frame_len = max(in_stride, out_frame_len);

    let mut preprocs = Vec::with_capacity(config.preprocs.len());
    for preproc_cfg in config.preprocs.iter() {
      match preproc_cfg {
        /*&Data3dPreproc::AddPixelwiseColorNoise{..} => {
          // FIXME(20160422)
          unimplemented!();
        }*/

        &VarData3dPreprocConfig::AddPixelwisePCALightingNoise{ref singular_vecs, ref singular_vals, std_dev} => {
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
          preprocs.push(VarData3dPreprocState::AddPixelwisePCALightingNoise{
            dist:       GaussianDist{mean: 0.0, std: std_dev},
            svecs_buf:  svecs_buf,
            svals_buf:  svals_buf,
            alphas_buf: alphas_buf,
          });
        }

        &VarData3dPreprocConfig::Crop{crop_width, crop_height} => {
          preprocs.push(VarData3dPreprocState::Crop);
        }

        &VarData3dPreprocConfig::FlipX => {
          preprocs.push(VarData3dPreprocState::FlipX{
            coin_flip:  Range::new(0, 2),
          });
        }

        &VarData3dPreprocConfig::ScaleBicubic{..} => {
          preprocs.push(VarData3dPreprocState::ScaleBicubic);
        }
      }
    }

    let r_forward = if capability.r_forward_enabled() {
      Some(VarData3dRFwdOperator{
        out_r_buf:  Rc::new(RefCell::new(DeviceBuffer::zeros(out_frame_len * batch_size, ctx))),
      })
    } else {
      None
    };

    VarData3dOperator{
      batch_cap:    batch_size,
      config:       config,
      context:      context.clone(),
      in_buf_h:     repeat(0).take(in_stride * batch_size).collect(),
      in_buf:       DeviceBuffer::zeros(in_stride * batch_size, ctx),
      in_shapes:    repeat((0, 0, 0)).take(batch_size).collect(),
      in_widths_h:  repeat(0).take(batch_size).collect(),
      in_heights_h: repeat(0).take(batch_size).collect(),
      in_widths:    DeviceBuffer::zeros(batch_size, ctx),
      in_heights:   DeviceBuffer::zeros(batch_size, ctx),
      in_x_offsets_h:   repeat(0).take(batch_size).collect(),
      in_y_offsets_h:   repeat(0).take(batch_size).collect(),
      in_x_offsets: DeviceBuffer::zeros(batch_size, ctx),
      in_y_offsets: DeviceBuffer::zeros(batch_size, ctx),
      // FIXME(20160506): assuming `in_stride` is larger than `out_stride`.
      tmp_buf:      DeviceBuffer::zeros(in_stride * batch_size, ctx),
      out_buf:      Rc::new(RefCell::new(DeviceBuffer::zeros(in_stride * batch_size, ctx))),
      rng:          Xorshiftplus128Rng::new(&mut thread_rng()),
      preprocs:     preprocs,
      r_forward:    r_forward,
    }
  }
}

impl Operator for VarData3dOperator {
  fn upcast_input(&mut self) -> &mut InputOperator {
    self
  }

  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_act(&self, _arm: usize) -> SharedDeviceBuf<f32> {
    assert_eq!(0, _arm);
    self.out_buf.clone()
  }

  fn get_output_delta(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    assert_eq!(0, _arm);
    None
  }

  fn get_output_r_act(&self, _arm: usize) -> Option<SharedDeviceBuf<f32>> {
    assert_eq!(0, _arm);
    self.r_forward.as_ref().map(|r_fwd| r_fwd.out_r_buf.clone())
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();

    /*let max_in_dims = self.config.max_in_dims;
    let (max_in_width, max_in_height, max_in_channels) = max_in_dims;*/
    let in_stride = self.config.in_stride;
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
        /*( &Data3dPreproc::AddPixelwiseColorNoise{..},
          &mut Data3dPreprocState::AddPixelwiseColorNoise,
        ) => {
        }*/

        ( &VarData3dPreprocConfig::AddPixelwisePCALightingNoise{..},
          &mut VarData3dPreprocState::AddPixelwisePCALightingNoise{ref dist, ref mut alphas_buf, ref mut svals_buf, ref mut svecs_buf},
        ) => {
          match phase {
            OpPhase::Inference => {
              src_buf.send(&mut target_buf);
            }
            OpPhase::Training{..} => {
              // FIXME(20160504): this version uses the output (post-crop) dimensions.
              assert_eq!(3, out_channels);
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

        ( &VarData3dPreprocConfig::Crop{crop_width, crop_height},
          &mut VarData3dPreprocState::Crop,
        ) => {
          /*for batch_idx in 0 .. batch_size {
            let (in_width, in_height, _) = self.in_shapes[batch_idx];
            let (offset_w, offset_h) = match phase {
              OpPhase::Inference => {
                let offset_w = (in_width - crop_width) / 2;
                let offset_h = (in_height - crop_height) / 2;
                (offset_w, offset_h)
              }
              OpPhase::Training{..} => {
                /*let offset_w = woff_range.ind_sample(&mut self.rng);
                let offset_h = hoff_range.ind_sample(&mut self.rng);*/
                let offset_w = self.rng.gen_range(0, in_width - crop_width + 1);
                let offset_h = self.rng.gen_range(0, in_height - crop_height + 1);
                (offset_w, offset_h)
              }
            };
            self.in_widths_h[batch_idx] = in_width as i32;
            self.in_heights_h[batch_idx] = in_height as i32;
            self.in_x_offsets_h[batch_idx] = offset_w as i32;
            self.in_y_offsets_h[batch_idx] = offset_h as i32;
          }
          self.in_widths.as_ref_mut(ctx).sync_load(&self.in_widths_h);
          self.in_heights.as_ref_mut(ctx).sync_load(&self.in_heights_h);
          self.in_x_offsets.as_ref_mut(ctx).sync_load(&self.in_x_offsets_h);
          self.in_y_offsets.as_ref_mut(ctx).sync_load(&self.in_y_offsets_h);
          unsafe { rembrandt_kernel_batch_image3_crop(
              src_buf.as_ptr(),
              in_stride as i32,
              batch_size as i32,
              self.in_widths.as_ref(ctx).as_ptr(),
              self.in_heights.as_ref(ctx).as_ptr(),
              out_channels as i32,
              self.in_x_offsets.as_ref(ctx).as_ptr(),
              self.in_y_offsets.as_ref(ctx).as_ptr(),
              target_buf.as_mut_ptr(),
              crop_width as i32,
              crop_height as i32,
              ctx.stream.ptr,
          ) };*/

          let target_stride = crop_width * crop_height * out_channels;
          for batch_idx in 0 .. batch_size {
            let (in_width, in_height, _) = self.in_shapes[batch_idx];
            assert!(in_width >= crop_width);
            assert!(in_height >= crop_height);
            let (offset_w, offset_h) = match phase {
              OpPhase::Inference => {
                let offset_w = (in_width - crop_width) / 2;
                let offset_h = (in_height - crop_height) / 2;
                (offset_w, offset_h)
              }
              OpPhase::Training{..} => {
                let offset_w = self.rng.gen_range(0, in_width - crop_width + 1);
                let offset_h = self.rng.gen_range(0, in_height - crop_height + 1);
                (offset_w, offset_h)
              }
            };
            unsafe { rembrandt_kernel_image3_crop(
                src_buf.as_ptr().offset((batch_idx * in_stride) as isize),
                in_width as i32,
                in_height as i32,
                out_channels as i32,
                offset_w as i32,
                offset_h as i32,
                target_buf.as_mut_ptr().offset((batch_idx * target_stride) as isize),
                crop_width as i32,
                crop_height as i32,
                ctx.stream.ptr,
            ) };
          }

          for batch_idx in 0 .. batch_size {
            self.in_shapes[batch_idx] = (crop_width, crop_height, out_channels);
          }
        }

        ( &VarData3dPreprocConfig::FlipX,
          &mut VarData3dPreprocState::FlipX{ref coin_flip},
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
                  // FIXME(20160504): this version of flip comes _after_ crop.
                  assert!(out_width <= 256);
                  unsafe { rembrandt_kernel_batch_blockmap256_flip(
                      src_buf.as_ptr(),
                      out_width as i32,
                      (out_height * out_channels * batch_size) as i32,
                      target_buf.as_mut_ptr(),
                      ctx.stream.ptr,
                  ) };
                }
                _ => unreachable!(),
              }
            }
          }
        }

        ( &VarData3dPreprocConfig::ScaleBicubic{scale_lower, scale_upper},
          &mut VarData3dPreprocState::ScaleBicubic,
        ) => {
          // FIXME(20160504): should use a batch bicubic scale.
          for batch_idx in 0 .. batch_size {
            let scale_dim = match phase {
              OpPhase::Inference => {
                scale_lower
              }
              OpPhase::Training{..} => {
                self.rng.gen_range(scale_lower, scale_upper + 1)
              }
            };
            assert!(scale_dim >= scale_lower);
            assert!(scale_dim <= scale_upper);
            let (in_width, in_height, _) = self.in_shapes[batch_idx];
            assert!(in_width > 0);
            assert!(in_height > 0);
            let min_dim = min(in_width, in_height);
            let (scale_width, scale_height) = if min_dim == in_width {
              (scale_dim, (scale_dim as f32 / in_width as f32 * in_height as f32).round() as usize)
            } else if min_dim == in_height {
              ((scale_dim as f32 / in_height as f32 * in_width as f32).round() as usize, scale_dim)
            } else {
              unreachable!();
            };
            if in_width == scale_width && in_height == scale_height {
              src_buf.range(batch_idx * in_stride, (batch_idx+1) * in_stride)
                .send(&mut target_buf.mut_range(batch_idx * in_stride, (batch_idx+1) * in_stride));
            } else {
              unsafe { rembrandt_kernel_image3_catmullrom_scale(
                  src_buf.as_ptr().offset((batch_idx * in_stride) as isize),
                  in_width as i32,
                  in_height as i32,
                  out_channels as i32,
                  target_buf.as_mut_ptr().offset((batch_idx * in_stride) as isize),
                  scale_width as i32,
                  scale_height as i32,
                  ctx.stream.ptr,
              ) };
            }
            self.in_shapes[batch_idx] = (scale_width, scale_height, out_channels);
          }
        }

        _ => unreachable!(),
      }
    }
  }

  fn backward(&mut self, _batch_size: usize) {
    // Do nothing.
  }

  fn r_forward(&mut self, _batch_size: usize) {
    // Do nothing.
  }
}

impl InputOperator for VarData3dOperator {
  fn downcast(&self) -> &Operator {
    self
  }

  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    assert!(batch_idx < self.batch_cap);
    let frame_len = self.config.in_stride;
    &mut self.in_buf_h[batch_idx * frame_len .. (batch_idx + 1) * frame_len]
  }

  fn load_frames(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    let in_stride = self.config.in_stride;
    let &mut VarData3dOperator{ref mut in_buf, ref in_buf_h, ..} = self;
    for batch_idx in 0 .. batch_size {
      let shape = self.in_shapes[batch_idx];
      let frame_len = shape.len();
      assert!(frame_len <= in_stride);
      let mut in_buf = in_buf.as_ref_mut_range(batch_idx * in_stride, batch_idx * in_stride + frame_len, ctx);
      unsafe { in_buf.unsafe_async_load(&in_buf_h[batch_idx * in_stride .. batch_idx * in_stride + frame_len]) };
    }
    ctx.blocking_sync();
    /*{
      // FIXME(20160329): does not use `batch_size` at all!
      let in_buf_h = &self.in_buf_h;
      let mut in_buf = self.in_buf.as_ref_mut(ctx);
      in_buf.sync_load(in_buf_h);
    }*/
  }

  fn stage_shape(&mut self, batch_idx: usize, shape: (usize, usize, usize)) {
    assert!(batch_idx < self.batch_cap);
    /*assert!(shape.0 <= self.config.max_in_dims.0);
    assert!(shape.1 <= self.config.max_in_dims.1);
    assert_eq!(shape.2, self.config.max_in_dims.2);*/
    assert!(shape.len() <= self.config.in_stride);
    self.in_shapes[batch_idx] = shape;
  }

  /*fn load_shapes(&mut self, batch_size: usize) {
    //assert!(batch_size <= self.batch_cap);
  }*/

  fn preload_frame(&mut self, batch_idx: usize) {
    assert!(batch_idx < self.batch_cap);
    let &mut VarData3dOperator{ref mut in_buf, ref in_buf_h, ..} = self;
    let ctx = &(*self.context).as_ref();
    let in_stride = self.config.in_stride;
    let shape = self.in_shapes[batch_idx];
    let frame_len = shape.len();
    assert!(frame_len <= in_stride);
    let mut in_buf = in_buf.as_ref_mut_range(batch_idx * in_stride, batch_idx * in_stride + frame_len, ctx);
    unsafe { in_buf.unsafe_async_load(&in_buf_h[batch_idx * in_stride .. batch_idx * in_stride + frame_len]) };
  }

  fn wait_preload_frames(&mut self, batch_size: usize) {
    assert!(batch_size <= self.batch_cap);
    let ctx = &(*self.context).as_ref();
    ctx.blocking_sync();
  }
}
