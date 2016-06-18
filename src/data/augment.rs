use data::{SampleDatum, SampleLabel};

use array::{ArrayZeroExt, Array3d};
//use epeg::{EpegImage};
//use image::{ImageBuffer, Luma};
//use image::imageops::{FilterType, resize};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};

pub trait TransformPreproc {
  //fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Rng) -> (SampleDatum, Option<SampleLabel>);
  fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>);
}

pub struct TransposeCWH2WHCPreproc;

impl TransformPreproc for TransposeCWH2WHCPreproc {
  fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, _rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>) {
    match datum {
      SampleDatum::CWHBytes(old_bytes) => {
        let dims = old_bytes.bound();
        let (channels, width, height) = dims;

        let mut new_bytes = Array3d::zeros((width, height, channels));
        {
          let old_bytes = old_bytes.as_slice();
          let mut new_bytes = new_bytes.as_mut_slice();
          for k in 0 .. channels {
            for j in 0 .. height {
              for i in 0 .. width {
                new_bytes[i + j * width + k * width * height] =
                    old_bytes[k + i * channels + j * channels * width];
              }
            }
          }
        }

        (SampleDatum::WHCBytes(new_bytes), label)
      }

      _ => unimplemented!(),
    }
  }
}

/*pub struct RandomImageScalePreproc {
  pub min_side_lo:  usize,
  pub min_side_hi:  usize,
}

impl TransformPreproc for RandomImageScalePreproc {
  fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>) {
    let min_side = rng.gen_range(self.min_side_lo, self.min_side_hi + 1);
    match datum {
      SampleDatum::WHCBytes(old_bytes) => {
        let old_dims = old_bytes.bound();
        let (old_width, old_height, channels) = old_dims;
        let (new_width, new_height) = if old_width < old_height {
          (min_side, (min_side as f32 / old_width as f32 * old_height as f32).round() as usize)
        } else if old_width > old_height {
          ((min_side as f32 / old_height as f32 * old_width as f32).round() as usize, min_side)
        } else {
          (min_side, min_side)
        };

        // FIXME(20160425): way too many allocations!
        let mut new_bytes = Vec::with_capacity(new_width * new_height * channels);
        for k in 0 .. channels {
          let old_plane: Vec<_> =
              (&old_bytes.as_slice()[old_width * old_height * k .. old_width * old_height * (k+1)])
                .iter()
                //.map(|&x| Luma{data: [x]})
                .map(|&x| x)
                .collect();
          let old_image_buf: ImageBuffer<Luma<u8>, Vec<u8>> =
              match ImageBuffer::from_vec(old_width as u32, old_height as u32, old_plane) {
                None => panic!("failed to create image buffer from bytes"),
                Some(buf) => buf,
              };
          // FIXME(20160428): image is way too fucking slow, use VIPS instead.
          //let new_image_buf = resize(&old_image_buf, new_width as u32, new_height as u32, FilterType::Lanczos3);
          //let new_image_buf = resize(&old_image_buf, new_width as u32, new_height as u32, FilterType::CatmullRom);
          let new_image_buf = resize(&old_image_buf, new_width as u32, new_height as u32, FilterType::Triangle);
          let new_plane = new_image_buf.to_vec();
          new_bytes.extend_from_slice(&new_plane);
        }

        let new_arr = Array3d::with_data(new_bytes, (new_width, new_height, channels));
        (SampleDatum::WHCBytes(new_arr), label)
      }

      _ => unimplemented!(),
    }
  }
}*/

pub struct RandomFlipXPreproc;

impl TransformPreproc for RandomFlipXPreproc {
  fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>) {
    match datum {
      SampleDatum::WHCBytes(old_bytes) => {
      }
      _ => unimplemented!(),
    }
    // FIXME(20160425)
    unimplemented!();
  }
}

pub struct RandomCropPreproc {
  pub crop_width:   usize,
  pub crop_height:  usize,
}

impl TransformPreproc for RandomCropPreproc {
  fn transform(&self, old_datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>) {
    let new_datum = match old_datum {
      SampleDatum::WHCBytes(old_bytes) => {
        let old_dims = old_bytes.bound();
        let (old_width, old_height, channels) = old_dims;

        let offset_w = if old_width > self.crop_width {
          rng.gen_range(0, old_width - self.crop_width)
        } else if old_width == self.crop_width {
          0
        } else {
          unreachable!()
        };
        let offset_h = if old_height > self.crop_height {
          rng.gen_range(0, old_height - self.crop_height)
        } else if old_height == self.crop_height {
          0
        } else {
          unreachable!()
        };

        let mut new_bytes = Array3d::zeros((self.crop_width, self.crop_height, channels));
        {
          let old_plane_len = old_width * old_height;
          let new_plane_len = self.crop_width * self.crop_height;
          let old_bytes = old_bytes.as_slice();
          let mut new_bytes = new_bytes.as_mut_slice();
          for k in 0 .. channels {
            for j in 0 .. self.crop_height {
              for i in 0 .. self.crop_width {
                new_bytes[i + j * self.crop_width + k * new_plane_len] =
                    old_bytes[offset_w + i + (offset_h + j) * old_width + k * old_plane_len]
              }
            }
          }
        }

        SampleDatum::WHCBytes(new_bytes)
      }

      _ => unimplemented!(),
    };
    (new_datum, label)
  }
}

pub struct CenterCropPreproc {
  pub crop_width:   usize,
  pub crop_height:  usize,
}

impl TransformPreproc for CenterCropPreproc {
  fn transform(&self, old_datum: SampleDatum, label: Option<SampleLabel>, _rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>) {
    let new_datum = match old_datum {
      SampleDatum::WHCBytes(old_bytes) => {
        let old_dims = old_bytes.bound();
        let (old_width, old_height, channels) = old_dims;
        let offset_w = (old_width - self.crop_width) / 2;
        let offset_h = (old_height - self.crop_height) / 2;

        let mut new_bytes = Array3d::zeros((self.crop_width, self.crop_height, channels));
        {
          let old_plane_len = old_width * old_height;
          let new_plane_len = self.crop_width * self.crop_height;
          let old_bytes = old_bytes.as_slice();
          let mut new_bytes = new_bytes.as_mut_slice();
          for k in 0 .. channels {
            for j in 0 .. self.crop_height {
              for i in 0 .. self.crop_width {
                new_bytes[i + j * self.crop_width + k * new_plane_len] =
                    old_bytes[offset_w + i + (offset_h + j) * old_width + k * old_plane_len]
              }
            }
          }
        }

        SampleDatum::WHCBytes(new_bytes)
      }

      _ => unimplemented!(),
    };
    (new_datum, label)
  }
}

/*pub struct EpegRandomScaleCropPreproc {
  pub scale_lo:     usize,
  pub scale_hi:     usize,
  pub crop_width:   usize,
  pub crop_height:  usize,
}

impl TransformPreproc for EpegRandomScaleCropPreproc {
  fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>) {
    match datum {
      SampleDatum::JpegBuffer(jpeg_buf) => {
        let mut image = match EpegImage::open_memory(jpeg_buf) {
          Err(e) => panic!("epeg decode error: {:?}", e),
          Ok(im) => im,
        };

        let scale = rng.gen_range(self.scale_lo, self.scale_hi + 1);
        let (old_width, old_height) = image.get_size();
        let (new_width, new_height) = if old_width < old_height {
          (scale, (scale as f32 / old_width as f32 * old_height as f32).round() as usize)
        } else if old_width > old_height {
          ((scale as f32 / old_height as f32 * old_width as f32).round() as usize, scale)
        } else {
          (scale, scale)
        };

        let offset_w = if new_width > self.crop_width {
          rng.gen_range(0, new_width - self.crop_width + 1)
        } else if new_width == self.crop_width {
          0
        } else {
          unreachable!()
        };
        let offset_h = if new_height > self.crop_height {
          rng.gen_range(0, new_height - self.crop_height + 1)
        } else if new_height == self.crop_height {
          0
        } else {
          unreachable!()
        };

        image.set_decode_size(new_width, new_height);
        image.set_decode_yuv8();
        //image.set_decode_rgb8();
        image.set_quality(100);
        image.enable_thumbnail_comments(false);

        let new_arr = {
          let pixels = match image.get_scaled_pixels(offset_w, offset_h, self.crop_width, self.crop_height) {
            Err(e) => panic!("epeg get pixels error: {:?}", e),
            Ok(pixels) => pixels,
          };
          let new_data = pixels.as_slice().to_vec();
          Array3d::with_data(new_data, (3, self.crop_width, self.crop_height))
        };
        (SampleDatum::CWHBytes(new_arr), label)
      }

      _ => unimplemented!(),
    }
  }
}

pub struct EpegFixedScaleCenterCropPreproc {
  pub scale:        usize,
  pub crop_width:   usize,
  pub crop_height:  usize,
}

impl TransformPreproc for EpegFixedScaleCenterCropPreproc {
  fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, _rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>) {
    match datum {
      SampleDatum::JpegBuffer(jpeg_buf) => {
        let mut image = match EpegImage::open_memory(jpeg_buf) {
          Err(e) => panic!("epeg decode error: {:?}", e),
          Ok(im) => im,
        };

        let (old_width, old_height) = image.get_size();
        let (new_width, new_height) = if old_width < old_height {
          (self.scale, (self.scale as f32 / old_width as f32 * old_height as f32).round() as usize)
        } else if old_width > old_height {
          ((self.scale as f32 / old_height as f32 * old_width as f32).round() as usize, self.scale)
        } else {
          (self.scale, self.scale)
        };

        let offset_w = (new_width - self.crop_width) / 2;
        let offset_h = (new_height - self.crop_height) / 2;

        image.set_decode_size(new_width, new_height);
        image.set_decode_yuv8();
        //image.set_decode_rgb8();
        image.set_quality(100);
        image.enable_thumbnail_comments(false);

        /*match image.scale_pixels(offset_w, offset_h, self.crop_width, self.crop_height) {
          Err(e) => panic!("epeg get pixels error: {:?}", e),
          Ok(pixels) => {
            let new_data = pixels.as_slice().to_vec();
            let new_arr = Array3d::with_data(new_data, (3, self.crop_width, self.crop_height));
            (SampleDatum::CWHBytes(new_arr), label)
          }
        }*/
        let new_arr = {
          let pixels = match image.get_scaled_pixels(offset_w, offset_h, self.crop_width, self.crop_height) {
            Err(e) => panic!("epeg get pixels error: {:?}", e),
            Ok(pixels) => pixels,
          };
          let new_data = pixels.as_slice().to_vec();
          Array3d::with_data(new_data, (3, self.crop_width, self.crop_height))
        };
        (SampleDatum::CWHBytes(new_arr), label)
      }

      _ => unimplemented!(),
    }
  }
}*/
