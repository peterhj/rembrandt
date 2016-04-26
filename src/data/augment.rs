use data_new::{SampleDatum, SampleLabel};

use array_new::{Array3d};
use image::{ImageBuffer, Luma};
use image::imageops::{FilterType, resize};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};

pub trait AugmentPreproc {
  //fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Rng) -> (SampleDatum, Option<SampleLabel>);
  fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>);
}

pub struct RandomScalePreproc {
  pub min_side_lo:  usize,
  pub min_side_hi:  usize,
}

impl AugmentPreproc for RandomScalePreproc {
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
          let new_image_buf = resize(&old_image_buf, new_width as u32, new_height as u32, FilterType::Lanczos3);
          let new_plane = new_image_buf.to_vec();
          new_bytes.extend_from_slice(&new_plane);
        }

        let new_arr = Array3d::with_data(new_bytes, (new_width, new_height, channels));
        (SampleDatum::WHCBytes(new_arr), label)
      }
    }
  }
}

pub struct RandomFlipXPreproc;

impl AugmentPreproc for RandomFlipXPreproc {
  fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>) {
    match datum {
      SampleDatum::WHCBytes(old_bytes) => {
      }
    }
    // FIXME(20160425)
    unimplemented!();
  }
}

pub struct RandomCropPreproc {
  pub crop_width:   usize,
  pub crop_height:  usize,
}

impl AugmentPreproc for RandomCropPreproc {
  fn transform(&self, datum: SampleDatum, label: Option<SampleLabel>, rng: &mut Xorshiftplus128Rng) -> (SampleDatum, Option<SampleLabel>) {
    match datum {
      SampleDatum::WHCBytes(old_bytes) => {
      }
    }
    // FIXME(20160425)
    unimplemented!();
  }
}
