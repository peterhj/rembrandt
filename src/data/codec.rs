use array_new::{ArrayZeroExt, Array3d};
use image::{GenericImage, DynamicImage, ImageFormat, load};

use std::io::{Read, Cursor};

pub trait DataCodec {
  type Output;

  fn decode(&self, value: &[u8]) -> Self::Output;
}

pub struct PngDataCodec;

impl DataCodec for PngDataCodec {
  type Output = Array3d<u8>;

  fn decode(&self, png_value: &[u8]) -> Array3d<u8> {
    let mut reader = Cursor::new(png_value);
    let image = match load(reader, ImageFormat::PNG) {
      Err(e) => panic!("png codec: failed to load png: {:?}", e),
      Ok(im) => im,
    };
    let image_dims = image.dimensions();
    //let (image_width, image_height) = image_dims;
    let image_width = image_dims.0 as usize;
    let image_height = image_dims.1 as usize;
    let image_buf = image.to_rgb().to_vec();
    assert_eq!(image_buf.len(), image_width * image_height * 3);

    let mut image_arr = Array3d::zeros((image_width, image_height, 3));
    {
      let mut image_arr = image_arr.as_mut_slice();
      for k in 0 .. 3 {
        for j in 0 .. image_height {
          for i in 0 .. image_width {
            image_arr[i + j * image_width + k * image_width * image_height] =
                image_buf[k + i * 3 + j * 3 * image_width];
          }
        }
      }
    }

    image_arr
  }
}
