use data_new::{SampleDatum};

use array::{ArrayZeroExt, NdArraySerialize, Array3d};
//use image::{GenericImage, DynamicImage, ImageFormat, load};
use stb_image::image::{Image, LoadResult, load_from_memory};
use turbojpeg::{TurbojpegDecoder, TurbojpegEncoder};

use std::io::{Read, Cursor};

pub trait DataCodec {
  //type Output;

  fn decode(&mut self, value: &[u8]) -> SampleDatum;
}

pub struct Array3dDataCodec;

impl Array3dDataCodec {
  pub fn new() -> Array3dDataCodec {
    Array3dDataCodec
  }
}

impl DataCodec for Array3dDataCodec {
  fn decode(&mut self, ndarr_value: &[u8]) -> SampleDatum {
    let mut reader = Cursor::new(ndarr_value);
    let arr: Array3d<u8> = match NdArraySerialize::deserialize(&mut reader) {
      Err(_) => panic!("array3d codec: failed to deserialize"),
      Ok(arr) => arr,
    };
    SampleDatum::WHCBytes(arr)
  }
}

pub struct FakeJpegDataCodec;

impl DataCodec for FakeJpegDataCodec {
  fn decode(&mut self, _value: &[u8]) -> SampleDatum {
    SampleDatum::WHCBytes(Array3d::zeros((256, 256, 3)))
  }
}

pub struct RawJpegDataCodec;

impl DataCodec for RawJpegDataCodec {
  fn decode(&mut self, jpeg_value: &[u8]) -> SampleDatum {
    SampleDatum::JpegBuffer(jpeg_value.to_vec())
  }
}

pub struct StbJpegDataCodec;

impl DataCodec for StbJpegDataCodec {
  fn decode(&mut self, jpeg_value: &[u8]) -> SampleDatum {
    let (pixels, width, height) = match load_from_memory(jpeg_value) {
      LoadResult::ImageU8(mut im) => {
        if im.depth != 3 && im.depth != 1 {
          panic!("stb jpeg codec: unsupported depth: {}", im.depth);
        }
        assert_eq!(im.depth * im.width * im.height, im.data.len());

        if im.depth == 1 {
          let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
          assert_eq!(im.width * im.height, im.data.len());
          for i in 0 .. im.data.len() {
            rgb_data.push(im.data[i]);
            rgb_data.push(im.data[i]);
            rgb_data.push(im.data[i]);
          }
          assert_eq!(3 * im.width * im.height, rgb_data.len());
          im = Image::new(im.width, im.height, 3, rgb_data);
        }
        assert_eq!(3, im.depth);

        (im.data, im.width, im.height)
      }
      LoadResult::Error(_) |
      LoadResult::ImageF32(_) => {
        panic!("stb jpeg codec: decoder failed");
      }
    };
    let mut transp_pixels = Vec::with_capacity(pixels.len());
    for c in 0 .. 3 {
      for y in 0 .. height {
        for x in 0 .. width {
          transp_pixels.push(pixels[c + x * 3 + y * 3 * width]);
        }
      }
    }
    assert_eq!(pixels.len(), transp_pixels.len());
    let arr = Array3d::with_data(transp_pixels, (width, height, 3));
    SampleDatum::WHCBytes(arr)
  }
}

pub struct TurboJpegDataCodec {
  turbo_decoder:    TurbojpegDecoder,
}

impl TurboJpegDataCodec {
  pub fn new() -> TurboJpegDataCodec {
    let mut decoder = TurbojpegDecoder::create().unwrap();
    TurboJpegDataCodec{
      turbo_decoder:    decoder,
    }
  }
}

impl DataCodec for TurboJpegDataCodec {
  fn decode(&mut self, jpeg_value: &[u8]) -> SampleDatum {
    let (pixels, width, height) = match self.turbo_decoder.decode_rgb8(jpeg_value) {
      Ok((head, pixels)) => {
        (pixels, head.width, head.height)
      }
      Err(_) => {
        match load_from_memory(jpeg_value) {
          LoadResult::ImageU8(mut im) => {
            if im.depth != 3 && im.depth != 1 {
              panic!("jpeg codec: unsupported depth: {}", im.depth);
            }
            assert_eq!(im.depth * im.width * im.height, im.data.len());

            if im.depth == 1 {
              let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
              assert_eq!(im.width * im.height, im.data.len());
              for i in 0 .. im.data.len() {
                rgb_data.push(im.data[i]);
                rgb_data.push(im.data[i]);
                rgb_data.push(im.data[i]);
              }
              assert_eq!(3 * im.width * im.height, rgb_data.len());
              im = Image::new(im.width, im.height, 3, rgb_data);
            }
            assert_eq!(3, im.depth);

            (im.data, im.width, im.height)
          }
          LoadResult::Error(_) |
          LoadResult::ImageF32(_) => {
            panic!("jpeg codec: backup stb_image decoder failed");
          }
        }
      }
    };
    let mut transp_pixels = Vec::with_capacity(pixels.len());
    for c in 0 .. 3 {
      for y in 0 .. height {
        for x in 0 .. width {
          transp_pixels.push(pixels[c + x * 3 + y * 3 * width]);
        }
      }
    }
    assert_eq!(pixels.len(), transp_pixels.len());
    let arr = Array3d::with_data(transp_pixels, (width, height, 3));
    SampleDatum::WHCBytes(arr)
  }
}

/*pub struct PngDataCodec;

impl DataCodec for PngDataCodec {
  //type Output = Array3d<u8>;

  fn decode(&self, png_value: &[u8]) -> SampleDatum {
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

    let image_arr = Array3d::with_data(image_buf, (3, image_width, image_height));

    /*let mut image_arr = Array3d::zeros((image_width, image_height, 3));
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
    }*/

    SampleDatum::CWHBytes(image_arr)
  }
}*/
