use array_new::{Array3d};

pub trait DataCodec {
  type Output;

  fn decode(&self, value: &[u8]) -> Self::Output;
}

pub struct PngDataCodec;

impl DataCodec for PngDataCodec {
  type Output = Array3d<u8>;

  fn decode(&self, png_value: &[u8]) -> Array3d<u8> {
    unimplemented!();
  }
}
