extern crate array_new;

use array_new::{NdArraySerialize, Array3d};

use std::fs::{File};

fn main() {
  let mut arr_file = File::open("imagenet_mean_256x256x3.ndarray").unwrap();
  let arr = Array3d::<f32>::deserialize(&mut arr_file).unwrap();
  let arr = arr.as_slice();
  println!("output: {} {} {}", arr[0], arr[1], arr[2]);
}
