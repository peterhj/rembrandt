extern crate array;

use array::{NdArraySerialize, Array3d};

use std::fs::{File};

fn main() {
  let mut arr_file = File::open("imagenet_mean_256x256x3.ndarray").unwrap();
  let arr = Array3d::<f32>::deserialize(&mut arr_file).unwrap();
  let arr = arr.as_slice();
  println!("R? {:?}", &arr[0 .. 256]);
  println!("G? {:?}", &arr[1*256 .. 2*256]);
  println!("B? {:?}", &arr[2*256 .. 3*256]);
}
