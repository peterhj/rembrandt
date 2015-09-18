//#![feature(box_syntax)]

extern crate array;
extern crate async;
extern crate async_cuda;
extern crate byteorder;
extern crate cuda;
extern crate cuda_blas;
extern crate linalg;
extern crate lmdb;
extern crate protobuf;
extern crate rand;
extern crate rembrandt_kernels;
extern crate time;
extern crate toml;

#[macro_use]
pub mod util;

pub mod caffe_proto;
pub mod data;
pub mod layer;
