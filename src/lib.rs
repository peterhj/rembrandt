//#![feature(box_syntax)]

extern crate array;
extern crate arraydb;
extern crate async;
extern crate async_cuda;
extern crate byteorder;
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
pub mod config;
pub mod data;
pub mod graph;
pub mod layer;
pub mod net;
pub mod opt;
