#![feature(clone_from_slice)]
#![feature(slice_bytes)]

//extern crate array;
extern crate array_cuda;
extern crate array_dist;
extern crate array_new;
extern crate arraydb;
//extern crate async;
//extern crate async_cuda;
extern crate cuda_dnn;
extern crate episodb;
//extern crate linalg;
extern crate lmdb;
extern crate procgroup;
extern crate protobuf;
extern crate rembrandt_kernels;
extern crate rng;
extern crate toggle;
extern crate worker;

#[macro_use]
extern crate log;
extern crate byteorder;
extern crate env_logger;
extern crate memmap;
extern crate rand;
extern crate threadpool;
extern crate time;
extern crate toml;

#[macro_use]
pub mod util;
pub mod caffe_proto;
//pub mod config;
//pub mod data;
//pub mod graph;
//pub mod layer;
//pub mod net;
//pub mod opt;

pub mod arch_new;
pub mod data_new;
pub mod layer_new;
pub mod opt_new;

//pub mod arch;
//pub mod comm;
pub mod operator;
pub mod opt;
