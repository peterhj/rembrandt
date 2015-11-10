#![allow(unused_imports)]

extern crate array;
extern crate async;
extern crate async_cuda;
extern crate cuda_sparse;
extern crate linalg;
extern crate rand;
extern crate time;

use array::{View, MutView, WithZeros, ArrayDeserialize, Array2d};
use async::{AsyncContext, AsyncLoad, AsyncStore, AsyncSend};
use async_cuda::array_num::{DeviceNumExt};
use async_cuda::array_rand::{DeviceRandExt};
use async_cuda::array_types::{DeviceArray2d, DeviceBuf};
use async_cuda::context::{DeviceContext};
use cuda_sparse::{CusparseMatrixDesc};
use linalg::{Transpose};
use linalg::blas::{BVector, BMatrix};
use linalg::sparse::*;
use rand::{Rng, thread_rng};
use time::{Duration, get_time};

fn main() {
  bench2();
}

fn bench2() {
  let ctx = DeviceContext::new(0);
  let mut rng = thread_rng();

  let trials = 1000;

  let m = 1;
  // XXX: first layer.
  let n = 2000;
  let k = 10 * 361;
  let x_nnz = 20;
  // XXX: second layer.
  /*let n = 361;
  let k = 2000;
  let x_nnz = 400;*/

  println!("DEBUG: allocating sparse array...");
  let mut x_val: DeviceArray2d<f32> = DeviceArray2d::with_zeros((1, x_nnz));
  let mut x_row_ptr: DeviceArray2d<i32> = DeviceArray2d::with_zeros((1, 2));
  let mut x_col_ind: DeviceArray2d<i32> = DeviceArray2d::with_zeros((1, x_nnz));
  let mut shuf_col_ind = vec![];
  for i in (0 .. k) {
    shuf_col_ind.push(i);
  }
  rng.shuffle(&mut shuf_col_ind);
  let mut x_col_ind_h = vec![];
  for j in (0 .. x_nnz) {
    x_col_ind_h.push(shuf_col_ind[j] as i32);
  }
  let x_col_ind_h = Array2d::with_data(x_col_ind_h, (1, x_nnz));
  let x_row_ptr_h = vec![0i32, (x_nnz) as i32];
  let x_row_ptr_h = Array2d::with_data(x_row_ptr_h, (1, 2));
  x_row_ptr.as_mut_view().sync_load(&x_row_ptr_h.as_view(), &ctx);
  x_col_ind.as_mut_view().sync_load(&x_col_ind_h.as_view(), &ctx);

  println!("DEBUG: allocating other arrays...");
  /*let w = DeviceArray2d::with_zeros((k, n));
  let mut y = DeviceArray2d::with_zeros((m, n));*/
  let w = DeviceArray2d::with_zeros((n, k));
  let mut y = DeviceArray2d::with_zeros((n, m));

  let mut work: DeviceArray2d<i32> = DeviceArray2d::with_zeros((4 * 1024, 1));

  let desc = CusparseMatrixDesc::create().unwrap();

  println!("DEBUG: benchmarking scsrmm...");
  ctx.synchronize();
  let start_time = get_time();
  for _ in (0 .. trials) {
    /*y.as_mut_view().matrix_csr_vector_prod(
        1.0,
        &desc,
        &x_val.as_view(), &x_row_ptr.as_view(), &x_col_ind.as_view(), x_nnz,
        &w.as_view(),
        0.0,
        &ctx);*/
    y.as_mut_view().matrix_sparse_vector_prod(
        1.0,
        &w.as_view(), Transpose::N,
        &x_val.as_view(), &x_col_ind.as_view(), x_nnz,
        0.0,
        &mut work.as_mut_view(),
        &ctx);
  }
  ctx.synchronize();
  let elapsed_ms = (get_time() - start_time).num_milliseconds();

  println!("stats per trial:");
  //println!("inputsize:  {} byte", 4 * m * n);
  //println!("outputsize: {} byte", 4 * m * k);
  //println!("flopcount:  {} flop", total_gemm_flops / trials);
  println!("elapsed:    {:.3} ms", (elapsed_ms as f32) / (trials as f32));
  //println!("gflops:     {:.3} Gflop/s", (total_gemm_flops as f32) / (elapsed_ms as f32 * 0.001) * 1.0e-9);

  // 1st layer

  let m = 8000;
  let n = 361;
  let k = 1;

  let total_gemm_flops = trials * 2 * m * n * k;

  let mut x = DeviceArray2d::with_zeros((m, n));
  let mut w = DeviceArray2d::with_zeros((n, k));
  let mut z = DeviceArray2d::with_zeros((m, k));

  x.as_mut_view().sample_uniform(&ctx);
  w.as_mut_view().sample_uniform(&ctx);

  ctx.synchronize();
  let start_time = get_time();
  for _ in (0 .. trials) {
    z.as_mut_view().matrix_prod(1.0, &x.as_view(), Transpose::N, &w.as_view(), Transpose::N, 0.0, &ctx);
  }
  ctx.synchronize();
  let elapsed_ms = (get_time() - start_time).num_milliseconds();

  println!("stats per trial:");
  //println!("inputsize:  {} byte", 4 * m * n);
  //println!("outputsize: {} byte", 4 * m * k);
  println!("flopcount:  {} flop", total_gemm_flops / trials);
  println!("elapsed:    {:.3} ms", (elapsed_ms as f32) / (trials as f32));
  println!("gflops:     {:.3} Gflop/s", (total_gemm_flops as f32) / (elapsed_ms as f32 * 0.001) * 1.0e-9);

  // 2nd layer

  let m = 361;
  let n = 8000;
  let k = 1;

  let total_gemm_flops = trials * 2 * m * n * k;

  let mut x = DeviceArray2d::with_zeros((m, n));
  let mut w = DeviceArray2d::with_zeros((n, k));
  let mut z = DeviceArray2d::with_zeros((m, k));

  x.as_mut_view().sample_uniform(&ctx);
  w.as_mut_view().sample_uniform(&ctx);

  ctx.synchronize();
  let start_time = get_time();
  for _ in (0 .. trials) {
    z.as_mut_view().matrix_prod(1.0, &x.as_view(), Transpose::N, &w.as_view(), Transpose::N, 0.0, &ctx);
  }
  ctx.synchronize();
  let elapsed_ms = (get_time() - start_time).num_milliseconds();

  println!("stats per trial:");
  //println!("inputsize:  {} byte", 4 * m * n);
  //println!("outputsize: {} byte", 4 * m * k);
  println!("flopcount:  {} flop", total_gemm_flops / trials);
  println!("elapsed:    {:.3} ms", (elapsed_ms as f32) / (trials as f32));
  println!("gflops:     {:.3} Gflop/s", (total_gemm_flops as f32) / (elapsed_ms as f32 * 0.001) * 1.0e-9);
}

fn bench1() {
  let ctx = DeviceContext::new(0);
  let trials = 1000;
  let batch = 256;
  let hidden = 3;

  let m = 19 * 19 * batch;
  let n = 5 * 5 * 4;
  let k = hidden;

  let total_gemm_flops = trials * 2 * m * n * k;

  let mut x = DeviceArray2d::with_zeros((m, n));
  let mut w = DeviceArray2d::with_zeros((n, k));
  let mut z = DeviceArray2d::with_zeros((m, k));

  x.as_mut_view().sample_uniform(&ctx);
  w.as_mut_view().sample_uniform(&ctx);

  ctx.synchronize();
  let start_time = get_time();
  for _ in (0 .. trials) {
    z.as_mut_view().matrix_prod(1.0, &x.as_view(), Transpose::N, &w.as_view(), Transpose::N, 0.0, &ctx);
  }
  ctx.synchronize();
  let elapsed_ms = (get_time() - start_time).num_milliseconds();
  println!("inputsize:  {}", 4 * m * n);
  println!("outputsize: {}", 4 * m * k);
  println!("flopcount:  {}", total_gemm_flops / trials);
  println!("elapsed:    {:.3} ms", (elapsed_ms as f32) / (trials as f32));
  println!("gflops:     {:.3} Gflop/s", (total_gemm_flops as f32) / (elapsed_ms as f32 * 0.001) * 1.0e-9);

  let avg_elapsed_ms_layer1 = elapsed_ms as f32 / trials as f32;

  let m = 19 * 19 * batch;
  let n = 3 * 3 * hidden;
  let k = 1;
  /*let m = batch;
  let n = 19 * 19 * hidden;
  let k = 19 * 19;*/

  let total_gemm_flops = trials * 2 * m * n * k;

  let mut x = DeviceArray2d::with_zeros((m, n));
  let mut w = DeviceArray2d::with_zeros((n, k));
  let mut z = DeviceArray2d::with_zeros((m, k));

  x.as_mut_view().sample_uniform(&ctx);
  w.as_mut_view().sample_uniform(&ctx);

  ctx.synchronize();
  let start_time = get_time();
  for _ in (0 .. trials) {
    z.as_mut_view().matrix_prod(1.0, &x.as_view(), Transpose::N, &w.as_view(), Transpose::N, 0.0, &ctx);
  }
  ctx.synchronize();
  let elapsed_ms = (get_time() - start_time).num_milliseconds();
  println!("inputsize:  {}", 4 * m * n);
  println!("outputsize: {}", 4 * m * k);
  println!("flopcount:  {}", total_gemm_flops / trials);
  println!("elapsed:    {:.3} ms", (elapsed_ms as f32) / (trials as f32));
  println!("gflops:     {:.3} Gflop/s", (total_gemm_flops as f32) / (elapsed_ms as f32 * 0.001) * 1.0e-9);

  let avg_elapsed_ms_layer2 = elapsed_ms as f32 / trials as f32;

  println!("games:      {:.3} games/s", (batch as f32 / 360.0) / ((avg_elapsed_ms_layer1 + avg_elapsed_ms_layer2) * 0.001));
}
