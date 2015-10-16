#![allow(unused_imports)]

extern crate array;
extern crate async;
extern crate async_cuda;
extern crate linalg;
extern crate time;

use array::{View, MutView, WithZeros, ArrayDeserialize, Array2d};
use async::{AsyncContext, AsyncLoad, AsyncStore, AsyncSend};
use async_cuda::array_num::{DeviceNumExt};
use async_cuda::array_rand::{DeviceRandExt};
use async_cuda::array_types::{DeviceArray2d, DeviceBuf};
use async_cuda::context::{DeviceContext};
use linalg::blas::{BVector, BMatrix, Transpose};
use time::{Duration, get_time};

fn main() {
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
