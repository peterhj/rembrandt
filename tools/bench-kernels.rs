//extern crate cuda;
extern crate async;
extern crate async_cuda;
extern crate rembrandt;
extern crate rembrandt_kernels;
extern crate time;

use async::{AsyncContext};
use async_cuda::context::{DeviceContext};
use rembrandt_kernels::*;

use time::{Duration, get_time};

fn main() {
  let ctx = DeviceContext::new(0);
  let n = 1000000;
  ctx.synchronize();
  let start_time = get_time();
  for _ in (0 .. n) {
    unsafe { rembrandt_kernel_map_noop(1024, ctx.stream.ptr) };
  }
  ctx.synchronize();
  let stop_time = get_time();
  let elapsed_s = ((stop_time - start_time).num_milliseconds() as f32 * 0.001);
  println!("overhead: {:.3} us", elapsed_s * 1.0e6 / n as f32);
}
