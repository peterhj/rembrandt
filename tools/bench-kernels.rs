//extern crate cuda;
extern crate async;
extern crate async_cuda;
extern crate cuda;
extern crate rembrandt;
extern crate rembrandt_kernels;
extern crate time;

use async::{AsyncContext};
use async_cuda::context::{DeviceContext};
use cuda::runtime::{CudaEvent, CudaEventStatus};
use rembrandt_kernels::*;

use time::{Duration, get_time};

fn main() {
  let ctx = DeviceContext::new(0);
  //let n = 3_000_000; // XXX: lowest overhead event creation.
  let n = 12_000_000; // XXX: set device.
  let blocking_event = CudaEvent::create_with_flags(0x01 | 0x02).unwrap();
  let event = CudaEvent::create_with_flags(0x02).unwrap();
  ctx.synchronize();
  let start_time = get_time();
  for _ in (0 .. n) {
    //unsafe { rembrandt_kernel_map_noop(1024, ctx.stream.ptr) };

    //let event = CudaEvent::create_with_flags(0x02).unwrap();
    ctx.set_device();

    /*event.record(&ctx.stream).unwrap();
    ctx.stream.wait_event(&event).unwrap();*/

    /*match event.query() {
      Ok(CudaEventStatus::Complete) => {}
      Ok(CudaEventStatus::NotReady) => {
        event.synchronize().unwrap();
      }
      Err(e) => panic!("failed to query event: {:?}", e),
    }*/
  }
  ctx.synchronize();
  blocking_event.record(&ctx.stream).unwrap();
  ctx.stream.wait_event(&blocking_event).unwrap();
  let stop_time = get_time();
  let elapsed_s = ((stop_time - start_time).num_milliseconds() as f32 * 0.001);
  println!("elapsed:  {:.3} s", elapsed_s);
  println!("overhead: {:.3} us", elapsed_s * 1.0e6 / n as f32);
}
