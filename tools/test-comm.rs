extern crate array_cuda;
extern crate comm_cuda;

extern crate crossbeam;
extern crate rand;

use array_cuda::device::{DeviceContext, DeviceBufferInitExt, DeviceBuffer};
use comm_cuda::{RingDeviceBufCommBuilder};
use crossbeam::{scope};

use std::rc::{Rc};

fn main() {
  let num_local_workers = 2;
  scope(|scope| {
    let builder = RingDeviceBufCommBuilder::new(num_local_workers);
    builder.set_buf_len(12 * 1024 * 1024);
    let mut guards = vec![];
    for tid in 0 .. num_local_workers {
      let builder = builder.clone();
      let guard = scope.spawn(move || {
        let context = Rc::new(DeviceContext::new(tid));
        let comm = builder.into_comm(tid, context.clone());
        for _ in 0 .. 1000 {
          comm.allreduce_average();
        }
        //let ctx = &(*context).as_ref();
        //ctx.blocking_sync();

        /*let context2 = Rc::new(DeviceContext::new(tid));
        let ctx = &(*context).as_ref();
        let ctx2 = &(*context2).as_ref();
        let mut buf = DeviceBuffer::<f32>::zeros(1000, ctx);
        let mut buf2 = DeviceBuffer::<f32>::zeros(1000, ctx2);
        buf.as_ref_mut(ctx).copy(&buf2.as_ref(ctx2));*/
      });
      guards.push(guard);
    }
    for guard in guards {
      guard.join();
    }
  });
}
