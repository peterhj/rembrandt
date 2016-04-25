use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::comm::{ReduceOperation, AverageReduceOperation, for_all_devices};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::memory::{RawDeviceBuffer};
use array_new::{AsyncArray};
use rng::xorshift::{Xorshiftplus128Rng};
use worker_::{WorkerData};

use rand::{Rng, SeedableRng, thread_rng};
use std::rc::{Rc};
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering};

/*pub trait CommWorkerBuilder: Send + Clone {
  type Worker: CommWorker;

  fn into_worker(self, tid: usize) -> Self::Worker;
}*/

pub trait CommWorker {
  fn next(&mut self) -> bool;
  fn signal_barrier(&mut self) { unimplemented!(); }
  fn wait_barrier(&mut self) -> bool { unimplemented!(); }
  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/);
  fn complete_load(&mut self);
  fn communicate(&mut self);
  fn communicate_exact(&mut self) { unimplemented!(); }
  fn allreduce(&mut self, _src_data: &[f32], _dst_data: &mut [f32]) { unimplemented!(); }
  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/);
  fn complete_store(&mut self);
}

/*#[derive(Clone)]
pub struct NullCommWorkerBuilder;

impl CommWorkerBuilder for NullCommWorkerBuilder {
  type Worker = NullCommWorker;

  fn into_worker(self, tid: usize) -> Self::Worker {
    NullCommWorker
  }
}*/

pub struct NullCommWorker;

impl CommWorker for NullCommWorker {
  fn next(&mut self) -> bool { false }
  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {}
  fn complete_load(&mut self) {}
  fn communicate(&mut self/*, ctx: &DeviceCtxRef*/) {}
  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {}
  fn complete_store(&mut self) {}
}

#[derive(Clone)]
pub struct DeviceAllReduceCommWorkerBuilder;

pub struct DeviceAllReduceCommWorker;

impl CommWorker for DeviceAllReduceCommWorker {
  fn next(&mut self) -> bool {
    unimplemented!();
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    unimplemented!();
  }

  fn complete_load(&mut self) {
    unimplemented!();
  }

  fn communicate(&mut self/*, ctx: &DeviceCtxRef*/) {
    unimplemented!();
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    unimplemented!();
  }

  fn complete_store(&mut self) {
    unimplemented!();
  }
}

#[derive(Clone, Copy)]
pub struct GossipConfig {
  //pub num_workers:  usize,
  pub num_rounds:   usize,
  pub buf_size:     usize,
}

impl GossipConfig {
  pub fn check(&self) {
    //assert!(self.num_workers >= 1);
    assert!(self.num_rounds >= 1);
    assert!(self.buf_size < 2 * 1024 * 1024 * 1024);
  }
}

#[derive(Clone)]
pub struct DeviceSyncGossipCommWorkerBuilder {
  num_workers:  usize,
  num_rounds:   usize,
  period:       usize,
  barrier:      Arc<Barrier>,
  shared_bufs:  Vec<Arc<RawDeviceBuffer<f32>>>,
  //shared_rns:   Arc<Vec<AtomicUsize>>,
  shared_seed:  [u64; 2],
}

impl DeviceSyncGossipCommWorkerBuilder {
  pub fn new(num_workers: usize, num_rounds: usize, buf_size: usize, /*contexts: &[DeviceContext]*/) -> DeviceSyncGossipCommWorkerBuilder {
    //let num_workers = contexts.len();
    let period = 1;
    assert!(period >= 1);
    let mut shared_bufs = Vec::with_capacity(2 * num_workers);
    for_all_devices(num_workers, |contexts| {
      for tid in 0 .. num_workers {
        let ctx = contexts[tid].as_ref();
        shared_bufs.push(Arc::new(unsafe { RawDeviceBuffer::new(buf_size, &ctx) }));
      }
      for tid in 0 .. num_workers {
        let ctx = contexts[tid].as_ref();
        shared_bufs.push(Arc::new(unsafe { RawDeviceBuffer::new(buf_size, &ctx) }));
      }
    });
    DeviceSyncGossipCommWorkerBuilder{
      num_workers:  num_workers,
      num_rounds:   num_rounds,
      period:       period,
      barrier:      Arc::new(Barrier::new(num_workers)),
      shared_bufs:  shared_bufs,
      //shared_rns:   Arc::new(vec![]), // FIXME(20160325)
      //shared_seed:  [seed_rng.next_u64(), seed_rng.next_u64()],
      shared_seed:  [thread_rng().next_u64(), thread_rng().next_u64()],
    }
  }
/*}

impl CommWorkerBuilder for DeviceSyncGossipCommWorkerBuilder {
  type Worker = DeviceSyncGossipCommWorker;*/

  pub fn into_worker(self, tid: usize, context: Rc<DeviceContext>) -> DeviceSyncGossipCommWorker {
    DeviceSyncGossipCommWorker{
      worker_data:  WorkerData::new(tid, self.num_workers),
      num_rounds:   self.num_rounds,
      period:       self.period,
      context:      context,
      barrier:      self.barrier,
      avg_reduce:   AverageReduceOperation::new(self.num_workers),
      rng:          Xorshiftplus128Rng::from_seed(self.shared_seed),
      shared_bufs:  self.shared_bufs,
      //shared_rns:   self.shared_rns,
      tids_perm:    (0 .. self.num_workers).collect(),
      iter_counter: 0,
    }
  }
}

pub struct DeviceSyncGossipCommWorker {
  worker_data:  WorkerData,
  num_rounds:   usize,
  period:       usize,
  context:      Rc<DeviceContext>,
  barrier:      Arc<Barrier>,
  avg_reduce:   AverageReduceOperation<f32>,
  // FIXME(20160324): for larger populations, should use a larger RNG.
  rng:          Xorshiftplus128Rng,
  shared_bufs:  Vec<Arc<RawDeviceBuffer<f32>>>,
  //shared_rns:   Arc<Vec<AtomicUsize>>,
  tids_perm:    Vec<usize>,
  iter_counter: usize,
}

impl CommWorker for DeviceSyncGossipCommWorker {
  fn next(&mut self) -> bool {
    self.iter_counter += 1;
    let should_comm = if self.iter_counter == self.period {
      self.iter_counter = 0;
      true
    } else {
      false
    };
    should_comm
  }

  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    let ctx = &(*self.context).as_ref();
    let self_tid = self.worker_data.tid();
    let data = data.as_view(ctx).data;
    data.raw_send(
        &(*self.shared_bufs[self_tid]).as_ref_range(offset, offset + data.len()),
    );
    ctx.sync();
    // FIXME(20160329): should generally replace barriers w/ round numbers,
    // but this is less important for device-only communication.
    self.barrier.wait();
  }

  fn complete_load(&mut self) {
    //unimplemented!();
  }

  fn communicate(&mut self/*, ctx: &DeviceCtxRef*/) {
    let num_workers = self.worker_data.num_workers();
    if num_workers <= 1 {
      return;
    }

    let ctx = &(*self.context).as_ref();

    /*let src_rn = self.shared_rns[src_tid].load(Ordering::Acquire);
    // FIXME(20160325): exponential backoff.
    loop {
      let dst_rn = self.shared_rns[dst_tid].load(Ordering::Acquire);
      if dst_rn >= src_rn {
        break;
      }
    }*/

    assert_eq!(self.num_rounds, 1);
    for _ in 0 .. self.num_rounds {
      self.rng.shuffle(&mut self.tids_perm);

      let self_tid = self.worker_data.tid();
      let other_tid = self.tids_perm[self_tid];

      // FIXME(20160331): flip the sense of the buffers.
      let src_offset = 0;
      let dst_offset = num_workers;

      self.shared_bufs[src_offset + self_tid].as_ref().raw_send(
          &self.shared_bufs[dst_offset + self_tid],
          ctx,
      );
      self.avg_reduce.reduce(
          &(*self.shared_bufs[src_offset + other_tid]).as_ref(),
          &(*self.shared_bufs[dst_offset + self_tid]).as_ref(),
          ctx,
      );
      ctx.sync();
      // FIXME(20160329): should generally replace barriers w/ round numbers,
      // but this is less important for device-only communication.
      self.barrier.wait();
    }
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>/*, ctx: &DeviceCtxRef*/) {
    let ctx = &(*self.context).as_ref();
    let num_workers = self.worker_data.num_workers();
    let self_tid = self.worker_data.tid();
    let mut data = data.as_view_mut(ctx).data;
    let data_len = data.len();
    // FIXME(20160331): flip the sense of the buffers.
    data.raw_recv(
        &(*self.shared_bufs[num_workers + self_tid]).as_ref_range(offset, offset + data_len),
    );
    ctx.sync();
    // FIXME(20160329): should generally replace barriers w/ round numbers,
    // but this is less important for device-only communication.
    self.barrier.wait();
  }

  fn complete_store(&mut self) {
    //unimplemented!();
  }
}
