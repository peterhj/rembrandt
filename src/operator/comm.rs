use array_cuda::device::array::{DeviceArray2d};
use array_cuda::device::comm::{ReduceOperation, AverageReduceOperation, for_all_devices};
use array_cuda::device::context::{DeviceContext, DeviceCtxRef};
use array_cuda::device::memory::{RawDeviceBuffer};
use array_new::{AsyncArray};
use rng::xorshift::{Xorshiftplus128Rng};
use worker::{WorkerData};

use rand::{Rng, SeedableRng, thread_rng};
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering};

pub trait CommWorkerBuilder: Send + Clone {
  type Worker: CommWorker;

  fn into_worker(self, tid: usize) -> Self::Worker;
}

pub trait CommWorker {
  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef);
  fn communicate(&mut self, ctx: &DeviceCtxRef);
  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef);
}

#[derive(Clone)]
pub struct NullCommWorkerBuilder;

impl CommWorkerBuilder for NullCommWorkerBuilder {
  type Worker = NullCommWorker;

  fn into_worker(self, tid: usize) -> Self::Worker {
    NullCommWorker
  }
}

pub struct NullCommWorker;

impl CommWorker for NullCommWorker {
  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {}
  fn communicate(&mut self, ctx: &DeviceCtxRef) {}
  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {}
}

#[derive(Clone)]
pub struct DeviceAllReduceCommWorkerBuilder;

pub struct DeviceAllReduceCommWorker;

impl CommWorker for DeviceAllReduceCommWorker {
  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn communicate(&mut self, ctx: &DeviceCtxRef) {
    unimplemented!();
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    unimplemented!();
  }
}

#[derive(Clone)]
pub struct DeviceSyncGossipCommWorkerBuilder {
  num_workers:  usize,
  num_rounds:   usize,
  barrier:      Arc<Barrier>,
  shared_bufs:  Vec<Arc<RawDeviceBuffer<f32>>>,
  //shared_rns:   Arc<Vec<AtomicUsize>>,
  shared_seed:  [u64; 2],
}

impl DeviceSyncGossipCommWorkerBuilder {
  pub fn new(num_workers: usize, num_rounds: usize, buf_size: usize, /*contexts: &[DeviceContext]*/) -> DeviceSyncGossipCommWorkerBuilder {
    //let num_workers = contexts.len();
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
      barrier:      Arc::new(Barrier::new(num_workers)),
      shared_bufs:  shared_bufs,
      //shared_rns:   Arc::new(vec![]), // FIXME(20160325)
      //shared_seed:  [seed_rng.next_u64(), seed_rng.next_u64()],
      shared_seed:  [thread_rng().next_u64(), thread_rng().next_u64()],
    }
  }
}

impl CommWorkerBuilder for DeviceSyncGossipCommWorkerBuilder {
  type Worker = DeviceSyncGossipCommWorker;

  fn into_worker(self, tid: usize) -> DeviceSyncGossipCommWorker {
    DeviceSyncGossipCommWorker{
      worker_data:  WorkerData::new(self.num_workers, tid),
      num_rounds:   self.num_rounds,
      barrier:      self.barrier,
      shared_bufs:  self.shared_bufs,
      //shared_rns:   self.shared_rns,
      tids_perm:    (0 .. self.num_workers).collect(),
      avg_reduce:   AverageReduceOperation::new(self.num_workers),
      rng:          Xorshiftplus128Rng::from_seed(self.shared_seed),
    }
  }
}

pub struct DeviceSyncGossipCommWorker {
  worker_data:  WorkerData,
  num_rounds:   usize,
  barrier:      Arc<Barrier>,
  shared_bufs:  Vec<Arc<RawDeviceBuffer<f32>>>,
  //shared_rns:   Arc<Vec<AtomicUsize>>,
  tids_perm:    Vec<usize>,
  avg_reduce:   AverageReduceOperation<f32>,
  // FIXME(20160324): for larger populations, should use a larger RNG.
  rng:          Xorshiftplus128Rng,
}

impl CommWorker for DeviceSyncGossipCommWorker {
  fn load(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    let src_tid = self.worker_data.tid();
    let data = data.as_view(ctx).data;
    data.raw_send(
        &(*self.shared_bufs[src_tid]).as_ref_range(offset, offset + data.len()),
    );
  }

  fn communicate(&mut self, ctx: &DeviceCtxRef) {
    let num_workers = self.worker_data.num_workers();
    if num_workers <= 1 {
      return;
    }

    /*let src_rn = self.shared_rns[src_tid].load(Ordering::Acquire);
    // FIXME(20160325): exponential backoff.
    loop {
      let dst_rn = self.shared_rns[dst_tid].load(Ordering::Acquire);
      if dst_rn >= src_rn {
        break;
      }
    }*/

    for _ in 0 .. self.num_rounds {
      self.rng.shuffle(&mut self.tids_perm);

      let src_tid = self.worker_data.tid();
      let dst_tid = self.tids_perm[src_tid];

      self.shared_bufs[dst_tid].as_ref().raw_send(
          //&(*self.shared_bufs[num_workers + src_tid]).as_ref(),
          &self.shared_bufs[num_workers + src_tid],
          ctx,
      );
      self.avg_reduce.reduce(
          &(*self.shared_bufs[src_tid]).as_ref(),
          &(*self.shared_bufs[num_workers + src_tid]).as_ref(),
          ctx,
      );
      ctx.sync();

      // FIXME(20160329): should generally replace barriers w/ round numbers,
      // but this is less important for device-only communication.
      self.barrier.wait();
    }
  }

  fn store(&mut self, offset: usize, data: &mut DeviceArray2d<f32>, ctx: &DeviceCtxRef) {
    let num_workers = self.worker_data.num_workers();
    let src_tid = self.worker_data.tid();
    let mut data = data.as_view_mut(ctx).data;
    let data_len = data.len();
    data.raw_recv(
        &(*self.shared_bufs[num_workers + src_tid]).as_ref_range(offset, offset + data_len),
    );
  }
}
