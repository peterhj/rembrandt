use array_cuda::device::array::{DeviceArray2dView};
use array_cuda::device::comm::{ReduceOperation, AverageReduceOperation};
use array_cuda::device::context::{DeviceCtxRef};
use array_cuda::device::memory::{RawDeviceBuffer};
use rng::xorshift::{Xorshiftplus128Rng};
use worker::{WorkerData};

use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering};

pub trait CommWorker {
  fn communicate(&mut self, data: &DeviceArray2dView, ctx: &DeviceCtxRef);
}

#[derive(Clone)]
pub struct DeviceAllReduceCommWorkerBuilder;

pub struct DeviceAllReduceCommWorker;

impl CommWorker for DeviceAllReduceCommWorker {
  fn communicate(&mut self, data: &DeviceArray2dView, ctx: &DeviceCtxRef) {
    unimplemented!();
  }
}

#[derive(Clone)]
pub struct DeviceGossipCommWorkerBuilder {
  num_workers:  usize,
  barrier:      Arc<Barrier>,
  shared_bufs:  Vec<Arc<RawDeviceBuffer<f32>>>,
  //shared_rns:   Arc<Vec<AtomicUsize>>,
  shared_seed:  [u64; 2],
}

impl DeviceGossipCommWorkerBuilder {
  pub fn new<R>(seed_rng: &mut R, contexts: &[DeviceContext]) -> DeviceGossipCommWorkerBuilder
  where R: Rng {
    let num_workers = contexts.len();
    DeviceGossipCommWorkerBuilder{
      num_workers:  num_workers,
      barrier:      Arc::new(Barrier::new(num_workers)),
      shared_bufs:  vec![], // FIXME(20160324)
      //shared_rns:   Arc::new(vec![]), // FIXME(20160325)
      shared_seed:  [seed_rng.next_u64(), seed_rng.next_u64()],
    }
  }

  pub fn into_worker(self, tid: usize) -> DeviceGossipCommWorker {
    DeviceGossipCommWorker{
      worker_data:  WorkerData::new(self.num_workers, tid),
      barrier:      self.barrier,
      shared_bufs:  self.shared_bufs,
      //shared_rns:   self.shared_rns,
      tids_perm:    (0 .. self.num_workers).collect(),
      avg_reduce:   AverageReduceOperation::new(self.num_workers),
      rng:          Xorshiftplus128Rng::new(self.shared_seed),
    }
  }
}

pub struct DeviceGossipCommWorker {
  worker_data:  WorkerData,
  barrier:      Arc<Barrier>,
  shared_bufs:  Vec<Arc<RawDeviceBuffer<f32>>>,
  //shared_rns:   Arc<Vec<AtomicUsize>>,
  tids_perm:    Vec<usize>,
  avg_reduce:   AverageReduceOperation<f32>,
  // FIXME(20160324): for larger populations, should use a larger RNG.
  rng:          Xorshiftplus128Rng,
}

impl CommWorker for DeviceGossipCommWorker {
  fn communicate(&mut self, data: &mut DeviceArray2d, ctx: &DeviceCtxRef) {
    self.rng.shuffle(&mut self.tids_perm);

    let num_workers = self.worker_data.num_workers();
    let src_tid = self.worker_data.tid();
    let dst_tid = self.tids_perm[src_tid];

    /*let src_rn = self.shared_rns[src_tid].load(Ordering::Acquire);
    // FIXME(20160325): exponential backoff.
    loop {
      let dst_rn = self.shared_rns[dst_tid].load(Ordering::Acquire);
      if dst_rn >= src_rn {
        break;
      }
    }*/

    data.as_view(ctx).data.raw_send(&self.shared_bufs[src_tid].as_ref());
    self.shared_bufs[dst_tid].as_ref().raw_send(&self.shared_bufs[num_workers + src_tid].as_ref(), ctx);
    self.avg_reduce.reduce(&self.shared_bufs[src_tid].as_ref(), &self.shared_bufs[num_workers + src_tid].as_ref(), ctx);
    ctx.sync();

    // FIXME(20160329): should generally replace barriers w/ round numbers,
    // but this is less important for device-only communication.
    self.barrier.wait();

    data.as_view_mut(ctx).data.raw_recv(&self.shared_bufs[num_workers + src_tid].as_ref());
  }
}
