use comm::{CommWorker};

use array_cuda::device::context::{DeviceCtxRef};
use worker::{WorkerData};

pub struct OperatorPipelineWorkerBuilder;

impl OperatorPipelineWorkerBuilder {
  pub fn into_worker<Comm>(self, comm_worker: Comm, ctx: &DeviceCtxRef) -> OperatorPipelineWorker<Comm> {
    unimplemented!();
  }
}

pub struct OperatorPipelineWorker<Comm> where Comm: CommWorker {
  worker_data:  WorkerData,
  comm_worker:  Comm,
}
