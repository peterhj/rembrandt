use operator::{CompleteOperator};

pub mod dev_allreduce;
//pub mod mpi_allreduce;
//pub mod numa_dev_allreduce;

pub trait ParallelOptWorker {
  fn worker_rank(&self) -> usize;
  fn num_workers(&self) -> usize;

  fn operator(&mut self) -> &mut CompleteOperator;
  fn shared_seed(&mut self) -> [u64; 2];
  fn reduce_scalar_f32(&self, count: f32) -> f32;

  fn signal_barrier(&mut self);
  fn wait_barrier(&mut self) -> bool;

  fn forward(&mut self);
  fn backward(&mut self);
  fn sync_loss(&mut self, batch_size: usize) -> f32;

  fn block(&mut self);
}
