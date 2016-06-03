use mpi::{Mpi};
use operator::comm::{CommWorker};

//pub mod allreduce_dev;
//pub mod allreduce_dist;
//pub mod elasticserver_dist;
//pub mod elasticserver_dist_rdma;
//pub mod gossip_dist;
//pub mod gossip_pull_dist_rdma;

pub trait MpiDistCommWorker: CommWorker {
  fn mpi(&self) -> &Mpi;
}
