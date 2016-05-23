use mpi::{Mpi};
use operator::comm::{CommWorker};

//pub mod allreduce_dev;
pub mod allreduce_dist;
pub mod elasticserver_dist;
//pub mod elasticserver_dist_p2p;
pub mod elasticserver_dist_rdma;
pub mod gossip_dist;
//pub mod gossip_dist_dir;
//pub mod gossip_dist_p2p;
//pub mod gossip_dist_rdma;
//pub mod gossip_pull_allreduce_dist_rdma;
//pub mod gossip_pull_allreduce_v2_dist_rdma;
pub mod gossip_pull_dist_rdma;
//pub mod paramserver_dist;

pub trait MpiDistCommWorker: CommWorker {
  fn mpi(&self) -> &Mpi;
}
