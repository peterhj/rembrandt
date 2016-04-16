extern crate mpich;

use mpich::{Mpi};

fn main() {
  let mpi = Mpi::new();
  println!("DEBUG: rank: {} size: {}", mpi.rank(), mpi.size());
}
