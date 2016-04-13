extern crate openmpi;

use openmpi::{Mpi};

fn main() {
  let mpi = Mpi::new();
  println!("DEBUG: rank: {} size: {}", mpi.rank(), mpi.size());
}
