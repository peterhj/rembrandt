extern crate varraydb;

extern crate rand;
extern crate time;

use varraydb::{VarrayDb};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};
use time::{get_time};

fn main() {
  let path1 = PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480/ilsvrc2012_scale480_train_data.varraydb");
  let path2 = PathBuf::from("/scratch/phj/data/ilsvrc2012_scale480/ilsvrc2012_scale480_train_data.varraydb");
  let mut db1 = VarrayDb::open(&path1).unwrap();
  //let mut db2 = VarrayDb::open(&path2).unwrap();
  //let mut dbs = vec![db1, db2];

  //let n = 1281117;
  let n = 1281117 / 16;
  let num_trials = 1600 * 100;
  let shard = 3;

  let start_time = get_time();
  db1.prefetch_range(shard * n, shard * n + n);
  let end_time = get_time();
  let elapsed = (end_time - start_time).num_milliseconds() as f32 * 0.001;
  println!("DEBUG: prefetch elapsed: {:.3}", elapsed);

  let mut buf = vec![];
  let start_time = get_time();
  for i in 0 .. num_trials {
    //let idx = n/4 + i;
    //let idx = 11 * n + (i % n);
    let idx = thread_rng().gen_range(shard * n, shard * n + n);
    let value = db1.get(idx);
    //let value = db2.get(idx);
    buf.clear();
    buf.extend_from_slice(value);
    //println!("DEBUG: round: {} idx: {} value sz: {}", i, idx, buf.len());
  }
  let end_time = get_time();
  let elapsed = (end_time - start_time).num_milliseconds() as f32 * 0.001;
  println!("DEBUG: iter elapsed: {:.3} num trials: {}", elapsed, num_trials);
}
