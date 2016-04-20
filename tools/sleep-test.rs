extern crate time;

use std::thread::{sleep};
use std::time::{Duration};
use time::{get_time};

fn main() {
  let num_trials = 200_000;
  let duration = Duration::new(0, 50_000);
  let start_time = get_time();
  for _ in 0 .. num_trials {
    sleep(duration);
  }
  let end_time = get_time();
  let elapsed_ms = (end_time - start_time).num_milliseconds();
  println!("elapsed: {:.3} s", elapsed_ms as f32 * 0.001);
  println!("sleep duration: {:.3} ms", elapsed_ms as f32 / num_trials as f32);
}
