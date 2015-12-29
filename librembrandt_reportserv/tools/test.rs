extern crate rembrandt_reportserv;

use rembrandt_reportserv::{ReportServer};

fn main() {
  let (thread, _) = ReportServer::spawn();
  thread.join().unwrap();
}
