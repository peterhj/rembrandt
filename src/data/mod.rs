use data::augment::{AugmentPreproc};
use data_new::{SampleDatum, SampleLabel};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::marker::{PhantomData};
use std::path::{Path};
use std::thread::{JoinHandle, spawn};
use std::thread::mpsc::{Receiver, sync_channel};

pub mod augment;
pub mod codec;
pub mod varraydb_data;

pub trait DataShard {
  fn num_shard_samples(&self) -> usize;
  fn num_total_samples(&self) -> usize;
}

/*pub trait SourceDataShard: DataShard {
  fn open(prefix: &Path) -> Self where Self: Sized;
  fn open_partition(prefix: &Path, part: usize, num_parts: usize) -> Self where Self: Sized;
}*/

pub trait IndexedDataShard: DataShard {
  fn get_sample(&mut self, idx: usize) -> (SampleDatum, Option<SampleLabel>);
}

pub trait EpisodicDataShard {
  // FIXME(20160425)
  fn num_episodes(&self) -> usize;
}

pub trait DataIter: Iterator<Item=(SampleDatum, Option<SampleLabel>)> {
  fn reset(&mut self) {}
}

pub struct RandomSampleDataIter<Shard> {
  range:    Range<usize>,
  rng:      Xorshiftplus128Rng,
  shard:    Shard,
}

impl<Shard> RandomSampleDataIter<Shard> where Shard: IndexedDataShard {
  pub fn new(shard: Shard) -> RandomSampleDataIter<Shard> {
    RandomSampleDataIter{
      range:    Range::new(0, shard.num_shard_samples()),
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      shard:    shard,
    }
  }
}

impl<Shard> DataIter for RandomSampleDataIter<Shard> where Shard: IndexedDataShard {
  fn reset(&mut self) {
    // Do nothing.
  }
}

impl<Shard> Iterator for RandomSampleDataIter<Shard> where Shard: IndexedDataShard {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    let idx = self.range.ind_sample(&mut self.rng);
    let (datum, label) = self.shard.get_sample(idx);
    Some((datum, label))
  }
}

pub struct AugmentDataIter<Preproc, Iter> {
  preproc:  Preproc,
  rng:      Xorshiftplus128Rng,
  inner:    Iter,
}

impl<Preproc, Iter> AugmentDataIter<Preproc, Iter> where Preproc: AugmentPreproc, Iter: DataIter {
  pub fn new(preproc: Preproc, inner: Iter) -> AugmentDataIter<Preproc, Iter> {
    AugmentDataIter{
      preproc:  preproc,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      inner:    inner,
    }
  }
}

impl<Preproc, Iter> DataIter for AugmentDataIter<Preproc, Iter> where Preproc: AugmentPreproc, Iter: DataIter {
  fn reset(&mut self) {
    self.inner.reset();
  }
}

impl<Preproc, Iter> Iterator for AugmentDataIter<Preproc, Iter> where Preproc: AugmentPreproc, Iter: DataIter {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    match self.inner.next() {
      None => None,
      Some((datum, label)) => {
        Some(self.preproc.transform(datum, label, &mut self.rng))
      }
    }
  }
}

enum QueueCtrlMsg {
  Reset,
}

enum QueueMsg {
  Sample(SampleDatum, Option<SampleLabel>),
  Eof,
  AckReset,
}

struct AsyncQueueFetchWorker<Iter> {
  ctrl_rx:  Receiver<QueueCtrlMsg>,
  fetch_tx: SyncSender<QueueMsg>,
  inner:    Iter,
}

pub struct AsyncQueueDataIter<Iter> {
  ctrl_tx:      SyncSender<QueueCtrlMsg>,
  fetch_rx:     Receiver<QueueMsg>,
  fetch_thr:    JoinHandle<()>,
  _marker:      PhantomData<Iter>,
}

impl<Iter> AsyncQueueDataIter<Iter> where Iter: DataIter {
  pub fn new(inner: Iter) -> AsyncQueueDataIter<Iter> {
    let (ctrl_tx, ctrl_rx) = sync_channel();
    let (fetch_tx, fetch_rx) = sync_channel();
    let fetch_thr = spawn(move || {
    });
    unimplemented!();
  }
}

impl<Iter> DataIter for AsyncQueueDataIter<Iter> where Iter: DataIter {
  fn reset(&mut self) {
    self.ctrl_tx.send(QueueCtrlMsg::Reset).unwrap();
    loop {
      match self.fetch_rx.recv() {
        Err(e) => panic!(),
        Ok(QueueMsg::AckReset) => {
          break;
        }
        Ok(_) => {}
      }
    }
  }
}

impl<Iter> Iterator for AsyncQueueDataIter<Iter> where Iter: DataIter {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>) {
    match self.fetch_rx.recv() {
      Err(e) => {
        println!("WARNING: async queue data iter: got recv error: {:?}", e);
        None
      }
      Ok(QueueMsg::Eof) => {
        None
      }
      Ok(QueueMsg::Sample(datum, label)) => {
        Some((datum, label))
      }
      _ => unreachable!(),
    }
  }
}
