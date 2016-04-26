use data::augment::{AugmentPreproc};
use data_new::{SampleDatum, SampleLabel};

use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::path::{Path};

pub mod augment;
pub mod varraydb_data;

pub trait DataShard: Iterator<Item=(SampleDatum, Option<SampleLabel>)> {
  fn num_samples(&self) -> usize;
  //fn epoch_each(&mut self, func: &mut FnMut(SampleDatum, Option<SampleLabel>));
  fn reset(&mut self) {}
}

pub trait SourceDataShard: DataShard {
  fn open(prefix: &Path) -> Self where Self: Sized;
  fn open_partition(prefix: &Path, part: usize, num_parts: usize) -> Self where Self: Sized;
}

pub trait IndexedDataShard: DataShard {
  fn get_sample(&mut self, idx: usize) -> (SampleDatum, Option<SampleLabel>);
}

pub trait EpisodicDataShard {
  // FIXME(20160425)
  fn num_episodes(&self) -> usize;
}

// FIXME(20160425)
pub struct AsyncQueueDataShard<Shard> {
  inner:    Shard,
}

pub struct AugmentDataShard<Preproc, Shard> {
  preproc:  Preproc,
  rng:      Xorshiftplus128Rng,
  inner:    Shard,
}

impl<Preproc, Shard> AugmentDataShard<Preproc, Shard> where Preproc: AugmentPreproc, Shard: DataShard {
  pub fn new(preproc: Preproc, inner: Shard) -> AugmentDataShard<Preproc, Shard> {
    AugmentDataShard{
      preproc:  preproc,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      inner:    inner,
    }
  }
}

impl<Preproc, Shard> DataShard for AugmentDataShard<Preproc, Shard> where Preproc: AugmentPreproc, Shard: DataShard {
  fn num_samples(&self) -> usize {
    self.inner.num_samples()
  }

  fn reset(&mut self) {
    self.inner.reset();
  }
}

impl<Preproc, Shard> Iterator for AugmentDataShard<Preproc, Shard> where Preproc: AugmentPreproc, Shard: DataShard {
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

pub struct RandomSampleDataShard<Shard> {
  range:    Range<usize>,
  rng:      Xorshiftplus128Rng,
  //counter:  usize,
  inner:    Shard,
}

impl<Shard> RandomSampleDataShard<Shard> where Shard: IndexedDataShard {
  pub fn new(inner: Shard) -> RandomSampleDataShard<Shard> {
    RandomSampleDataShard{
      range:    Range::new(0, inner.num_samples()),
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      //counter:  0,
      inner:    inner,
    }
  }
}

impl<Shard> DataShard for RandomSampleDataShard<Shard> where Shard: IndexedDataShard {
  fn num_samples(&self) -> usize {
    self.inner.num_samples()
  }

  fn reset(&mut self) {
    //self.counter = 0;
    self.inner.reset();
  }
}

impl<Shard> Iterator for RandomSampleDataShard<Shard> where Shard: IndexedDataShard {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    /*if self.counter >= self.num_samples() {
      return None;
    }*/
    //self.counter += 1;
    let idx = self.range.ind_sample(&mut self.rng);
    let (datum, label) = self.inner.get_sample(idx);
    Some((datum, label))
  }
}
