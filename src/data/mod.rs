use data::augment::{TransformPreproc};
use data_new::{SampleDatum, SampleLabel};

use rng::xorshift::{Xorshiftplus128Rng};
use threadpool::{ThreadPool};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::cmp::{max};
use std::collections::{BTreeMap};
use std::marker::{PhantomData};
use std::path::{Path};
use std::sync::{Arc, Barrier};
use std::sync::mpsc::{Sender, SyncSender, Receiver, TryRecvError, channel, sync_channel};
use std::thread::{JoinHandle, spawn};

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

pub trait DataIter: DataShard + Iterator<Item=(SampleDatum, Option<SampleLabel>)> {
  fn reset(&mut self) {}
}

pub struct RangeDataShard<Shard> {
  start_idx:    usize,
  end_idx:      usize,
  shard:    Shard,
}

impl<Shard> RangeDataShard<Shard> where Shard: IndexedDataShard {
  pub fn new(start_idx: usize, end_idx: usize, shard: Shard) -> RangeDataShard<Shard> {
    RangeDataShard{
      start_idx:    start_idx,
      end_idx:      end_idx,
      shard:    shard,
    }
  }
}

impl<Shard> DataShard for RangeDataShard<Shard> where Shard: IndexedDataShard {
  fn num_shard_samples(&self) -> usize {
    self.end_idx - self.start_idx
  }

  fn num_total_samples(&self) -> usize {
    self.shard.num_total_samples()
  }
}

impl<Shard> IndexedDataShard for RangeDataShard<Shard> where Shard: IndexedDataShard {
  fn get_sample(&mut self, offset_idx: usize) -> (SampleDatum, Option<SampleLabel>) {
    let idx = self.start_idx + offset_idx;
    assert!(idx >= self.start_idx);
    assert!(idx < self.end_idx);
    self.shard.get_sample(idx)
  }
}

pub struct RangeWrapDataShard<Shard> {
  start_idx:    usize,
  end_idx:      usize,
  wr_start_idx: Option<usize>,
  wr_end_idx:   Option<usize>,
  front_len:    usize,
  length:       usize,
  shard:    Shard,
}

impl<Shard> RangeWrapDataShard<Shard> where Shard: IndexedDataShard {
  pub fn new(range: (usize, usize), wrap_range: Option<(usize, usize)>, shard: Shard) -> RangeWrapDataShard<Shard> {
    let front_len = range.1 - range.0;
    let mut length = front_len;
    if let Some(wrap_range) = wrap_range {
      assert_eq!(0, wrap_range.0);
      length += wrap_range.1 - wrap_range.0;
    }
    RangeWrapDataShard{
      start_idx:    range.0,
      end_idx:      range.1,
      wr_start_idx: wrap_range.map(|r| r.0),
      wr_end_idx:   wrap_range.map(|r| r.1),
      front_len:    front_len,
      length:       length,
      shard:    shard,
    }
  }
}

impl<Shard> DataShard for RangeWrapDataShard<Shard> where Shard: IndexedDataShard {
  fn num_shard_samples(&self) -> usize {
    self.length
  }

  fn num_total_samples(&self) -> usize {
    self.shard.num_total_samples()
  }
}

impl<Shard> IndexedDataShard for RangeWrapDataShard<Shard> where Shard: IndexedDataShard {
  fn get_sample(&mut self, offset_idx: usize) -> (SampleDatum, Option<SampleLabel>) {
    if offset_idx < self.front_len {
      let idx = self.start_idx + offset_idx;
      assert!(idx >= self.start_idx);
      assert!(idx < self.end_idx);
      self.shard.get_sample(idx)
    } else {
      let idx = offset_idx - self.front_len;
      let wrap_end_idx = self.wr_end_idx.unwrap();
      assert!(idx < wrap_end_idx);
      self.shard.get_sample(idx)
    }
  }
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

impl<Shard> DataShard for RandomSampleDataIter<Shard> where Shard: IndexedDataShard {
  fn num_shard_samples(&self) -> usize {
    self.shard.num_shard_samples()
  }

  fn num_total_samples(&self) -> usize {
    self.shard.num_total_samples()
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

pub struct CyclicSampleDataIter<Shard> {
  counter:  usize,
  shard:    Shard,
}

impl<Shard> CyclicSampleDataIter<Shard> where Shard: IndexedDataShard {
  pub fn new(shard: Shard) -> CyclicSampleDataIter<Shard> {
    CyclicSampleDataIter{
      counter:  0,
      shard:    shard,
    }
  }
}

impl<Shard> DataIter for CyclicSampleDataIter<Shard> where Shard: IndexedDataShard {
  fn reset(&mut self) {
    self.counter = 0;
  }
}

impl<Shard> DataShard for CyclicSampleDataIter<Shard> where Shard: IndexedDataShard {
  fn num_shard_samples(&self) -> usize {
    self.shard.num_shard_samples()
  }

  fn num_total_samples(&self) -> usize {
    self.shard.num_total_samples()
  }
}

impl<Shard> Iterator for CyclicSampleDataIter<Shard> where Shard: IndexedDataShard {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    let length = self.num_shard_samples();
    if self.counter >= length {
      self.counter = 0;
    }
    let idx = self.counter;
    if idx < length {
      let (datum, label) = self.shard.get_sample(idx);
      self.counter += 1;
      Some((datum, label))
    } else {
      None
    }
  }
}

pub struct TransformDataIter<Preproc, Iter> {
  preproc:  Preproc,
  rng:      Xorshiftplus128Rng,
  inner:    Iter,
}

impl<Preproc, Iter> TransformDataIter<Preproc, Iter> where Preproc: TransformPreproc, Iter: DataIter {
  pub fn new(preproc: Preproc, inner: Iter) -> TransformDataIter<Preproc, Iter> {
    TransformDataIter{
      preproc:  preproc,
      rng:      Xorshiftplus128Rng::new(&mut thread_rng()),
      inner:    inner,
    }
  }
}

impl<Preproc, Iter> DataIter for TransformDataIter<Preproc, Iter> where Preproc: TransformPreproc, Iter: DataIter {
  fn reset(&mut self) {
    self.inner.reset();
  }
}

impl<Preproc, Iter> DataShard for TransformDataIter<Preproc, Iter> where Preproc: TransformPreproc, Iter: DataIter {
  fn num_shard_samples(&self) -> usize {
    self.inner.num_shard_samples()
  }

  fn num_total_samples(&self) -> usize {
    self.inner.num_total_samples()
  }
}

impl<Preproc, Iter> Iterator for TransformDataIter<Preproc, Iter> where Preproc: TransformPreproc, Iter: DataIter {
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
  Quit,
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

impl<Iter> AsyncQueueFetchWorker<Iter> where Iter: DataIter {
  pub fn run(&mut self) {
    loop {
      match self.ctrl_rx.try_recv() {
        Err(TryRecvError::Empty) => {
          // Do nothing.
        }
        Err(TryRecvError::Disconnected) |
        Ok(QueueCtrlMsg::Quit) => {
          break;
        }
        Ok(QueueCtrlMsg::Reset) => {
          self.inner.reset();
          self.fetch_tx.send(QueueMsg::AckReset).unwrap();
        }
      }
      match self.inner.next() {
        None => {
          self.fetch_tx.send(QueueMsg::Eof).unwrap();
        }
        Some((datum, label)) => {
          self.fetch_tx.send(QueueMsg::Sample(datum, label)).unwrap();
        }
      }
    }
  }
}

pub struct AsyncQueueDataIter<Iter> {
  shard_len:    usize,
  total_len:    usize,
  ctrl_tx:      Sender<QueueCtrlMsg>,
  fetch_rx:     Receiver<QueueMsg>,
  fetch_thr:    JoinHandle<()>,
  _marker:      PhantomData<Iter>,
}

impl<Iter> AsyncQueueDataIter<Iter> where Iter: 'static + DataIter + Send {
  pub fn new(inner: Iter) -> AsyncQueueDataIter<Iter> {
    let queue_len = 256;
    let shard_len = inner.num_shard_samples();
    let total_len = inner.num_total_samples();
    let (ctrl_tx, ctrl_rx) = channel();
    let (fetch_tx, fetch_rx) = sync_channel(queue_len);
    let fetch_thr = spawn(move || {
      AsyncQueueFetchWorker{
        ctrl_rx:    ctrl_rx,
        fetch_tx:   fetch_tx,
        inner:      inner,
      }.run();
    });
    AsyncQueueDataIter{
      shard_len:    shard_len,
      total_len:    total_len,
      ctrl_tx:      ctrl_tx,
      fetch_rx:     fetch_rx,
      fetch_thr:    fetch_thr,
      _marker:      PhantomData,
    }
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

impl<Iter> DataShard for AsyncQueueDataIter<Iter> where Iter: DataIter {
  fn num_shard_samples(&self) -> usize {
    self.shard_len
  }

  fn num_total_samples(&self) -> usize {
    self.total_len
  }
}

impl<Iter> Iterator for AsyncQueueDataIter<Iter> where Iter: DataIter {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
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

enum PoolCtrlMsg {
  Reset,
  Quit,
}

enum PoolMsg {
  Sample(usize, SampleDatum, Option<SampleLabel>),
  Eof,
  //AckReset,
}

struct AsyncPoolFetchWorker<Iter> {
  //ctrl_rx:  Receiver<PoolCtrlMsg>,
  //fetch_tx: SyncSender<PoolMsg>,
  num_threads:  usize,
  counter:      usize,
  fetch_txs:    Vec<SyncSender<PoolMsg>>,
  inner:    Iter,
}

impl<Iter> AsyncPoolFetchWorker<Iter> where Iter: DataIter {
  pub fn run(&mut self) {
    loop {
      /*match self.ctrl_rx.try_recv() {
        Err(TryRecvError::Empty) => {
          // Do nothing.
        }
        Err(TryRecvError::Disconnected) |
        Ok(PoolCtrlMsg::Quit) => {
          break;
        }
        Ok(PoolCtrlMsg::Reset) => {
          self.inner.reset();
          self.fetch_tx.send(PoolMsg::AckReset).unwrap();
        }
      }*/
      match self.inner.next() {
        None => {
          //self.fetch_tx.send(PoolMsg::Eof).unwrap();
        }
        Some((datum, label)) => {
          self.fetch_txs[self.counter % self.num_threads].send(PoolMsg::Sample(self.counter, datum, label)).unwrap();
          self.counter += 1;
        }
      }
    }
  }
}

struct AsyncPoolJoinWorker<Iter> {
  fetch_rx: Receiver<PoolMsg>,
  src_tx:   Sender<(SampleDatum, Option<SampleLabel>)>,
  join_tx:  SyncSender<PoolMsg>,
  inner:    Iter,
}

impl<Iter> AsyncPoolJoinWorker<Iter> where Iter: DataIter {
  pub fn run(&mut self) {
    loop {
      match self.fetch_rx.recv() {
        Err(_) => {
          break;
        }
        Ok(PoolMsg::Sample(idx, datum, label)) => {
          self.src_tx.send((datum, label)).unwrap();
          match self.inner.next() {
            None => {
              //self.fetch_tx.send(PoolMsg::Eof).unwrap();
            }
            Some((datum, label)) => {
              self.join_tx.send(PoolMsg::Sample(idx, datum, label)).unwrap();
            }
          }
        }
        _ => unreachable!(),
      }
    }
  }
}

pub struct SourceDataIter {
  shard_len:    usize,
  total_len:    usize,
  src_rx:   Receiver<(SampleDatum, Option<SampleLabel>)>,
}

impl DataIter for SourceDataIter {
  fn reset(&mut self) {
  }
}

impl DataShard for SourceDataIter {
  fn num_shard_samples(&self) -> usize {
    self.shard_len
  }

  fn num_total_samples(&self) -> usize {
    self.total_len
  }
}

impl Iterator for SourceDataIter {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    match self.src_rx.recv() {
      Err(_) => None,
      Ok((datum, label)) => {
        Some((datum, label))
      }
    }
  }
}

pub struct AsyncPoolDataIter<JoinIter, SrcIter> {
  shard_len:    usize,
  total_len:    usize,
  counter:      usize,
  cache:        BTreeMap<usize, (SampleDatum, Option<SampleLabel>)>,
  //ctrl_tx:      Sender<PoolCtrlMsg>,
  //fetch_rx:     Receiver<PoolMsg>,
  fetch_thr:    JoinHandle<()>,
  join_pool:    ThreadPool,
  join_rx:      Receiver<PoolMsg>,
  _marker:      PhantomData<(JoinIter, SrcIter)>,
}

impl<JoinIter, SrcIter> AsyncPoolDataIter<JoinIter, SrcIter>
where JoinIter: 'static + DataIter + Send,
      SrcIter: 'static + DataIter + Send,
{
  //pub fn new<F>(num_threads: usize, inner_gen: F, src: SrcIter) -> AsyncPoolDataIter<JoinIter, SrcIter>
  //where F: Fn(SourceDataIter) -> JoinIter {
  pub fn new(num_threads: usize, inner_gen: fn (SourceDataIter) -> JoinIter, src: SrcIter) -> AsyncPoolDataIter<JoinIter, SrcIter> {
    let queue_len = 256;
    let shard_len = src.num_shard_samples();
    let total_len = src.num_total_samples();
    //let (ctrl_tx, ctrl_rx) = channel();
    let (join_tx, join_rx) = sync_channel(queue_len);
    let mut fetch_txs = vec![];
    let join_pool = ThreadPool::new(num_threads);
    for _ in 0 .. num_threads {
      let (fetch_tx, fetch_rx) = sync_channel(queue_len);
      fetch_txs.push(fetch_tx);
      let join_tx = join_tx.clone();
      let (src_tx, src_rx) = channel();
      //let inner_gen = inner_gen.clone();
      join_pool.execute(move || {
        let inner_iter = inner_gen(SourceDataIter{
          shard_len:  shard_len,
          total_len:  total_len,
          src_rx: src_rx,
        });
        AsyncPoolJoinWorker{
          fetch_rx: fetch_rx,
          src_tx:   src_tx,
          join_tx:  join_tx,
          inner:    inner_iter,
        }.run();
      });
    }
    let fetch_thr = spawn(move || {
      AsyncPoolFetchWorker{
        //ctrl_rx:    ctrl_rx,
        num_threads:    num_threads,
        counter:    0,
        fetch_txs:  fetch_txs,
        inner:      src,
      }.run();
    });
    AsyncPoolDataIter{
      shard_len:    shard_len,
      total_len:    total_len,
      counter:      0,
      cache:        BTreeMap::new(),
      //ctrl_tx:      ctrl_tx,
      //fetch_rx:     fetch_rx,
      fetch_thr:    fetch_thr,
      join_pool:    join_pool,
      join_rx:      join_rx,
      _marker:      PhantomData,
    }
  }
}

impl<JoinIter, SrcIter> DataIter for AsyncPoolDataIter<JoinIter, SrcIter> where JoinIter: DataIter, SrcIter: DataIter {
  fn reset(&mut self) {
    // FIXME(20160427)
    /*self.ctrl_tx.send(PoolCtrlMsg::Reset).unwrap();
    loop {
      match self.fetch_rx.recv() {
        Err(e) => panic!(),
        Ok(PoolMsg::AckReset) => {
          break;
        }
        Ok(_) => {}
      }
    }*/
  }
}

//impl<Iter> DataShard for AsyncPoolDataIter<Iter> where Iter: DataIter {
impl<JoinIter, SrcIter> DataShard for AsyncPoolDataIter<JoinIter, SrcIter> where JoinIter: DataIter, SrcIter: DataIter {
  fn num_shard_samples(&self) -> usize {
    self.shard_len
  }

  fn num_total_samples(&self) -> usize {
    self.total_len
  }
}

//impl<Iter> Iterator for AsyncPoolDataIter<Iter> where Iter: DataIter {
impl<JoinIter, SrcIter> Iterator for AsyncPoolDataIter<JoinIter, SrcIter> where JoinIter: DataIter, SrcIter: DataIter {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    loop {
      if self.cache.contains_key(&self.counter) {
        let idx = self.counter;
        self.counter += 1;
        return self.cache.remove(&idx);
      }
      match self.join_rx.recv() {
        Err(e) => {
          println!("WARNING: async queue data iter: got recv error: {:?}", e);
          return None;
        }
        Ok(PoolMsg::Eof) => {
          return None;
        }
        Ok(PoolMsg::Sample(idx, datum, label)) => {
          if idx == self.counter {
            self.counter += 1;
            return Some((datum, label));
          } else {
            self.cache.insert(idx, (datum, label));
          }
        }
        //_ => unreachable!(),
      }
    }
  }
}

enum ParallelJoinMsg {
  Shape{worker_shard_len: usize, worker_total_len: usize},
  Sample{datum: SampleDatum, label: Option<SampleLabel>},
  Eof,
}

struct ParallelJoinWorker<Iter> {
  join_tx:      SyncSender<ParallelJoinMsg>,
  inner_iter:   Iter,
}

impl<Iter> ParallelJoinWorker<Iter> where Iter: DataIter {
  pub fn run(&mut self) {
    self.join_tx.send(ParallelJoinMsg::Shape{
      worker_shard_len: self.inner_iter.num_shard_samples(),
      worker_total_len: self.inner_iter.num_total_samples(),
    }).unwrap();
    loop {
      match self.inner_iter.next() {
        None => {
          // FIXME(20160503): how to handle eof?
          self.join_tx.send(ParallelJoinMsg::Eof).unwrap();
          break;
        }
        Some((datum, label)) => {
          self.join_tx.send(ParallelJoinMsg::Sample{
            datum:  datum,
            label:  label,
          }).unwrap();
        }
      }
    }
  }
}

pub struct ParallelJoinDataIter<Iter> {
  shard_len:    usize,
  total_len:    usize,
  counter:      usize,
  //cache:        BTreeMap<usize, (SampleDatum, Option<SampleLabel>)>,
  //ctrl_tx:      Sender<PoolCtrlMsg>,
  //fetch_rx:     Receiver<PoolMsg>,
  join_pool:    ThreadPool,
  join_rx:      Receiver<ParallelJoinMsg>,
  _marker:      PhantomData<Iter>,
}

impl<Iter> ParallelJoinDataIter<Iter>
where Iter: 'static + DataIter + Send {
  pub fn new(num_threads: usize, inner_gen: fn () -> Iter) -> ParallelJoinDataIter<Iter> {
    let queue_len = 256;
    //let shard_len = src.num_shard_samples();
    //let total_len = src.num_total_samples();
    //let (ctrl_tx, ctrl_rx) = channel();
    let (join_tx, join_rx) = sync_channel(queue_len);
    //let mut fetch_txs = vec![];
    let join_pool = ThreadPool::new(num_threads);
    for _ in 0 .. num_threads {
      //let (fetch_tx, fetch_rx) = sync_channel(queue_len);
      //fetch_txs.push(fetch_tx);
      let join_tx = join_tx.clone();
      //let (src_tx, src_rx) = channel();
      //let inner_gen = inner_gen.clone();
      join_pool.execute(move || {
        let inner_iter = inner_gen();
        ParallelJoinWorker{
          //fetch_rx: fetch_rx,
          //src_tx:   src_tx,
          join_tx:      join_tx,
          inner_iter:   inner_iter,
        }.run();
      });
    }
    /*let fetch_thr = spawn(move || {
      ParallelJoinFetchWorker{
        //ctrl_rx:    ctrl_rx,
        num_threads:    num_threads,
        counter:    0,
        fetch_txs:  fetch_txs,
        inner:      src,
      }.run();
    });*/
    let mut shard_len = 0;
    let mut total_len = 0;
    for _ in 0 .. num_threads {
      match join_rx.recv() {
        Err(e) => panic!("ParallelJoinDataIter: failed to query worker shape: {:?}", e),
        Ok(ParallelJoinMsg::Shape{worker_shard_len, worker_total_len}) => {
          shard_len = max(shard_len, worker_shard_len);
          total_len = max(total_len, worker_total_len);
        }
        _ => unreachable!(),
      }
    }
    ParallelJoinDataIter{
      shard_len:    shard_len,
      total_len:    total_len,
      counter:      0,
      //cache:        BTreeMap::new(),
      //ctrl_tx:      ctrl_tx,
      //fetch_rx:     fetch_rx,
      //fetch_thr:    fetch_thr,
      join_pool:    join_pool,
      join_rx:      join_rx,
      _marker:      PhantomData,
    }
  }
}

impl<Iter> DataIter for ParallelJoinDataIter<Iter> where Iter: DataIter {
  fn reset(&mut self) {
    // FIXME(20160427)
    /*self.ctrl_tx.send(PoolCtrlMsg::Reset).unwrap();
    loop {
      match self.fetch_rx.recv() {
        Err(e) => panic!(),
        Ok(PoolMsg::AckReset) => {
          break;
        }
        Ok(_) => {}
      }
    }*/
  }
}

impl<Iter> DataShard for ParallelJoinDataIter<Iter> where Iter: DataIter {
  fn num_shard_samples(&self) -> usize {
    self.shard_len
  }

  fn num_total_samples(&self) -> usize {
    self.total_len
  }
}

impl<Iter> Iterator for ParallelJoinDataIter<Iter> where Iter: DataIter {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    //loop {
    /*if self.cache.contains_key(&self.counter) {
      let idx = self.counter;
      self.counter += 1;
      return self.cache.remove(&idx);
    }*/
    match self.join_rx.recv() {
      Err(e) => {
        println!("WARNING: ParallelJoinDataIter: got recv error: {:?}", e);
        return None;
      }
      Ok(ParallelJoinMsg::Eof) => {
        return None;
      }
      Ok(ParallelJoinMsg::Sample{datum, label}) => {
        /*if idx == self.counter {
          self.counter += 1;
          return Some((datum, label));
        } else {
          self.cache.insert(idx, (datum, label));
        }*/
        self.counter += 1;
        return Some((datum, label));
      }
      _ => unreachable!(),
    }
    //}
  }
}

/*enum PoolMsg {
  Sample(usize, SampleDatum, Option<SampleLabel>),
  Eof,
}

struct AsyncPoolFetchWorker<Iter> {
  //ctrl_rx:  Receiver<PoolCtrlMsg>,
  work_rx:  Receiver<PoolMsg>,
  fetch_tx: SyncSender<PoolMsg>,
  inner:    Iter,
}

impl<Iter> AsyncPoolFetchWorker<Iter> where Iter: DataIter {
  pub fn run(&mut self) {
    loop {
      /*match self.ctrl_rx.try_recv() {
        Err(TryRecvError::Empty) => {
          // Do nothing.
        }
        Err(TryRecvError::Disconnected) |
        Ok(PoolCtrlMsg::Quit) => {
          break;
        }
        Ok(PoolCtrlMsg::Reset) => {
          self.inner.reset();
          self.fetch_tx.send(PoolMsg::AckReset).unwrap();
        }
      }*/
      match self.inner.next() {
        None => {
          self.fetch_tx.send(PoolMsg::Eof).unwrap();
        }
        Some((datum, label)) => {
          self.fetch_tx.send(PoolMsg::Sample(datum, label)).unwrap();
        }
      }
    }
  }
}

pub struct AsyncPoolDataIter<SrcIter, Iter> where SrcIter: DataIter, Iter: DataIter {
  shard_len:    usize,
  total_len:    usize,
  work_txs:     Vec<SyncSender<PoolMsg>>,
  fetch_rx:     Receiver<PoolMsg>,
  src:      SrcIter,
  _marker:  PhantomData<Iter>,
}

impl<SrcIter, Iter> AsyncPoolDataIter<SrcIter, Iter> where SrcIter: DataIter, Iter: DataIter {
  pub fn new<F>(num_threads: usize, inner_gen: F, src_iter: SrcIter) -> AsyncPoolDataIter<SrcIter, Iter>
  where F: FnMut() -> Iter {
    let queue_len = 256;
    let shard_len = inner.num_shard_samples();
    let total_len = inner.num_total_samples();
    //let (ctrl_tx, ctrl_rx) = channel();
    let pool = ThreadPool::new(num_threads);
    let join_bar = Arc::new(Barrier::new(num_threads + 1));
    let mut work_txs = vec![];
    let (fetch_tx, fetch_rx) = sync_channel(queue_len);
    for _ in 0 .. num_threads {
      let (work_tx, work_rx) = sync_channel(queue_len);
      work_txs.push(work_tx);
      let fetch_tx = fetch_tx.clone();
      let inner = inner_gen();
      pool.execute(move || {
      AsyncPoolFetchWorker{
        work_rx:    work_rx,
        fetch_tx:   fetch_tx,
        inner:      inner,
      }.run();
      });
    }
    AsyncPoolDataIter{
      shard_len:    shard_len,
      total_len:    total_len,
      work_txs:     work_txs,
      fetch_rx:     fetch_rx,
      src:          src,
      _marker:      PhantomData,
    }
  }
}

impl<Iter> DataIter for AsyncPoolDataIter<Iter> where Iter: DataIter {
  fn reset(&mut self) {
    // FIXME(20160427)
    /*self.ctrl_tx.send(PoolCtrlMsg::Reset).unwrap();
    loop {
      match self.fetch_rx.recv() {
        Err(e) => panic!(),
        Ok(PoolMsg::AckReset) => {
          break;
        }
        Ok(_) => {}
      }
    }*/
  }
}

impl<Iter> DataShard for AsyncPoolDataIter<Iter> where Iter: DataIter {
  fn num_shard_samples(&self) -> usize {
    self.shard_len
  }

  fn num_total_samples(&self) -> usize {
    self.total_len
  }
}

impl<Iter> Iterator for AsyncPoolDataIter<Iter> where Iter: DataIter {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    match self.src.next() {
      None => None,
      Some((datum, label)) => {
      }
    }
    /*match self.fetch_rx.recv() {
      Err(e) => {
        println!("WARNING: async queue data iter: got recv error: {:?}", e);
        None
      }
      Ok(PoolMsg::Eof) => {
        None
      }
      Ok(PoolMsg::Sample(datum, label)) => {
        Some((datum, label))
      }
      _ => unreachable!(),
    }*/
  }
}*/
