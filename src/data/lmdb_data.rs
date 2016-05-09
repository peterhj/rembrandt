use caffe_proto::{Datum};
use data::{DataShard, DataIter};
use data::codec::{DataCodec};
use data_new::{SampleDatum, SampleLabel};
use util::{partition_range};

use array_new::{Array3d};
use byteorder::{ReadBytesExt, LittleEndian};
use lmdb::{LmdbEnv, LmdbRcCursor, LmdbRcCursorIterator};
use protobuf::{MessageStatic, parse_from_bytes};
use rng::xorshift::{Xorshiftplus128Rng};
use varraydb::{VarrayDb};

use std::cmp::{min};
use std::io::{Read, Write, Cursor};
use std::path::{Path};
use std::rc::{Rc};

/*pub struct LmdbCaffeShardAlt<'env> {
  total_length: usize,
  start_idx:    usize,
  end_idx:      usize,
  //iter: Option<(LmdbEnv, LmdbCursor, LmdbCursorIterator)>,
  lmdb_env:     Box<LmdbEnv>,
  lmdb_cursor:  Option<Box<LmdbCursor<'env>>>,
  //lmdb_iter:    Option<Box<LmdbCursorIterator<'env>>>,
}

impl<'env> LmdbCaffeShardAlt<'env> {
  pub fn open_partition(data_path: &Path, part: usize, num_parts: usize) -> Self {
    let mut env = match LmdbEnv::open_read_only(data_path) {
      Ok(env) => Box::new(env),
      Err(e) => panic!("failed to open lmdb env: {:?}", e),
    };
    if let Err(e) = env.set_map_size(1099511627776) {
      panic!("failed to set lmdb env map size: {:?}", e);
    }
    let total_length = match env.stat() {
      Ok(stat) => stat.entries(),
      Err(e) => panic!("failed to query lmdb env stat: {:?}", e),
    };
    let cursor = match LmdbCursor::new_read_only(&*env) {
      Ok(cursor) => Box::new(cursor),
      Err(e) => panic!("failed to open lmdb cursor: {:?}", e),
    };
    LmdbCaffeShardAlt{
      total_length: total_length,
      start_idx:    0,
      end_idx:      0,
      /*lmdb_env:     Box::new(env),
      lmdb_cursor:  Some(Box::new(cursor)),*/
      lmdb_env:     env,
      lmdb_cursor:  Some(cursor),
    }
  }
}*/

pub struct LmdbCaffeShard {
  total_length: usize,
  start_idx:    usize,
  end_idx:      usize,
  lmdb_env:     Rc<LmdbEnv>,
  lmdb_cursor:  Option<Rc<LmdbRcCursor>>,
  lmdb_iter:    Option<LmdbRcCursorIterator>,
  counter:      usize,
}

impl LmdbCaffeShard {
  pub fn open_partition(data_path: &Path, part: usize, num_parts: usize) -> Self {
    let mut env = match LmdbEnv::open_read_only(data_path) {
      Ok(env) => env,
      Err(e) => panic!("failed to open lmdb env: {:?}", e),
    };
    if let Err(e) = env.set_map_size(1099511627776) {
      panic!("failed to set lmdb env map size: {:?}", e);
    }
    let total_length = match env.stat() {
      Ok(stat) => stat.entries(),
      Err(e) => panic!("failed to query lmdb env stat: {:?}", e),
    };
    LmdbCaffeShard{
      total_length: total_length,
      start_idx:    0,
      end_idx:      0,
      lmdb_env:     Rc::new(env),
      lmdb_cursor:  None,
      lmdb_iter:    None,
      counter:      0,
    }
  }
}

impl DataShard for LmdbCaffeShard {
  fn num_shard_samples(&self) -> usize {
    self.end_idx - self.start_idx
  }

  fn num_total_samples(&self) -> usize {
    self.total_length
  }
}

impl DataIter for LmdbCaffeShard {
  fn reset(&mut self) {
    let cursor = match LmdbRcCursor::new_read_only(self.lmdb_env.clone()) {
      Ok(cursor) => Rc::new(cursor),
      Err(e) => panic!("failed to open lmdb cursor: {:?}", e),
    };
    let iter = if self.start_idx == 0 && self.end_idx == self.total_length {
      LmdbRcCursor::iter(cursor.clone())
    } else {
      let mut start_key: Vec<u8> = Vec::with_capacity(8);
      write!(&mut start_key, "{:08}", self.start_idx);
      LmdbRcCursor::seek_iter(cursor.clone(), start_key)
    };
    self.lmdb_cursor = Some(cursor);
    self.lmdb_iter = Some(iter);
    self.counter = 0;
  }
}

impl Iterator for LmdbCaffeShard {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    assert!(self.lmdb_iter.is_some());
    if self.counter >= self.end_idx - self.start_idx {
      return None;
    }
    match self.lmdb_iter.as_mut().unwrap().next() {
      None => None,
      Some(kv) => {
        let value_bytes: Vec<u8> = kv.value;
        let mut datum: Datum = match parse_from_bytes(&value_bytes) {
          Err(e) => panic!("LmdbCaffeShard: failed to parse Datum: {}", e),
          Ok(m) => m,
        };
        let channels = datum.get_channels() as usize;
        let height = datum.get_height() as usize;
        let width = datum.get_width() as usize;
        let label = datum.get_label() as i32;
        let image_flat_bytes = datum.take_data();
        assert_eq!(image_flat_bytes.len(), width * height * channels);
        let image_bytes = Array3d::with_data(image_flat_bytes, (width, height, channels));
        self.counter += 1;
        Some((SampleDatum::WHCBytes(image_bytes), Some(SampleLabel::Category{category: label})))
      }
    }
  }
}
