use data::{DataShard, IndexedDataShard};
use data::codec::{DataCodec};
use data_new::{SampleDatum, SampleLabel};
use util::{partition_range};

use array::{Array3d};
use byteorder::{ReadBytesExt, LittleEndian};
use rng::xorshift::{Xorshiftplus128Rng};
use varraydb::{VarrayDb};

use std::cmp::{min};
use std::io::{Read, Cursor};
use std::path::{Path};

pub struct VarrayDbShard<Codec> {
  start_idx:    usize,
  end_idx:      usize,
  codec:    Codec,
  data:     VarrayDb,
  labels:   VarrayDb,
}

impl<Codec> VarrayDbShard<Codec> where Codec: DataCodec {
  pub fn open_range(data_path: &Path, labels_path: &Path, codec: Codec, start_idx: usize, end_idx: usize) -> Self {
    let mut data = VarrayDb::open(data_path).unwrap();
    let mut labels = VarrayDb::open(labels_path).unwrap();
    assert_eq!(data.len(), labels.len());
    let length = data.len();
    //data.prefetch_range(start_idx, end_idx);
    //labels.prefetch_range(start_idx, end_idx);
    VarrayDbShard{
      start_idx:    start_idx,
      end_idx:      end_idx,
      codec:    codec,
      data:     data,
      labels:   labels,
    }
  }

  pub fn open_partition(data_path: &Path, labels_path: &Path, codec: Codec, part: usize, num_parts: usize) -> Self {
    let mut data = VarrayDb::open(data_path).unwrap();
    let mut labels = VarrayDb::open(labels_path).unwrap();
    assert_eq!(data.len(), labels.len());
    /*let length = data.len();
    let part_len = (length + num_parts - 1) / num_parts;
    let start_idx = part * part_len;
    let end_idx = min(length, (part+1) * part_len);*/
    let part_bounds = partition_range(data.len(), num_parts);
    let (start_idx, end_idx) = part_bounds[part];
    //data.prefetch_range(start_idx, end_idx);
    //labels.prefetch_range(start_idx, end_idx);
    VarrayDbShard{
      start_idx:    start_idx,
      end_idx:      end_idx,
      codec:    codec,
      data:     data,
      labels:   labels,
    }
  }
}

impl<Codec> DataShard for VarrayDbShard<Codec> where Codec: DataCodec {
  fn num_shard_samples(&self) -> usize {
    self.end_idx - self.start_idx
  }

  fn num_total_samples(&self) -> usize {
    self.data.len()
  }
}

impl<Codec> IndexedDataShard for VarrayDbShard<Codec> where Codec: DataCodec {
  fn get_sample(&mut self, offset_idx: usize) -> (SampleDatum, Option<SampleLabel>) {
    let idx = self.start_idx + offset_idx;
    assert!(idx >= self.start_idx);
    assert!(idx < self.end_idx);

    let datum_value = self.data.get(idx);
    let datum = self.codec.decode(datum_value);
    //let datum = SampleDatum::WHCBytes(datum_arr);

    let label_value = self.labels.get(idx);
    let label_cat = Cursor::new(label_value).read_i32::<LittleEndian>().unwrap();
    let label = SampleLabel::Category{category: label_cat};

    (datum, Some(label))
  }
}
