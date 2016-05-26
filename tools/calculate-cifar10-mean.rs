extern crate array;
extern crate rembrandt;

use array::{NdArraySerialize, Array3d};
use rembrandt::data_new::{
  DataIterator,
  SampleDatumConfig, SampleDatum,
  SampleLabelConfig,
  DatasetConfig,
  SampleIterator, RandomEpisodeIterator,
  PartitionDataSource,
};
use std::fs::{File};
use std::path::{PathBuf};

fn main() {
  let datum_cfg = SampleDatumConfig::Bytes3d;
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 10,
  };

  let dataset_cfg = DatasetConfig::open(&PathBuf::from("examples/cifar10.data"));
  let mut train_data =
      //RandomEpisodeIterator::new(
      SampleIterator::new(
          dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "train"),
      );

  let frame_len = 32 * 32 * 3;
  let mut count = 0;
  let mut running_sum: Vec<i32> = Vec::with_capacity(frame_len);
  for _ in 0 .. frame_len {
    running_sum.push(0);
  }
  train_data.each_sample(datum_cfg, label_cfg, &mut |epoch_idx, datum, maybe_label| {
    match datum {
      &SampleDatum::WHCBytes(ref frame_bytes) => {
        count += 1;
        for (i, &x) in frame_bytes.as_slice().iter().enumerate() {
          running_sum[i] += x as i32;
        }
      }
      _ => unreachable!(),
    }
  });
  let mean_buf: Vec<_> = running_sum.iter().map(|&x| x as f32 / count as f32).collect();
  //println!("DEBUG: {:?}", mean_buf);
  let mean_arr = Array3d::with_data(mean_buf, (32, 32, 3));
  let mut mean_file = File::create(&PathBuf::from("cifar10_mean_32x32x3.ndarray")).unwrap();
  mean_arr.serialize(&mut mean_file);
}
