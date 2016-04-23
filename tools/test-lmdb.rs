extern crate rembrandt;

use rembrandt::data_new::{DataSourceConfig, LmdbCaffeDataIterator};

use std::path::{PathBuf};
use std::str::{from_utf8};

fn main() {
  let config = DataSourceConfig {
    data_path:            PathBuf::from("/scratch/phj/data/ilsvrc2012/ilsvrc2012_train_256x256_lmdb"),
    maybe_labels_path:    None,
    maybe_labels2_path:   None,
    datum_cfg:    None,
    label_cfg:    None,
  };
  let mut iter = LmdbCaffeDataIterator::open_seek(config, b"00123456".to_vec());
  let mut counter = 0;
  iter.each_kv(&mut |key, value| {
    let key_str = from_utf8(key).unwrap();
    println!("DEBUG: counter: {} key: {}", counter, key_str);
    counter += 1;
    if counter >= 10 {
      None
    } else {
      Some(())
    }
  });
}
