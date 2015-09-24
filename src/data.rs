use caffe_proto::{Datum};

use array::{Array3d};
use byteorder::{ReadBytesExt, BigEndian};
use lmdb::{LmdbEnv, LmdbCursor};
use protobuf::{MessageStatic, parse_from_bytes};
use toml::{Parser};

use std::collections::{HashMap};
use std::fs::{File};
use std::io::{Read, BufReader};
use std::marker::{PhantomData};
use std::path::{PathBuf};
//use std::str::{from_utf8};
use std::thread;
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};

pub enum SampleDatum {
  RawFeatures(Array3d<f32>),
  RgbPerChannelBytes(Array3d<u8>),
  RgbInterleavedBytes(Array3d<u8>),
}

#[derive(Clone, Copy)]
pub struct SampleLabel(pub i32);

#[derive(Clone, Debug)]
pub enum DataMeanConfig {
  MeanPath(PathBuf),
  MeanValues(Vec<f32>),
}

#[derive(Clone, Debug)]
pub struct DataSourceConfig {
  pub data_path:                PathBuf,
  pub maybe_labels_path:        Option<PathBuf>,
  pub maybe_mean:               Option<DataMeanConfig>,
  pub maybe_limit_per_category: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct DatasetConfiguration {
  pub datasets: HashMap<String, (String, DataSourceConfig)>,
}

impl DatasetConfiguration {
  pub fn open(path: &PathBuf) -> DatasetConfiguration {
    let mut file = File::open(path)
      .ok().expect("failed to open file!");
    let mut s = String::new();
    file.read_to_string(&mut s)
      .ok().expect("failed to read config file as string!");
    let mut parser = Parser::new(&s);
    let root = parser.parse()
      .expect("failed to parse config file as toml!");

    let data_node = &root[&"dataset".to_string()];

    // Parse [data] components.
    let mut datasets: HashMap<String, (String, DataSourceConfig)> = HashMap::new();
    let data_table = data_node.as_table().expect("[data] should be a table!");
    for (data_name, node) in data_table.iter() {
      let source_name = unwrap!(node.lookup("source").and_then(|x| x.as_str()));
      let data_path = PathBuf::from(unwrap!(node.lookup("data_path").and_then(|x| x.as_str())));
      let maybe_labels_path = node.lookup("labels_path").map(|x| PathBuf::from(unwrap!(x.as_str())));
      let maybe_mean_path = node.lookup("mean_path").map(|x| PathBuf::from(unwrap!(x.as_str())));
      let maybe_mean_values: Option<Vec<f32>> = node.lookup("mean_values").map(|mean_values| {
        unwrap!(mean_values.as_slice()).iter()
          .map(|x| unwrap!(x.as_float()) as f32)
          .collect()
      });
      assert!(!(maybe_mean_path.is_some() && maybe_mean_values.is_some()));
      let maybe_mean =
          maybe_mean_path.map_or(
          maybe_mean_values.map_or(None,
              |x| Some(DataMeanConfig::MeanValues(x))),
              |x| Some(DataMeanConfig::MeanPath(x)));
      let maybe_limit_per_category = node.lookup("limit_per_category").map(|x| unwrap!(x.as_integer()) as usize);
      let data_config = DataSourceConfig{
        data_path: data_path,
        maybe_labels_path: maybe_labels_path,
        maybe_mean: maybe_mean,
        maybe_limit_per_category: maybe_limit_per_category,
      };
      datasets.insert(data_name.clone(), (source_name.to_string(), data_config));
    }

    DatasetConfiguration{datasets: datasets}
  }
}

pub trait DataSource: Send {
  fn request_sample(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)>;

  fn reset(&mut self) { unimplemented!(); }
  fn len(&self) -> usize;
}

impl Iterator for DataSource {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<Self::Item> {
    self.request_sample()
  }
}

pub struct DataSourceBuilder;

impl DataSourceBuilder {
  pub fn build(source_name: &str, data_config: DataSourceConfig) -> Box<DataSource> {
    match source_name {
      "lmdb_caffe"  => Box::new(LmdbCaffeDataSource::new(data_config)),
      "mnist"       => Box::new(MnistDataSource::new(data_config)),
      s => panic!("unknown data source kind: {}", s),
    }
  }
}

pub struct LmdbCaffeDataSource<'a> {
  loader_guard: Option<thread::JoinHandle<()>>,
  rx: Option<Receiver<Option<(i32, Array3d<u8>)>>>,
  _marker: PhantomData<&'a ()>,
}

fn lmdb_caffe_loader_process(input_data: DataSourceConfig, tx: SyncSender<Option<(i32, Array3d<u8>)>>) {
  println!("DEBUG: lmdb path: {:?}", &input_data.data_path);
  let mut env = LmdbEnv::open_read_only(&input_data.data_path)
    .ok().expect("failed to open lmdb env!");
  env.set_map_size(1099511627776)
    .ok().expect("failed to set lmdb env map size!");
  let cursor = LmdbCursor::new_read_only(&env)
    .ok().expect("failed to open lmdb cursor!");

  // FIXME(20150424): hard-coding this for now.
  let labels_count: usize = 1000;
  let limit_per_category: Option<usize> = None;

  let mut label_sample_counts: HashMap<i32, usize> = HashMap::new();
  let mut samples_count: usize = 0;

  //for (i, kv) in cursor.iter().enumerate() {
  for kv in cursor.iter() {
    // Stop loading if we reached the total sample limit.
    if limit_per_category.is_some() {
      if samples_count >= limit_per_category.unwrap() * labels_count {
        break;
      }
    }

    // Parse a Caffe-style value. The image data (should) be stored as raw bytes.
    //let key_bytes: &[u8] = kv.key;
    //let key = from_utf8(key_bytes)
    //  .ok().expect("key is not valid UTF-8!");
    let value_bytes: &[u8] = kv.value;
    let mut datum: Datum = match parse_from_bytes(value_bytes) {
      Ok(m) => m,
      Err(e) => panic!("failed to parse Datum: {}", e),
    };
    //println!("DEBUG: key: {:?}, value len: {}, label: {}, encoded: {}",
    //  key, value_bytes.len(), datum.get_label(), datum.get_encoded());
    let channels = datum.get_channels() as usize;
    let height = datum.get_height() as usize;
    let width = datum.get_width() as usize;
    let label = datum.get_label();

    // Skip this key-value pair if there is a limit specified.
    if !label_sample_counts.contains_key(&label) {
      label_sample_counts.insert(label, 0);
    }
    let label_count = *label_sample_counts.get(&label).unwrap();
    if limit_per_category.is_some() {
      if label_count >= limit_per_category.unwrap() {
        continue;
      }
    }

    let image_flat_bytes = datum.take_data();
    assert_eq!(image_flat_bytes.len(), width * height * channels);
    let image_bytes = Array3d::with_data(image_flat_bytes, (width, height, channels));
    tx.send(Some((label, image_bytes))).unwrap();

    label_sample_counts.insert(label, label_count + 1);
    samples_count += 1;
  }

  tx.send(None).unwrap();
}

impl<'a> LmdbCaffeDataSource<'a> {
  fn new(input_data: DataSourceConfig) -> LmdbCaffeDataSource<'a> {
    let (tx, rx) = sync_channel::<Option<(i32, Array3d<u8>)>>(64);
    let loader_guard = thread::spawn(move || {
      lmdb_caffe_loader_process(input_data, tx);
    });
    LmdbCaffeDataSource{
      loader_guard: Some(loader_guard),
      rx: Some(rx),
      _marker: PhantomData,
    }
  }

  fn close(&mut self) {
    assert_eq!(self.loader_guard.is_some(), self.rx.is_some());
    let loader_guard = self.loader_guard.take();
    if let Some(loader_guard) = loader_guard {
      loader_guard.join()
        .ok().expect("failed to join loader thread!");
    }
  }
}

impl<'a> Drop for LmdbCaffeDataSource<'a> {
  fn drop(&mut self) {
    self.close();
  }
}

impl<'a> DataSource for LmdbCaffeDataSource<'a> {
  fn request_sample(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    match self.rx.as_ref().unwrap().recv().unwrap() {
      Some((image_label, image_bytes)) => {
        Some((SampleDatum::RgbPerChannelBytes(image_bytes), Some(SampleLabel(image_label))))
      }
      None => None,
    }
  }

  fn len(&self) -> usize {
    // TODO
    unimplemented!();
  }
}

pub struct MnistDataSource {
  data_path:      PathBuf,
  labels_path:    Option<PathBuf>,
  data_reader:    BufReader<File>,
  labels_reader:  BufReader<File>,
  image_width:    usize,
  image_height:   usize,
  n:              usize,
  i:              usize,
}

impl MnistDataSource {
  pub fn new(input_data: DataSourceConfig) -> MnistDataSource {
    let data_file = File::open(&input_data.data_path)
      .ok().expect("failed to open mnist data file!");
    let labels_file = File::open(unwrap!(input_data.maybe_labels_path.as_ref()))
      .ok().expect("failed to open mnist labels file!");
    let mut data_reader = BufReader::new(data_file);
    let mut labels_reader = BufReader::new(labels_file);
    let (data_n, width, height) = {
      let magic0 = unwrap!(data_reader.read_u8().ok());
      let magic1 = unwrap!(data_reader.read_u8().ok());
      let magic2_dty = unwrap!(data_reader.read_u8().ok());
      let magic3_ndim = unwrap!(data_reader.read_u8().ok());
      assert_eq!(magic0, 0);
      assert_eq!(magic1, 0);
      assert_eq!(magic2_dty, 0x08);
      assert_eq!(magic3_ndim, 3);
      let n = unwrap!(data_reader.read_u32::<BigEndian>().ok()) as usize;
      let height = unwrap!(data_reader.read_u32::<BigEndian>().ok()) as usize;
      let width = unwrap!(data_reader.read_u32::<BigEndian>().ok()) as usize;
      (n, width, height)
    };
    let labels_n = {
      let magic0 = unwrap!(labels_reader.read_u8().ok());
      let magic1 = unwrap!(labels_reader.read_u8().ok());
      let magic2_dty = unwrap!(labels_reader.read_u8().ok());
      let magic3_ndim = unwrap!(labels_reader.read_u8().ok());
      assert_eq!(magic0, 0);
      assert_eq!(magic1, 0);
      assert_eq!(magic2_dty, 0x08);
      assert_eq!(magic3_ndim, 1);
      let n = unwrap!(labels_reader.read_u32::<BigEndian>().ok()) as usize;
      n
    };
    assert_eq!(data_n, labels_n);
    //println!("DEBUG: mnist dims: {} {}", width, height);
    MnistDataSource{
      data_path:      input_data.data_path.clone(),
      labels_path:    input_data.maybe_labels_path.clone(),
      data_reader:    data_reader,
      labels_reader:  labels_reader,
      image_width:    width,
      image_height:   height,
      n: data_n,
      i: 0,
    }
  }

  fn open(&self, data_path: &PathBuf, maybe_labels_path: Option<&PathBuf>) -> (BufReader<File>, BufReader<File>) {
    let data_file = File::open(data_path)
      .ok().expect("failed to open mnist data file!");
    let labels_file = File::open(unwrap!(maybe_labels_path))
      .ok().expect("failed to open mnist labels file!");
    let mut data_reader = BufReader::new(data_file);
    let mut labels_reader = BufReader::new(labels_file);
    let (data_n, width, height) = {
      let magic0 = unwrap!(data_reader.read_u8().ok());
      let magic1 = unwrap!(data_reader.read_u8().ok());
      let magic2_dty = unwrap!(data_reader.read_u8().ok());
      let magic3_ndim = unwrap!(data_reader.read_u8().ok());
      assert_eq!(magic0, 0);
      assert_eq!(magic1, 0);
      assert_eq!(magic2_dty, 0x08);
      assert_eq!(magic3_ndim, 3);
      let n = unwrap!(data_reader.read_u32::<BigEndian>().ok()) as usize;
      let height = unwrap!(data_reader.read_u32::<BigEndian>().ok()) as usize;
      let width = unwrap!(data_reader.read_u32::<BigEndian>().ok()) as usize;
      (n, width, height)
    };
    let labels_n = {
      let magic0 = unwrap!(labels_reader.read_u8().ok());
      let magic1 = unwrap!(labels_reader.read_u8().ok());
      let magic2_dty = unwrap!(labels_reader.read_u8().ok());
      let magic3_ndim = unwrap!(labels_reader.read_u8().ok());
      assert_eq!(magic0, 0);
      assert_eq!(magic1, 0);
      assert_eq!(magic2_dty, 0x08);
      assert_eq!(magic3_ndim, 1);
      let n = unwrap!(labels_reader.read_u32::<BigEndian>().ok()) as usize;
      n
    };
    assert_eq!(data_n, labels_n);
    (data_reader, labels_reader)
  }
}

impl DataSource for MnistDataSource {
  fn request_sample(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    if self.i >= self.n {
      return None;
    }
    let image_len = self.image_width * self.image_height;
    let mut image_bytes: Vec<u8> = Vec::with_capacity(image_len);
    unsafe { image_bytes.set_len(image_len) };
    let mut head: usize = 0;
    loop {
      match self.data_reader.read(&mut image_bytes[head .. image_len]) {
        Ok(amnt_read) => {
          if amnt_read == 0 {
            assert_eq!(head, image_len);
            break;
          } else {
            head += amnt_read;
          }
        }
        Err(e) => panic!("i/o error while reading mnist: {}", e),
      }
    }
    let image = Array3d::with_data(image_bytes, (self.image_width, self.image_height, 1));
    let label = unwrap!(self.labels_reader.read_u8().ok()) as i32;
    self.i += 1;
    Some((SampleDatum::RgbPerChannelBytes(image), Some(SampleLabel(label))))
  }

  fn reset(&mut self) {
    let (data_reader, labels_reader) = self.open(&self.data_path, self.labels_path.as_ref());
    self.data_reader = data_reader;
    self.labels_reader = labels_reader;
    self.i = 0;
  }

  fn len(&self) -> usize {
    self.n
  }
}

pub struct CifarDataSource {
  data_path:      PathBuf,
  data_reader:    BufReader<File>,
  n:              usize,
  i:              usize,
}

impl CifarDataSource {
  pub fn new(input_data: DataSourceConfig) -> CifarDataSource {
    let data_file = File::open(&input_data.data_path)
      .ok().expect("failed to open cifar data file!");
    let mut data_reader = BufReader::new(data_file);
    CifarDataSource{
      data_path:    input_data.data_path.clone(),
      data_reader:  data_reader,
      n:            50000, // FIXME
      i:            0,
    }
  }
}

impl DataSource for CifarDataSource {
  fn request_sample(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    if self.i >= self.n {
      return None;
    }
    let label = unwrap!(self.data_reader.read_u8().ok()) as i32;
    let image_len: usize = 32 * 32 * 3;
    let mut image_bytes: Vec<u8> = Vec::with_capacity(image_len);
    unsafe { image_bytes.set_len(image_len) };
    let mut head: usize = 0;
    loop {
      match self.data_reader.read(&mut image_bytes[head .. image_len]) {
        Ok(amnt_read) => {
          if amnt_read == 0 {
            assert_eq!(head, image_len);
            break;
          } else {
            head += amnt_read;
          }
        }
        Err(e) => panic!("i/o error while reading cifar: {}", e),
      }
    }
    let image = Array3d::with_data(image_bytes, (32, 32, 3));
    self.i += 1;
    Some((SampleDatum::RgbPerChannelBytes(image), Some(SampleLabel(label))))
  }

  fn reset(&mut self) {
    // TODO(20150917)
  }

  fn len(&self) -> usize {
    self.n
  }
}
