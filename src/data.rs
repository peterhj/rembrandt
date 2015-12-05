use caffe_proto::{Datum};

use array::{ArrayDeserialize, Array3d};
use arraydb::{ArrayDb};
use byteorder::{ReadBytesExt, BigEndian, LittleEndian};
use episodb::{EpisoDb};
use lmdb::{LmdbEnv, LmdbCursor};
use protobuf::{MessageStatic, parse_from_bytes};
use toml::{Parser};

use std::collections::{HashMap};
use std::fs::{File};
use std::io::{Read, BufReader, Cursor};
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

/*#[derive(Clone, Copy)]
pub struct SampleLabel(pub i32);*/

#[derive(Clone, Copy)]
pub enum SampleLabelConfig {
  Category,
  Category2,
  Lookahead{lookahead: usize},
  Lookahead2{lookahead: usize},
}

#[derive(Clone)]
pub enum SampleLabel {
  Category{category: i32},
  Category2{category: i32, category2: i32},
  MultiCategory{categories: Vec<i32>},
  MultiCategory2{categories1: Vec<i32>, category2: i32},
}

#[derive(Clone, Debug)]
pub enum DataMeanConfig {
  MeanPath(PathBuf),
  MeanValues(Vec<f32>),
}

#[derive(Clone, Debug)]
pub struct DataSourceConfig {
  pub data_path:                PathBuf,
  pub maybe_labels_path:        Option<PathBuf>,
  pub maybe_labels2_path:       Option<PathBuf>,
  pub maybe_mean:               Option<DataMeanConfig>,
  //pub maybe_limit_per_category: Option<usize>,
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
      let maybe_labels2_path = node.lookup("labels2_path").map(|x| PathBuf::from(unwrap!(x.as_str())));
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
      //let maybe_limit_per_category = node.lookup("limit_per_category").map(|x| unwrap!(x.as_integer()) as usize);
      let data_config = DataSourceConfig{
        data_path: data_path,
        maybe_labels_path: maybe_labels_path,
        maybe_labels2_path: maybe_labels2_path,
        maybe_mean: maybe_mean,
        //maybe_limit_per_category: maybe_limit_per_category,
      };
      datasets.insert(data_name.clone(), (source_name.to_string(), data_config));
    }

    DatasetConfiguration{datasets: datasets}
  }
}

pub trait DataSource: Send {
  fn request_sample(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> { unimplemented!(); }
  fn reset(&mut self) { unimplemented!(); }
  fn len(&self) -> usize { unimplemented!(); }

  fn num_samples(&self) -> usize { self.len() }
  fn num_episodes(&self) -> usize { unimplemented!(); }

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<SampleLabel>)) {
    unimplemented!();
  }

  fn each_episode(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &[(SampleDatum, Option<SampleLabel>)])) {
    unimplemented!();
  }

  fn get_episode_range(&mut self, ep_idx: usize) -> (usize, usize) {
    unimplemented!();
  }

  fn get_episode(&mut self, label_cfg: SampleLabelConfig, ep_idx: usize) -> Vec<(SampleDatum, Option<SampleLabel>)> {
    unimplemented!();
  }

  fn get_episode_sample(&mut self, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    unimplemented!();
  }
}

/*impl Iterator for DataSource {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<Self::Item> {
    self.request_sample()
  }
}*/

pub struct DataSourceBuilder;

impl DataSourceBuilder {
  pub fn build(source_name: &str, data_config: DataSourceConfig) -> Box<DataSource> {
    match source_name {
      "arraydb"     => Box::new(ArrayDbDataSource::open(data_config)),
      "episodb"     => Box::new(EpisoDbDataSource::open(data_config)),
      "lmdb_caffe"  => Box::new(LmdbCaffeDataSource::new(data_config)),
      "mnist"       => Box::new(MnistDataSource::new(data_config)),
      s => panic!("unknown data source kind: {}", s),
    }
  }
}

pub struct EpisoDbDataSource {
  config:     DataSourceConfig,
  frames_db:  EpisoDb,
  labels_db:  EpisoDb,
  labels2_db: Option<EpisoDb>,
  tmp_ep:         usize,
  tmp_categories: Vec<i32>,
}

impl EpisoDbDataSource {
  pub fn open(config: DataSourceConfig) -> EpisoDbDataSource {
    let frames_db = EpisoDb::open_read_only(config.data_path.clone());
    let labels_db = EpisoDb::open_read_only(config.maybe_labels_path.clone()
      .expect("episodb source requires category labels!"));
    assert_eq!(frames_db.num_frames(), labels_db.num_frames());
    assert_eq!(frames_db.num_episodes(), labels_db.num_episodes());
    EpisoDbDataSource{
      config:     config,
      frames_db:  frames_db,
      labels_db:  labels_db,
      labels2_db: None,
      tmp_ep:         -1,
      tmp_categories: vec![],
    }
  }
}

impl DataSource for EpisoDbDataSource {
  fn num_samples(&self) -> usize {
    self.frames_db.num_frames()
  }

  fn num_episodes(&self) -> usize {
    self.frames_db.num_episodes()
  }

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<SampleLabel>)) {
    let mut epoch_idx = 0;
    self.each_episode(label_cfg, &mut |ep_idx, episode| {
      for &(ref datum, ref label) in episode.iter() {
        f(epoch_idx, datum, label.clone());
        epoch_idx += 1;
      }
    });
  }

  fn each_episode(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &[(SampleDatum, Option<SampleLabel>)])) {
    for ep_idx in (0 .. self.num_episodes()) {
      let episode = self.get_episode(label_cfg, ep_idx);
      f(ep_idx, &episode);
    }
  }

  fn get_episode_range(&mut self, ep_idx: usize) -> (usize, usize) {
    let (start_idx, end_idx) = self.frames_db.get_episode(ep_idx).unwrap();
    (start_idx, end_idx)
  }

  fn get_episode(&mut self, label_cfg: SampleLabelConfig, ep_idx: usize) -> Vec<(SampleDatum, Option<SampleLabel>)> {
    let mut episode = vec![];
    let (start_idx, end_idx) = self.frames_db.get_episode(ep_idx).unwrap();

    /*self.tmp_categories.clear();
    for idx in (start_idx .. end_idx) {
      let category_value = self.labels_db.get_frame(idx).unwrap();
      let category = Cursor::new(category_value).read_i32::<LittleEndian>().unwrap();
      self.tmp_categories.push(category);
    }*/

    episode.clear();
    'for_each_sample:
    for (i, idx) in (start_idx .. end_idx).enumerate() {
      if let Some(sample) = self.get_episode_sample(label_cfg, ep_idx, idx) {
        episode.push(sample);
      }
    }

    episode
  }

  fn get_episode_sample(&mut self, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    let (start_idx, end_idx) = self.frames_db.get_episode(ep_idx).unwrap();
    let (start_idx2, end_idx2) = self.labels_db.get_episode(ep_idx).unwrap();
    assert_eq!(start_idx, start_idx2);
    assert_eq!(end_idx, end_idx2);
    assert!(start_idx <= sample_idx);
    assert!(sample_idx < end_idx);
    //let i = sample_idx - start_idx;

    // FIXME(20151129): Right now we skip samples with category `-1`;
    // should allow user to specify how to handle those.
    let sample_label = match label_cfg {
      SampleLabelConfig::Category => {
        let category_value = self.labels_db.get_frame(sample_idx).unwrap();
        let category = Cursor::new(category_value).read_i32::<LittleEndian>().unwrap();
        if category == -1 {
          return None;
        }
        SampleLabel::Category{category: category}
      }
      SampleLabelConfig::Category2 => {
        // TODO(20151129)
        unimplemented!();
      }
      SampleLabelConfig::Lookahead{lookahead} => {
        assert!(lookahead >= 1);
        if sample_idx + lookahead > end_idx {
          //println!("DEBUG: episodb: past lookahead: {} {} {}", start_idx, sample_idx, end_idx);
          return None;
        }
        let mut lookahead_cats = Vec::with_capacity(lookahead);
        for k in (0 .. lookahead) {
          let category_value = self.labels_db.get_frame(sample_idx + k).unwrap();
          let category = Cursor::new(category_value).read_i32::<LittleEndian>().unwrap();
          if category == -1 {
            //println!("DEBUG: episodb: invalid category, returning None");
            return None;
          }
          lookahead_cats.push(category);
        }
        SampleLabel::MultiCategory{categories: lookahead_cats}
      }
      SampleLabelConfig::Lookahead2{lookahead} => {
        // TODO(20151129)
        unimplemented!();
      }
    };
    let frame_value = self.frames_db.get_frame(sample_idx).unwrap();
    let sample_datum = SampleDatum::RgbPerChannelBytes(Array3d::<u8>::deserialize(&mut Cursor::new(frame_value))
      .ok().expect("arraydb source failed to deserialize datum!"));
    Some((sample_datum, Some(sample_label)))
  }
}

pub struct ArrayDbDataSource {
  config:     DataSourceConfig,
  frames_db:  ArrayDb,
  labels_db:  ArrayDb,
  labels2_db: Option<ArrayDb>,
  //mean_array: Array3d<f32>,
}

impl ArrayDbDataSource {
  pub fn open(config: DataSourceConfig) -> ArrayDbDataSource {
    let frames_db = ArrayDb::open(config.data_path.clone(), true);
    let labels_db = ArrayDb::open(config.maybe_labels_path.clone()
      .expect("arraydb source requires labels!"), true);
    let labels2_db = if let Some(ref labels2_path) = config.maybe_labels2_path {
      Some(ArrayDb::open(labels2_path.clone(), true))
    } else {
      None
    };
    assert_eq!(frames_db.len(), labels_db.len());
    ArrayDbDataSource{
      config:     config,
      frames_db:  frames_db,
      labels_db:  labels_db,
      labels2_db: labels2_db,
    }
  }
}

impl DataSource for ArrayDbDataSource {
  fn request_sample(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> { None }
  fn reset(&mut self) {}

  fn len(&self) -> usize {
    self.frames_db.len()
  }

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<SampleLabel>)) {
    for i in (0 .. self.frames_db.len()) {
      let datum_value = self.frames_db.get(i).unwrap();
      let datum = Array3d::<u8>::deserialize(&mut Cursor::new(datum_value))
        .ok().expect("arraydb source failed to deserialize datum!");
      let label_value = self.labels_db.get(i).unwrap();
      let label = Cursor::new(label_value).read_i32::<LittleEndian>().unwrap();
      let maybe_label2 = if let Some(ref labels2_db) = self.labels2_db {
        let label2_value = labels2_db.get(i).unwrap();
        let label2 = Cursor::new(label2_value).read_i32::<LittleEndian>().unwrap();
        Some(label2)
      } else {
        None
      };
      let sample_label = match (label, maybe_label2) {
        (label, None) => SampleLabel::Category{category: label},
        (label, Some(label2)) => SampleLabel::Category2{category: label, category2: label2},
      };
      f(i, &SampleDatum::RgbPerChannelBytes(datum), Some(sample_label));
    }
  }
}

pub struct LmdbCaffeDataIterator<'env> {
  cursor: LmdbCursor<'env>,
}

pub struct LmdbCaffeDataSource<'a> {
  loader_guard: Option<thread::JoinHandle<()>>,
  rx: Option<Receiver<Option<(i32, Array3d<u8>)>>>,
  env: LmdbEnv,
  length: usize,
  //cursor: Option<LmdbCursor<'a>>,
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
  //let limit_per_category: Option<usize> = None;

  loop {
    let mut label_sample_counts: HashMap<i32, usize> = HashMap::new();
    let mut samples_count: usize = 0;

    //for (i, kv) in cursor.iter().enumerate() {
    for kv in cursor.iter() {
      // Stop loading if we reached the total sample limit.
      /*if limit_per_category.is_some() {
        if samples_count >= limit_per_category.unwrap() * labels_count {
          break;
        }
      }*/

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
      /*if limit_per_category.is_some() {
        if label_count >= limit_per_category.unwrap() {
          continue;
        }
      }*/

      let image_flat_bytes = datum.take_data();
      assert_eq!(image_flat_bytes.len(), width * height * channels);
      let image_bytes = Array3d::with_data(image_flat_bytes, (width, height, channels));
      tx.send(Some((label, image_bytes))).unwrap();

      label_sample_counts.insert(label, label_count + 1);
      samples_count += 1;
    }
  }

  tx.send(None).unwrap();
}

impl<'a> LmdbCaffeDataSource<'a> {
  fn new(input_data: DataSourceConfig) -> LmdbCaffeDataSource<'a> {
    let mut env = LmdbEnv::open_read_only(&input_data.data_path)
      .ok().expect("failed to open lmdb env!");
    let length = env.stat().ok().expect("failed to get lmdb env stat!")
      .entries();
    //println!("DEBUG: lmdb length: {}", length);
    env.set_map_size(1099511627776)
      .ok().expect("failed to set lmdb env map size!");
    //let cursor = LmdbCursor::new_read_only(&env)
    //  .ok().expect("failed to open lmdb cursor!");
    let (tx, rx) = sync_channel::<Option<(i32, Array3d<u8>)>>(64);
    let loader_guard = thread::spawn(move || {
      lmdb_caffe_loader_process(input_data, tx);
    });
    LmdbCaffeDataSource{
      loader_guard: Some(loader_guard),
      rx: Some(rx),
      env: env,
      length: length,
      //cursor: None,
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
    /*if self.cursor.is_none() {
      let cursor = LmdbCursor::new_read_only(&self.env)
        .ok().expect("failed to open lmdb cursor!");
      self.cursor = Some(cursor);
    }*/
    match self.rx.as_ref().unwrap().recv().unwrap() {
      Some((image_label, image_bytes)) => {
        Some((SampleDatum::RgbPerChannelBytes(image_bytes), Some(SampleLabel::Category{category: image_label})))
      }
      None => None,
    }
  }

  fn len(&self) -> usize {
    self.length
  }

  fn reset(&mut self) {
    // TODO(20150917)
  }

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<SampleLabel>)) {
    let cursor = LmdbCursor::new_read_only(&self.env)
      .ok().expect("failed to open lmdb cursor!");
    for (epoch_idx, kv) in cursor.iter().enumerate() {
      // Parse a Caffe-style value. The image data (should) be stored as raw bytes.
      let value_bytes: &[u8] = kv.value;
      let mut datum: Datum = match parse_from_bytes(value_bytes) {
        Ok(m) => m,
        Err(e) => panic!("failed to parse Datum: {}", e),
      };
      let channels = datum.get_channels() as usize;
      let height = datum.get_height() as usize;
      let width = datum.get_width() as usize;
      let category = datum.get_label();

      let image_flat_bytes = datum.take_data();
      assert_eq!(image_flat_bytes.len(), width * height * channels);
      let image_bytes = Array3d::with_data(image_flat_bytes, (width, height, channels));
      let (datum, label) = (SampleDatum::RgbPerChannelBytes(image_bytes), Some(SampleLabel::Category{category: category}));
      f(epoch_idx, &datum, label);
    }
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
    Some((SampleDatum::RgbPerChannelBytes(image), Some(SampleLabel::Category{category: label})))
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

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<SampleLabel>)) {
    self.reset();
    for i in (0 .. self.n) {
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
      f(i, &SampleDatum::RgbPerChannelBytes(image), Some(SampleLabel::Category{category: label}));
    }
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
    Some((SampleDatum::RgbPerChannelBytes(image), Some(SampleLabel::Category{category: label})))
  }

  fn reset(&mut self) {
    // TODO(20150917)
  }

  fn len(&self) -> usize {
    self.n
  }
}
