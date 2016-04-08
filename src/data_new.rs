use caffe_proto::{Datum};

use array_new::{Shape, Array, ArrayViewMut, NdArraySerialize, Array3d, BitArray3d};
use byteorder::{LittleEndian, BigEndian, ReadBytesExt};
use episodb::{EpisoDb};
use lmdb::{LmdbEnv, LmdbCursor};
use memmap::{Mmap, Protection};
use protobuf::{MessageStatic, parse_from_bytes};
//use random::{XorShift128Plus};
use rng::xorshift::{Xorshiftplus128Rng};
use toml::{Parser};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::collections::{HashMap};
use std::fs::{File};
use std::io::{Read, BufReader, Cursor};
use std::marker::{PhantomData};
use std::path::{PathBuf};

#[derive(Clone, Copy, Debug)]
pub enum SampleDatumConfig {
  Bytes3d,
  Bits3d{scale: u8},
  BitsThenBytes3d{scale: u8},
}

impl SampleDatumConfig {
  pub fn decode(&self, value: &[u8]) -> SampleDatum {
    match *self {
      SampleDatumConfig::Bytes3d => {
        SampleDatum::WHCBytes(Array3d::deserialize(&mut Cursor::new(value))
          .ok().expect("failed to decode Bytes3d!"))
      }
      SampleDatumConfig::Bits3d{scale} => {
        let bit_arr = BitArray3d::deserialize(&mut Cursor::new(value))
          .ok().expect("failed to decode Bits3d!");
        let arr = bit_arr.into_bytes(scale);
        SampleDatum::WHCBytes(arr)
      }
      SampleDatumConfig::BitsThenBytes3d{scale} => {
        let mut reader = Cursor::new(value);
        //let bit_size = reader.read_u64::<LittleEndian>().unwrap() as usize;
        let bit_arr = BitArray3d::deserialize(&mut reader)
          .ok().expect("failed to decode BitsThenBytes3d bits half!");
        //let bytes_size = value.read_u64::<LittleEndian>().unwrap() as usize;
        //assert_eq!(bit_size + bytes_size + 16, value.len());
        let bytes_arr: Array3d<u8> = Array3d::deserialize(&mut reader)
          .ok().expect("failed to decode BitsThenBytes3d bytes half!");
        assert_eq!(bit_arr.bound().0, bytes_arr.bound().0);
        assert_eq!(bit_arr.bound().1, bytes_arr.bound().1);
        let (width, height) = (bit_arr.bound().0, bit_arr.bound().1);
        let bit_chs = bit_arr.bound().2;
        let bytes_chs = bytes_arr.bound().2;
        let channels = bit_chs + bytes_chs;

        // FIXME(20160202)
        let mut arr: Array3d<u8> = unsafe { Array3d::new((width, height, channels)) };
        {
          let mut arr = arr.as_view_mut().view_mut((0, 0, 0), (width, height, bit_chs));
          bit_arr.write_bytes(scale, &mut arr);
        }
        {
          let mut arr = arr.as_view_mut().view_mut((0, 0, bit_chs), (width, height, channels));
          arr.copy_from(&bytes_arr.as_view());
        }
        SampleDatum::WHCBytes(arr)
      }
    }
  }
}

#[derive(Clone)]
pub enum SampleDatum {
  WHCBytes(Array3d<u8>),
}

#[derive(Clone, Copy, Debug)]
pub enum SampleLabelConfig {
  Category{num_categories: i32},
  //Category2,
  LookaheadCategories{num_categories: i32, lookahead: usize},
  //Lookahead2{lookahead: usize},
}

#[derive(Clone)]
pub enum SampleLabel {
  Category{category: i32},
  //Category2{category: i32, category2: i32},
  MultiCategory{categories: Vec<i32>},
  //MultiCategory2{categories1: Vec<i32>, category2: i32},
}

pub trait DataIterator {
  /*type Iter: Iterator<Item=(SampleDatum, Option<SampleLabel>)>;

  fn iter(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig) -> Self::Iter { unimplemented!(); }*/

  fn max_num_samples(&self) -> usize;
  fn each_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, /*filter: &Fn(usize) -> bool,*/ f: &mut FnMut(usize, &SampleDatum, Option<&SampleLabel>));

  /*fn get_episode_indices(&mut self, ep_idx: usize) -> Option<(usize, usize)> {
    unimplemented!();
  }

  fn get_episode_sample(&mut self, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    unimplemented!();
  }*/
}

pub struct DummyIter;

impl Iterator for DummyIter {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    None
  }
}

pub struct SampleIterator {
  data: Box<DataSource>,
}

impl SampleIterator {
  pub fn new(data: Box<DataSource>) -> SampleIterator {
    SampleIterator{data: data}
  }
}

impl DataIterator for SampleIterator {
  // FIXME(20160129)
  //type Iter = DummyIter;

  fn max_num_samples(&self) -> usize {
    self.data.num_samples()
  }

  fn each_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, /*filter: &Fn(usize) -> bool,*/ f: &mut FnMut(usize, &SampleDatum, Option<&SampleLabel>)) {
    let mut epoch_idx = 0;
    let (ep_idx_start, ep_idx_end) = self.data.get_episodes_range();
    for ep_idx in ep_idx_start .. ep_idx_end {
      let (start_idx, end_idx) = self.data.get_episode_indices(ep_idx).unwrap();
      for sample_idx in start_idx .. end_idx {
        if let Some((datum, maybe_label)) = self.data.get_episode_sample(datum_cfg, label_cfg, ep_idx, sample_idx) {
          f(epoch_idx, &datum, maybe_label.as_ref());
          epoch_idx += 1;
        }
      }
    }
  }
}

pub struct RandomEpisodeIterator<'a> {
  data: Box<DataSource>,
  _marker:  PhantomData<&'a ()>,
}

impl<'a> RandomEpisodeIterator<'a> {
  pub fn new(data: Box<DataSource>) -> RandomEpisodeIterator<'a> {
    RandomEpisodeIterator{
      data: data,
      _marker:  PhantomData,
    }
  }
}

pub struct RandomEpisodeIter<'a> {
  ep_idx_range: Range<usize>,
  datum_cfg:    SampleDatumConfig,
  label_cfg:    SampleLabelConfig,

  epoch_idx:    usize,
  num_samples:  usize,
  rng:          Xorshiftplus128Rng,
  source:       &'a mut RandomEpisodeIterator<'a>,
}

impl<'a> Iterator for RandomEpisodeIter<'a> {
  type Item = (SampleDatum, Option<SampleLabel>);

  fn next(&mut self) -> Option<(SampleDatum, Option<SampleLabel>)> {
    while self.epoch_idx < self.num_samples {
      let ep_idx = self.ep_idx_range.ind_sample(&mut self.rng);
      let (start_idx, end_idx) = self.source.data.get_episode_indices(ep_idx).unwrap();
      let sample_idx = self.rng.gen_range(start_idx, end_idx);
      self.epoch_idx += 1;
      if let Some((datum, maybe_label)) = self.source.data.get_episode_sample(self.datum_cfg, self.label_cfg, ep_idx, sample_idx) {
        return Some((datum, maybe_label));
      }
    }
    None
  }
}

impl<'a> DataIterator for RandomEpisodeIterator<'a> {
  /*type Iter = RandomEpisodeIter<'a>;

  pub fn iter(&'a mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig) -> RandomEpisodeIter<'a> {
    let (ep_idx_start, ep_idx_end) = self.data.get_episodes_range();
    RandomEpisodeIter{
      ep_idx_range: Range::new(ep_idx_start, ep_idx_end),
      datum_cfg:    datum_cfg,
      label_cfg:    label_cfg,

      epoch_idx:    0,
      num_samples:  self.data.num_samples(),
      rng:          Xorshiftplus128Rng::new(&mut thread_rng()),
      source:       self,
    }
  }*/

  fn max_num_samples(&self) -> usize {
    self.data.num_samples()
  }

  fn each_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<&SampleLabel>)) {
    let mut epoch_idx = 0;
    let (ep_idx_start, ep_idx_end) = self.data.get_episodes_range();
    let ep_idx_range = Range::new(ep_idx_start, ep_idx_end);
    let num_samples = self.max_num_samples();
    for _ in 0 .. num_samples {
      let ep_idx = ep_idx_range.ind_sample(&mut thread_rng());
      let (start_idx, end_idx) = self.data.get_episode_indices(ep_idx).unwrap();
      let sample_idx = thread_rng().gen_range(start_idx, end_idx);
      if let Some((datum, maybe_label)) = self.data.get_episode_sample(datum_cfg, label_cfg, ep_idx, sample_idx) {
        f(epoch_idx, &datum, maybe_label.as_ref());
        epoch_idx += 1;
      }
    }
  }
}

pub struct CyclicEpisodeIterator {
  //rng:  XorShift128PlusRng,
  data: Box<DataSource>,
}

impl CyclicEpisodeIterator {
  pub fn new(data: Box<DataSource>) -> CyclicEpisodeIterator {
    CyclicEpisodeIterator{
      //rng:  XorShift128PlusRng::from_seed([thread_rng().gen(), thread_rng().gen()]),
      data: data,
    }
  }
}

impl DataIterator for CyclicEpisodeIterator {
  // FIXME(20160129)
  //type Iter = DummyIter;

  fn max_num_samples(&self) -> usize {
    self.data.num_samples()
  }

  fn each_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<&SampleLabel>)) {
    let limit = self.data.num_samples();
    let mut counter = 0;
    let mut epoch_idx = 0;
    let (ep_idx_start, ep_idx_end) = self.data.get_episodes_range();
    loop {
      for ep_idx in ep_idx_start .. ep_idx_end {
        let (start_idx, end_idx) = self.data.get_episode_indices(ep_idx).unwrap();
        let sample_idx = thread_rng().gen_range(start_idx, end_idx);
        if let Some((datum, maybe_label)) = self.data.get_episode_sample(datum_cfg, label_cfg, ep_idx, sample_idx) {
          f(epoch_idx, &datum, maybe_label.as_ref());
          epoch_idx += 1;
        }
        counter += 1;
        if counter >= limit {
          return;
        }
      }
    }
  }
}

#[derive(Clone, Debug)]
pub struct DataSourceConfig {
  pub data_path:            PathBuf,
  pub maybe_labels_path:    Option<PathBuf>,
  pub maybe_labels2_path:   Option<PathBuf>,
  //pub maybe_mean:               Option<DataMeanConfig>,
  pub datum_cfg:    Option<SampleDatumConfig>,
  pub label_cfg:    Option<SampleLabelConfig>,
}

#[derive(Clone, Debug)]
pub struct DatasetConfig {
  pub datasets: HashMap<String, (String, DataSourceConfig)>,
}

impl DatasetConfig {
  pub fn open(path: &PathBuf) -> DatasetConfig {
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
      /*let maybe_mean_path = node.lookup("mean_path").map(|x| PathBuf::from(unwrap!(x.as_str())));
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
              |x| Some(DataMeanConfig::MeanPath(x)));*/

      let data_config = DataSourceConfig{
        data_path: data_path,
        maybe_labels_path: maybe_labels_path,
        maybe_labels2_path: maybe_labels2_path,
        //maybe_mean: maybe_mean,
        datum_cfg:  None,
        label_cfg:  None,
      };
      datasets.insert(data_name.clone(), (source_name.to_string(), data_config));
    }

    DatasetConfig{datasets: datasets}
  }

  pub fn build(&self, key: &str) -> Box<DataSource> {
    match self.datasets.get(key) {
      Some(&(ref src_name, ref data_cfg)) => {
        match src_name as &str {
          "episodb" => Box::new(EpisoDbDataSource::open(data_cfg.clone())),
          "mnist"   => Box::new(MnistDataSource::open(data_cfg.clone())),
          "cifar10" => Box::new(Cifar10DataSource::open(data_cfg.clone())),
          _ => panic!("unknown data source: '{}'", src_name),
        }
      }
      None => panic!("dataset missing key: '{}'", key),
    }
  }

  pub fn build_with_cfg(&self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, key: &str) -> Box<DataSource> {
    match self.datasets.get(key) {
      Some(&(ref src_name, ref data_cfg)) => {
        let mut data_cfg = data_cfg.clone();
        data_cfg.datum_cfg = Some(datum_cfg);
        data_cfg.label_cfg = Some(label_cfg);
        match src_name as &str {
          "episodb" => Box::new(EpisoDbDataSource::open(data_cfg)),
          "mnist"   => Box::new(MnistDataSource::open(data_cfg)),
          "cifar10" => Box::new(Cifar10DataSource::open(data_cfg)),
          _ => panic!("unknown data source: '{}'", src_name),
        }
      }
      None => panic!("dataset missing key: '{}'", key),
    }
  }

  pub fn build_iterator(&self, key: &str) -> Box<DataIterator> {
    match self.datasets.get(key) {
      Some(&(ref src_name, ref data_cfg)) => {
        match src_name as &str {
          "lmdb_caffe"  => Box::new(LmdbCaffeDataIterator::open(data_cfg.clone())),
          _ => panic!("unknown data iterator: '{}'", src_name),
        }
      }
      None => panic!("dataset missing key: '{}'", key),
    }
  }

  pub fn build_source(&self, key: &str) -> Box<DataSource> {
    // FIXME(20160408)
    unimplemented!();
  }
}

pub trait DataSource {
  fn num_samples(&self) -> usize;
  fn count_prefix_samples(&self, ep_idx: usize) -> usize;
  fn count_suffix_samples(&self, ep_idx: usize) -> usize;
  fn num_episodes(&self) -> usize;

  fn get_episodes_range(&self) -> (usize, usize);
  fn get_episode_indices(&mut self, ep_idx: usize) -> Option<(usize, usize)>;
  fn get_episode_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)>;
}

pub trait Augment {
  fn transform(&mut self, datum: SampleDatum, maybe_label: Option<SampleLabel>) -> (SampleDatum, Option<SampleLabel>);
}

pub struct AugmentDataSource<A> where A: Augment {
  inner:    Box<DataSource>,
  augment:  A,
  //_marker:  PhantomData<A>,
}

impl<A> AugmentDataSource<A> where A: Augment {
  pub fn new(augment: A, inner: Box<DataSource>) -> AugmentDataSource<A> {
    AugmentDataSource{
      inner:    inner,
      augment:  augment,
      //_marker:  PhantomData,
    }
  }
}

impl<A> DataSource for AugmentDataSource<A> where A: Augment {
  fn num_samples(&self) -> usize {
    self.inner.num_samples()
  }

  fn count_prefix_samples(&self, ep_idx: usize) -> usize {
    self.inner.count_prefix_samples(ep_idx)
  }

  fn count_suffix_samples(&self, ep_idx: usize) -> usize {
    self.inner.count_suffix_samples(ep_idx)
  }

  fn num_episodes(&self) -> usize {
    self.inner.num_episodes()
  }

  fn get_episodes_range(&self) -> (usize, usize) {
    self.inner.get_episodes_range()
  }

  fn get_episode_indices(&mut self, ep_idx: usize) -> Option<(usize, usize)> {
    self.inner.get_episode_indices(ep_idx)
  }

  fn get_episode_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    if let Some((orig_datum, orig_label)) = self.inner.get_episode_sample(datum_cfg, label_cfg, ep_idx, sample_idx) {
      let (new_datum, new_label) = self.augment.transform(orig_datum, orig_label);
      Some((new_datum, new_label))
    } else {
      None
    }
  }
}

pub struct PartitionDataSource {
  part_idx:     usize,
  parts:        usize,
  ep_idx_start: usize,
  ep_idx_end:   usize,
  prefix_offset:    usize,
  suffix_offset:    usize,
  data: Box<DataSource>,
}

impl PartitionDataSource {
  pub fn new(part_idx: usize, parts: usize, data: Box<DataSource>) -> PartitionDataSource {
    assert!(part_idx < parts);
    let (src_ep_idx_start, src_ep_idx_end) = data.get_episodes_range();
    let num_eps_per_part = (src_ep_idx_end - src_ep_idx_start) / parts;
    let part_ep_idx_start = src_ep_idx_start + part_idx * num_eps_per_part;
    let part_ep_idx_end = src_ep_idx_start + (part_idx + 1) * num_eps_per_part;
    let prefix_offset = data.count_prefix_samples(part_ep_idx_start);
    let suffix_offset = data.count_suffix_samples(part_ep_idx_start);
    PartitionDataSource{
      part_idx:     part_idx,
      parts:        parts,
      ep_idx_start: part_ep_idx_start,
      ep_idx_end:   part_ep_idx_end,
      prefix_offset:    prefix_offset,
      suffix_offset:    suffix_offset,
      data: data,
    }
  }
}

impl DataSource for PartitionDataSource {
  fn num_samples(&self) -> usize {
    let start_count = self.data.count_prefix_samples(self.ep_idx_start);
    let end_count = self.data.count_suffix_samples(self.ep_idx_end - 1);
    end_count - start_count
  }

  fn count_prefix_samples(&self, ep_idx: usize) -> usize {
    self.data.count_prefix_samples(ep_idx) - self.prefix_offset
  }

  fn count_suffix_samples(&self, ep_idx: usize) -> usize {
    self.data.count_suffix_samples(ep_idx) - self.suffix_offset
  }

  fn num_episodes(&self) -> usize {
    self.ep_idx_end - self.ep_idx_start
  }

  fn get_episodes_range(&self) -> (usize, usize) {
    (self.ep_idx_start, self.ep_idx_end)
  }

  fn get_episode_indices(&mut self, ep_idx: usize) -> Option<(usize, usize)> {
    if ep_idx >= self.ep_idx_start && ep_idx < self.ep_idx_end {
      self.data.get_episode_indices(ep_idx)
    } else {
      None
    }
  }

  fn get_episode_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    if ep_idx >= self.ep_idx_start && ep_idx < self.ep_idx_end {
      self.data.get_episode_sample(datum_cfg, label_cfg, ep_idx, sample_idx)
    } else {
      None
    }
  }
}

pub struct EpisoDbDataSource {
  config:     DataSourceConfig,
  data_db:    EpisoDb,
  labels_db:  EpisoDb,
  labels2_db: Option<EpisoDb>,
}

impl EpisoDbDataSource {
  pub fn open(config: DataSourceConfig) -> EpisoDbDataSource {
    let data_db = EpisoDb::open_read_only(config.data_path.clone());
    let labels_db = EpisoDb::open_read_only(config.maybe_labels_path.clone()
      .expect("episodb source requires category labels!"));
    assert_eq!(data_db.num_frames(), labels_db.num_frames());
    assert_eq!(data_db.num_episodes(), labels_db.num_episodes());
    EpisoDbDataSource{
      config:     config,
      data_db:    data_db,
      labels_db:  labels_db,
      labels2_db: None,
    }
  }
}

impl DataSource for EpisoDbDataSource {
  fn num_samples(&self) -> usize {
    self.data_db.num_frames()
  }

  fn count_prefix_samples(&self, ep_idx: usize) -> usize {
    self.data_db.get_prefix_frame_count(ep_idx).unwrap()
  }

  fn count_suffix_samples(&self, ep_idx: usize) -> usize {
    self.data_db.get_suffix_frame_count(ep_idx).unwrap()
  }

  fn num_episodes(&self) -> usize {
    self.data_db.num_episodes()
  }

  fn get_episodes_range(&self) -> (usize, usize) {
    (0, self.data_db.num_episodes())
  }

  fn get_episode_indices(&mut self, ep_idx: usize) -> Option<(usize, usize)> {
    let (start_idx, end_idx) = self.data_db.get_episode(ep_idx).unwrap();
    Some((start_idx, end_idx))
  }

  fn get_episode_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    let (start_idx, end_idx) = self.data_db.get_episode(ep_idx).unwrap();
    let (start_idx2, end_idx2) = self.labels_db.get_episode(ep_idx).unwrap();
    assert_eq!(start_idx, start_idx2);
    assert_eq!(end_idx, end_idx2);
    assert!(start_idx <= sample_idx);
    assert!(sample_idx < end_idx);

    // FIXME(20151129): Right now we skip samples with category `-1`;
    // should allow user to specify how to handle those.
    // XXX(20160129): Skip categories that are negative or out of bounds.
    let sample_label = match label_cfg {
      SampleLabelConfig::Category{num_categories} => {
        let category_value = self.labels_db.get_frame(sample_idx).unwrap();
        let category = Cursor::new(category_value).read_i32::<LittleEndian>().unwrap();
        if category < 0 || category >= num_categories {
          return None;
        }
        SampleLabel::Category{category: category}
      }
      SampleLabelConfig::LookaheadCategories{num_categories, lookahead} => {
        assert!(lookahead >= 1);
        if sample_idx + lookahead > end_idx {
          return None;
        }
        let mut lookahead_cats = Vec::with_capacity(lookahead);
        for k in 0 .. lookahead {
          let category_value = match self.labels_db.get_frame(sample_idx + k) {
            Some(value) => value,
            None => return None,
          };
          let category = Cursor::new(category_value).read_i32::<LittleEndian>().unwrap();
          if category < 0 || category >= num_categories {
            return None;
          }
          lookahead_cats.push(category);
        }
        SampleLabel::MultiCategory{categories: lookahead_cats}
      }
    };

    let datum_value = self.data_db.get_frame(sample_idx).unwrap();
    let sample_datum = datum_cfg.decode(datum_value);

    Some((sample_datum, Some(sample_label)))
  }
}

pub struct MnistDataSource {
  config:       DataSourceConfig,
  num_samples:  usize,
  frame_size:   usize,
  data_dims:    (usize, usize, usize),
  //data_buf:     Vec<u8>,
  data_file:    File,
  data_buf:     Mmap,
  labels_buf:   Vec<u8>,
}

impl MnistDataSource {
  pub fn open(config: DataSourceConfig) -> MnistDataSource {
    let mut data_file = match File::open(&config.data_path) {
      Ok(file) => file,
      Err(e) => panic!("failed to open mnist data file: {:?}", e),
    };
    //let mut data_reader = BufReader::new(data_file);
    //let (n, data_dims, data_buf) = Self::open_idx_file(data_reader);
    let (n, data_dims, data_buf) = Self::mmap_idx_file(&mut data_file);
    let data_dims = data_dims.unwrap();
    let labels_file = match File::open(config.maybe_labels_path.as_ref().unwrap()) {
      Ok(file) => file,
      Err(e) => panic!("failed to open mnist labels file: {:?}", e),
    };
    let mut labels_reader = BufReader::new(labels_file);
    let (_, _, labels_buf) = Self::open_idx_file(labels_reader);
    MnistDataSource{
      config:       config,
      num_samples:  n,
      frame_size:   data_dims.len(),
      data_dims:    data_dims,
      data_file:    data_file,
      data_buf:     data_buf,
      labels_buf:   labels_buf,
    }
  }

  fn open_idx_file<R>(mut reader: R) -> (usize, Option<(usize, usize, usize)>, Vec<u8>) where R: Read {
    let magic: u32 = reader.read_u32::<BigEndian>().unwrap();
    //println!("DEBUG: mnist: magic: {:x}", magic);
    let magic2 = (magic >> 8) as u8;
    let magic3 = (magic >> 0) as u8;
    assert_eq!(magic2, 0x08);
    let ndims = magic3 as usize;
    let mut dims = vec![];
    for d in 0 .. ndims {
      dims.push(reader.read_u32::<BigEndian>().unwrap() as usize);
    }
    let n = dims[0] as usize;
    let mut frame_size = 1;
    for d in 1 .. ndims {
      frame_size *= dims[d] as usize;
    }
    let mut buf = Vec::with_capacity(n * frame_size);
    reader.read_to_end(&mut buf);
    assert_eq!(buf.len(), n * frame_size);
    if ndims == 3 {
      (n, Some((dims[2], dims[1], 1)), buf)
    } else if ndims == 1 {
      (n, None, buf)
    } else {
      unimplemented!();
    }
  }

  fn mmap_idx_file(file: &mut File) -> (usize, Option<(usize, usize, usize)>, Mmap) {
    let magic: u32 = file.read_u32::<BigEndian>().unwrap();
    //println!("DEBUG: mnist: magic: {:x}", magic);
    let magic2 = (magic >> 8) as u8;
    let magic3 = (magic >> 0) as u8;
    assert_eq!(magic2, 0x08);
    let ndims = magic3 as usize;
    let mut dims = vec![];
    for d in 0 .. ndims {
      dims.push(file.read_u32::<BigEndian>().unwrap() as usize);
    }
    let n = dims[0] as usize;
    let mut frame_size = 1;
    for d in 1 .. ndims {
      frame_size *= dims[d] as usize;
    }
    //let mut buf = Vec::with_capacity(n * frame_size);
    let buf = match Mmap::open_with_offset(file, Protection::Read, (1 + ndims) * 4, frame_size * n) {
      Ok(buf) => buf,
      Err(e) => panic!("failed to mmap buffer: {:?}", e),
    };
    //reader.read_to_end(&mut buf);
    assert_eq!(buf.len(), n * frame_size);
    if ndims == 3 {
      (n, Some((dims[2], dims[1], 1)), buf)
    } else if ndims == 1 {
      (n, None, buf)
    } else {
      unimplemented!();
    }
  }
}

impl DataSource for MnistDataSource {
  fn num_samples(&self) -> usize {
    self.num_samples
  }

  fn count_prefix_samples(&self, ep_idx: usize) -> usize {
    ep_idx
  }

  fn count_suffix_samples(&self, ep_idx: usize) -> usize {
    ep_idx + 1
  }

  fn num_episodes(&self) -> usize {
    self.num_samples
  }

  fn get_episodes_range(&self) -> (usize, usize) {
    (0, self.num_samples)
  }

  fn get_episode_indices(&mut self, ep_idx: usize) -> Option<(usize, usize)> {
    //Some((idx, idx + 1))
    Some((0, 1))
  }

  fn get_episode_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    assert_eq!(sample_idx, 0);
    if ep_idx >= self.num_samples {
      return None;
    }
    //let datum_value = (&self.data_buf[ep_idx * self.frame_size .. (ep_idx + 1) * self.frame_size]).to_vec();
    let datum_value = (&unsafe { self.data_buf.as_slice() }[ep_idx * self.frame_size .. (ep_idx + 1) * self.frame_size]).to_vec();
    let sample_datum = match datum_cfg {
      SampleDatumConfig::Bytes3d => {
        SampleDatum::WHCBytes(Array3d::with_data(datum_value, self.data_dims))
      }
      _ => unimplemented!(),
    };
    let sample_label = match label_cfg {
      SampleLabelConfig::Category{num_categories} => {
        let category = self.labels_buf[ep_idx] as i32;
        SampleLabel::Category{category: category}
      }
      _ => unimplemented!(),
    };
    Some((sample_datum, Some(sample_label)))
  }
}

pub struct Cifar10DataSource {
  config:       DataSourceConfig,
  num_samples:  usize,
  frame_size:   usize,
  data_dims:    (usize, usize, usize),
  data_file:    File,
  data_buf:     Mmap,
}

impl Cifar10DataSource {
  pub fn open(config: DataSourceConfig) -> Cifar10DataSource {
    let mut data_file = match File::open(&config.data_path) {
      Ok(file) => file,
      Err(e) => panic!("failed to open cifar10 data file: {:?}", e),
    };
    let (n, data_dims, data_buf) = Self::mmap_bin_file(&mut data_file);
    Cifar10DataSource{
      config:       config,
      num_samples:  n,
      frame_size:   1 + data_dims.len(),
      data_dims:    data_dims,
      data_file:    data_file,
      data_buf:     data_buf,
    }
  }

  fn mmap_bin_file(file: &mut File) -> (usize, (usize, usize, usize), Mmap) {
    let dims = (32, 32, 3);
    let buf = match Mmap::open(file, Protection::Read) {
      Ok(buf) => buf,
      Err(e) => panic!("failed to mmap buffer: {:?}", e),
    };
    let buf_len = buf.len();
    let frame_size = 1 + dims.len();
    let n = buf_len / frame_size;
    assert_eq!(0, buf_len % frame_size);
    (n, dims, buf)
  }
}

impl DataSource for Cifar10DataSource {
  fn num_samples(&self) -> usize {
    self.num_samples
  }

  fn count_prefix_samples(&self, ep_idx: usize) -> usize {
    ep_idx
  }

  fn count_suffix_samples(&self, ep_idx: usize) -> usize {
    ep_idx + 1
  }

  fn num_episodes(&self) -> usize {
    self.num_samples
  }

  fn get_episodes_range(&self) -> (usize, usize) {
    (0, self.num_samples)
  }

  fn get_episode_indices(&mut self, ep_idx: usize) -> Option<(usize, usize)> {
    Some((0, 1))
  }

  fn get_episode_sample(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    assert_eq!(sample_idx, 0);
    if ep_idx >= self.num_samples {
      return None;
    }
    let datum_value = (&unsafe { self.data_buf.as_slice() }[ep_idx * self.frame_size + 1 .. (ep_idx + 1) * self.frame_size]).to_vec();
    let sample_datum = match datum_cfg {
      SampleDatumConfig::Bytes3d => {
        SampleDatum::WHCBytes(Array3d::with_data(datum_value, self.data_dims))
      }
      _ => unimplemented!(),
    };
    let label_value = &unsafe { self.data_buf.as_slice() }[ep_idx * self.frame_size .. ep_idx * self.frame_size + 1];
    let sample_label = match label_cfg {
      SampleLabelConfig::Category{num_categories} => {
        SampleLabel::Category{category: label_value[0] as i32}
      }
      _ => unimplemented!(),
    };
    Some((sample_datum, Some(sample_label)))
  }
}

pub struct LmdbCaffeDataIterator {
  env:      LmdbEnv,
  length:   usize,
}

impl LmdbCaffeDataIterator {
  pub fn open(config: DataSourceConfig) -> LmdbCaffeDataIterator {
    let mut env = match LmdbEnv::open_read_only(&config.data_path) {
      Ok(env) => env,
      Err(e) => panic!("failed to open lmdb env: {:?}", e),
    };
    if let Err(e) = env.set_map_size(1099511627776) {
      panic!("failed to set lmdb env map size: {:?}", e);
    }
    let length = match env.stat() {
      Ok(stat) => stat.entries(),
      Err(e) => panic!("failed to query lmdb env stat: {:?}", e),
    };
    LmdbCaffeDataIterator{
      env:      env,
      length:   length,
      //cursor:   cursor,
    }
  }
}

impl DataIterator for LmdbCaffeDataIterator {
  fn max_num_samples(&self) -> usize {
    self.length
  }

  fn each_sample(&mut self,
      datum_cfg: SampleDatumConfig,
      label_cfg: SampleLabelConfig,
      func: &mut FnMut(usize, &SampleDatum, Option<&SampleLabel>))
  {
    let cursor = match LmdbCursor::new_read_only(&self.env) {
      Ok(cursor) => cursor,
      Err(e) => panic!("failed to open lmdb cursor: {:?}", e),
    };
    for (i, kv) in cursor.iter().enumerate() {
      let value_bytes: &[u8] = kv.value;
      let mut datum: Datum = match parse_from_bytes(value_bytes) {
        Ok(m) => m,
        Err(e) => panic!("LmdbCaffeDataIterator: failed to parse Datum: {}", e),
      };
      let channels = datum.get_channels() as usize;
      let height = datum.get_height() as usize;
      let width = datum.get_width() as usize;
      let label = datum.get_label() as i32;
      let image_flat_bytes = datum.take_data();
      assert_eq!(image_flat_bytes.len(), width * height * channels);
      let image_bytes = Array3d::with_data(image_flat_bytes, (width, height, channels));
      func(i, &SampleDatum::WHCBytes(image_bytes), Some(&SampleLabel::Category{category: label}));
    }
  }
}
