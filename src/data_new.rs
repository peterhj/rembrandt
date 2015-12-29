use array_new::{NdArraySerialize, Array3d};
use byteorder::{LittleEndian, ReadBytesExt};
use episodb::{EpisoDb};
//use random::{XorShift128Plus};
use toml::{Parser};

use rand::{Rng, thread_rng};
use rand::distributions::{IndependentSample};
use rand::distributions::range::{Range};
use std::collections::{HashMap};
use std::fs::{File};
use std::io::{Read, BufReader, Cursor};
use std::path::{PathBuf};

pub enum SampleDatum {
  RowMajorBytes(Array3d<u8>),
}

#[derive(Clone, Copy)]
pub enum SampleLabelConfig {
  Category,
  //Category2,
  Lookahead{lookahead: usize},
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
  fn max_num_samples(&self) -> usize;
  fn each_sample(&mut self, label_cfg: SampleLabelConfig, /*filter: &Fn(usize) -> bool,*/ f: &mut FnMut(usize, &SampleDatum, Option<&SampleLabel>));

  /*fn get_episode_indices(&mut self, ep_idx: usize) -> Option<(usize, usize)> {
    unimplemented!();
  }

  fn get_episode_sample(&mut self, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    unimplemented!();
  }*/
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
  fn max_num_samples(&self) -> usize {
    self.data.num_samples()
  }

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, /*filter: &Fn(usize) -> bool,*/ f: &mut FnMut(usize, &SampleDatum, Option<&SampleLabel>)) {
    let mut epoch_idx = 0;
    let (ep_idx_start, ep_idx_end) = self.data.get_episodes_range();
    for ep_idx in (ep_idx_start .. ep_idx_end) {
      let (start_idx, end_idx) = self.data.get_episode_indices(ep_idx).unwrap();
      for sample_idx in (start_idx .. end_idx) {
        if let Some((datum, maybe_label)) = self.data.get_episode_sample(label_cfg, ep_idx, sample_idx) {
          f(epoch_idx, &datum, maybe_label.as_ref());
          epoch_idx += 1;
        }
      }
    }
  }
}

pub struct RandomEpisodeIterator {
  //rng:  XorShift128PlusRng,
  data: Box<DataSource>,
}

impl RandomEpisodeIterator {
  pub fn new(data: Box<DataSource>) -> RandomEpisodeIterator {
    RandomEpisodeIterator{
      //rng:  XorShift128PlusRng::from_seed([thread_rng().gen(), thread_rng().gen()]),
      data: data,
    }
  }
}

impl DataIterator for RandomEpisodeIterator {
  fn max_num_samples(&self) -> usize {
    self.data.num_samples()
  }

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<&SampleLabel>)) {
    let mut epoch_idx = 0;
    let (ep_idx_start, ep_idx_end) = self.data.get_episodes_range();
    let ep_idx_range = Range::new(ep_idx_start, ep_idx_end);
    for _ in (0 .. self.data.num_samples()) {
      let ep_idx = ep_idx_range.ind_sample(&mut thread_rng());
      let (start_idx, end_idx) = self.data.get_episode_indices(ep_idx).unwrap();
      let sample_idx = thread_rng().gen_range(start_idx, end_idx);
      if let Some((datum, maybe_label)) = self.data.get_episode_sample(label_cfg, ep_idx, sample_idx) {
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
  fn max_num_samples(&self) -> usize {
    self.data.num_samples()
  }

  fn each_sample(&mut self, label_cfg: SampleLabelConfig, f: &mut FnMut(usize, &SampleDatum, Option<&SampleLabel>)) {
    let limit = self.data.num_samples();
    let mut counter = 0;
    let mut epoch_idx = 0;
    let (ep_idx_start, ep_idx_end) = self.data.get_episodes_range();
    loop {
      for ep_idx in ep_idx_start .. ep_idx_end {
        let (start_idx, end_idx) = self.data.get_episode_indices(ep_idx).unwrap();
        let sample_idx = thread_rng().gen_range(start_idx, end_idx);
        if let Some((datum, maybe_label)) = self.data.get_episode_sample(label_cfg, ep_idx, sample_idx) {
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
          _ => panic!("unknown data source: '{}'", src_name),
        }
      }
      None => panic!("dataset missing key: '{}'", key),
    }
  }
}

pub trait DataSource {
  fn num_samples(&self) -> usize;
  fn count_prefix_samples(&self, ep_idx: usize) -> usize;
  fn count_suffix_samples(&self, ep_idx: usize) -> usize;
  fn num_episodes(&self) -> usize;

  fn get_episodes_range(&self) -> (usize, usize);
  fn get_episode_indices(&mut self, ep_idx: usize) -> Option<(usize, usize)>;
  fn get_episode_sample(&mut self, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)>;
}

pub struct PartitionDataSource {
  part_idx:     usize,
  parts:        usize,
  ep_idx_start: usize,
  ep_idx_end:   usize,
  data: Box<DataSource>,
}

impl PartitionDataSource {
  pub fn new(part_idx: usize, parts: usize, data: Box<DataSource>) -> PartitionDataSource {
    assert!(part_idx < parts);
    let (src_ep_idx_start, src_ep_idx_end) = data.get_episodes_range();
    let num_eps_per_part = (src_ep_idx_end - src_ep_idx_start) / parts;
    let part_ep_idx_start = src_ep_idx_start + part_idx * num_eps_per_part;
    let part_ep_idx_end = src_ep_idx_start + (part_idx + 1) * num_eps_per_part;
    PartitionDataSource{
      part_idx:     part_idx,
      parts:        parts,
      ep_idx_start: part_ep_idx_start,
      ep_idx_end:   part_ep_idx_end,
      data: data,
    }
  }
}

impl DataSource for PartitionDataSource {
  fn num_samples(&self) -> usize {
    let start_count = self.data.count_prefix_samples(self.ep_idx_start);
    let end_count = self.data.count_suffix_samples(self.ep_idx_end);
    end_count - start_count
  }

  fn count_prefix_samples(&self, ep_idx: usize) -> usize {
    unimplemented!();
  }

  fn count_suffix_samples(&self, ep_idx: usize) -> usize {
    unimplemented!();
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

  fn get_episode_sample(&mut self, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    if ep_idx >= self.ep_idx_start && ep_idx < self.ep_idx_end {
      self.data.get_episode_sample(label_cfg, ep_idx, sample_idx)
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

  fn get_episode_sample(&mut self, label_cfg: SampleLabelConfig, ep_idx: usize, sample_idx: usize) -> Option<(SampleDatum, Option<SampleLabel>)> {
    let (start_idx, end_idx) = self.data_db.get_episode(ep_idx).unwrap();
    let (start_idx2, end_idx2) = self.labels_db.get_episode(ep_idx).unwrap();
    assert_eq!(start_idx, start_idx2);
    assert_eq!(end_idx, end_idx2);
    assert!(start_idx <= sample_idx);
    assert!(sample_idx < end_idx);

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
      SampleLabelConfig::Lookahead{lookahead} => {
        assert!(lookahead >= 1);
        if sample_idx + lookahead > end_idx {
          return None;
        }
        let mut lookahead_cats = Vec::with_capacity(lookahead);
        for k in (0 .. lookahead) {
          let category_value = self.labels_db.get_frame(sample_idx + k).unwrap();
          let category = Cursor::new(category_value).read_i32::<LittleEndian>().unwrap();
          if category == -1 {
            return None;
          }
          lookahead_cats.push(category);
        }
        SampleLabel::MultiCategory{categories: lookahead_cats}
      }
    };
    let datum_value = self.data_db.get_frame(sample_idx).unwrap();
    let sample_datum = SampleDatum::RowMajorBytes(Array3d::deserialize(&mut Cursor::new(datum_value))
      .ok().expect("arraydb source failed to deserialize datum!"));
    Some((sample_datum, Some(sample_label)))
  }
}