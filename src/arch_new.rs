use layer_new::{
  Layer, InputLayer, LossLayer,
  LayerConfig, Phase,
  Data3dLayerConfig,
  Conv2dLayerConfig,
  CategoricalLossLayerConfig,
  MultiCategoricalLossLayerConfig,
};

use array_cuda::device::{
  DeviceContext, DeviceCtxRef,
};
use array_cuda::device::comm::allreduce::{
  DeviceAllReduceSharedData, DeviceAllReduceWorker,
};
use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, SeedableRng};
use std::fs::{OpenOptions, copy, create_dir_all, read_link, remove_file};
use std::io::{Read, BufRead, Write, BufReader};
use std::os::unix::fs::{symlink};
use std::path::{Path, PathBuf};
use std::sync::{Arc};

pub trait AtomicData {
  fn reset(&self);
  fn update(&self, batch_size: usize, loss_layer: &mut LossLayer, phase: Phase, ctx: &DeviceCtxRef);
}

impl AtomicData for () {
  #[inline]
  fn reset(&self) {
  }

  #[inline]
  fn update(&self, _: usize, _: &mut LossLayer, _: Phase, _: &DeviceCtxRef) {
  }
}

pub trait Worker {
  fn tid(&self) -> usize;
  fn num_workers(&self) -> usize;
  fn sync_workers(&self);
}

pub trait ArchWorker<A>: Worker where A: AtomicData {
  fn batch_size(&self) -> usize;
  fn initialize_layer_params(&mut self, ctx: &DeviceCtxRef);
  fn load_layer_params(&mut self, maybe_t: Option<usize>, ctx: &DeviceCtxRef);
  fn load_layer_params_dir(&mut self, save_dir: &Path, maybe_t: Option<usize>, ctx: &DeviceCtxRef);
  fn save_layer_params(&mut self, t: usize, ctx: &DeviceCtxRef);
  fn save_layer_params_dir(&mut self, save_dir: &Path, t: usize, ctx: &DeviceCtxRef);
  fn input_layer(&mut self) -> &mut InputLayer;
  fn loss_layer(&mut self) -> &mut LossLayer;
  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef);
  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef);
  fn descend(&mut self, scale: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef);
  fn reset_gradients(&mut self, momentum: f32, ctx: &DeviceCtxRef);
  fn dev_allreduce_sum_gradients(&mut self, ctx: &DeviceCtxRef);
  fn reduce_loss(&mut self, ctx: &DeviceCtxRef) -> f32;
  fn reset_loss(&mut self, ctx: &DeviceCtxRef);
  fn get_atomic_data(&self) -> &A;
  fn reset_atomic_data(&self);
  fn update_atomic_data(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef);
}

#[derive(Clone)]
pub struct PipelineArchConfig {
  pub input:    Option<LayerConfig>,
  pub hidden:   Vec<LayerConfig>,
  pub loss:     Option<LayerConfig>,
}

impl PipelineArchConfig {
  pub fn new() -> PipelineArchConfig {
    PipelineArchConfig{
      input:    None,
      hidden:   vec![],
      loss:     None,
    }
  }

  pub fn data3d(&mut self, layer_cfg: Data3dLayerConfig) -> &mut PipelineArchConfig {
    self.input = Some(LayerConfig::Data3d(layer_cfg));
    self
  }

  pub fn conv2d(&mut self, layer_cfg: Conv2dLayerConfig) -> &mut PipelineArchConfig {
    self.hidden.push(LayerConfig::Conv2d(layer_cfg));
    self
  }

  pub fn softmax_kl_loss(&mut self, layer_cfg: CategoricalLossLayerConfig) -> &mut PipelineArchConfig {
    self.loss = Some(LayerConfig::SoftmaxKLLoss(layer_cfg));
    self
  }

  pub fn multi_softmax_kl_loss(&mut self, layer_cfg: MultiCategoricalLossLayerConfig) -> &mut PipelineArchConfig {
    self.loss = Some(LayerConfig::MultiSoftmaxKLLoss(layer_cfg));
    self
  }
}

pub struct PipelineArchSharedData {
  num_workers:      usize,
  dev_allreduce:    DeviceAllReduceSharedData<f32>,
}

impl PipelineArchSharedData {
  pub fn new(num_workers: usize, config: &PipelineArchConfig, ctxs: &[DeviceContext]) -> PipelineArchSharedData {
    let mut num_params = 0;
    for layer in config.hidden.iter() {
      num_params += layer.params_len();
    }
    PipelineArchSharedData{
      num_workers:      num_workers,
      dev_allreduce:    DeviceAllReduceSharedData::new(num_workers, num_params, ctxs),
    }
  }
}

// FIXME(20160117)
#[derive(Clone)]
pub struct PipelineArchWorkerBuilder;

impl PipelineArchWorkerBuilder {
  pub fn into_worker<A>(self) -> PipelineArchWorker<A> where A: AtomicData {
    // FIXME(20160117)
    unimplemented!();
  }
}

pub struct PipelineArchWorker<A> where A: AtomicData {
  batch_size:       usize,
  params_dir:       PathBuf,
  shared_seed:      [u64; 2],

  input_layer:      Box<InputLayer>,
  hidden_layers:    Vec<Box<Layer>>,
  loss_layer:       Box<LossLayer>,

  local_tid:        usize,
  num_workers:      usize,
  dev_allreduce:    DeviceAllReduceWorker<f32>,
  atomic_data:      Arc<A>,
}

impl<A> PipelineArchWorker<A> where A: AtomicData {
  pub fn new(
      batch_size:       usize,
      config:           PipelineArchConfig,
      params_dir:       PathBuf,
      local_tid:        usize,
      shared_seed:      [u64; 2],
      shared_data:      &PipelineArchSharedData,
      atomic_data:      Arc<A>,
      ctx:              &DeviceCtxRef)
      -> PipelineArchWorker<A>
  {
    assert!(config.input.is_some());
    assert!(config.loss.is_some());
    let input_layer = config.input.unwrap().build_input_layer(batch_size, ctx);
    let mut hidden_layers = vec![];
    for (i, layer) in config.hidden.iter().enumerate() {
      if i == 0 {
        let layer = layer.build_layer(batch_size, Some(input_layer.downcast()), ctx);
        hidden_layers.push(layer);
      } else {
        let layer = layer.build_layer(batch_size, Some(&*hidden_layers[i-1]), ctx);
        hidden_layers.push(layer);
      }
    }
    let loss_layer = config.loss.unwrap().build_loss_layer(batch_size, Some(&*hidden_layers[hidden_layers.len()-1]), ctx);
    let num_workers = shared_data.num_workers;

    PipelineArchWorker{
      batch_size:       batch_size,
      params_dir:       params_dir,
      shared_seed:      shared_seed,
      input_layer:      input_layer,
      hidden_layers:    hidden_layers,
      loss_layer:       loss_layer,
      local_tid:        local_tid,
      num_workers:      num_workers,
      dev_allreduce:    DeviceAllReduceWorker::<f32>::new(local_tid, &shared_data.dev_allreduce, ctx),
      atomic_data:      atomic_data,
    }
  }
}

impl<A> Worker for PipelineArchWorker<A> where A: AtomicData {
  fn tid(&self) -> usize {
    self.local_tid
  }

  fn num_workers(&self) -> usize {
    self.dev_allreduce.num_workers()
  }

  fn sync_workers(&self) {
    self.dev_allreduce.barrier.wait();
  }
}

impl<A> ArchWorker<A> for PipelineArchWorker<A> where A: AtomicData {
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn initialize_layer_params(&mut self, ctx: &DeviceCtxRef) {
    /*println!("DEBUG: arch: {}/{} init params seed: {:?}",
        self.local_tid, self.num_workers, self.shared_seed);*/
    let mut rng = Xorshiftplus128Rng::from_seed(self.shared_seed);
    for layer in self.hidden_layers.iter_mut() {
      let h1 = rng.next_u64();
      let h2 = rng.next_u64();
      /*println!("DEBUG: arch: {}/{} layer seed: {:?}",
          self.local_tid, self.num_workers, [h1, h2]);*/
      layer.initialize_params([h1, h2], ctx);
    }
  }

  fn load_layer_params(&mut self, maybe_t: Option<usize>, ctx: &DeviceCtxRef) {
    let save_dir = self.params_dir.clone();
    self.load_layer_params_dir(&save_dir, maybe_t, ctx);
  }

  fn load_layer_params_dir(&mut self, save_dir: &Path, maybe_t: Option<usize>, ctx: &DeviceCtxRef) {
    let mut checkpoint_path = PathBuf::from(save_dir);
    checkpoint_path.push("checkpoint");

    let checkpoint_file = OpenOptions::new()
      .read(true)
      .open(&checkpoint_path)
      .ok().expect("failed to open checkpoint file!");
    let mut latest_t: usize = 0;
    for line in BufReader::new(checkpoint_file).lines() {
      let line = line.unwrap();
      latest_t = line.parse().unwrap();
      break;
    }

    let blob_path = match maybe_t {
      Some(t) => {
        assert!(t <= latest_t, "failed to load layer params: t: {} checkpoint: {}", t, latest_t);
        let mut t_path = PathBuf::from(save_dir);
        t_path.push(&format!("layer_params.t_{}.blob", t));
        t_path
      }
      None => {
        let mut latest_path = PathBuf::from(save_dir);
        latest_path.push("layer_params.latest.blob");
        latest_path
      }
    };

    let mut blob_file = OpenOptions::new()
      .read(true)
      .open(&blob_path)
      .ok().expect("failed to open blob file to load layer params!");
    let mut blob = Vec::new();
    blob_file.read_to_end(&mut blob)
      .ok().expect("failed to read from blob file!");

    let mut cursor_offset = 0;
    for layer in self.hidden_layers.iter_mut() {
      let progress = layer.load_params(&blob[cursor_offset ..], ctx);
      cursor_offset += progress;
    }
  }

  fn save_layer_params(&mut self, t: usize, ctx: &DeviceCtxRef) {
    let save_dir = self.params_dir.clone();
    self.save_layer_params_dir(&save_dir, t, ctx);
  }

  fn save_layer_params_dir(&mut self, save_dir: &Path, t: usize, ctx: &DeviceCtxRef) {
    create_dir_all(save_dir);
      //.ok().expect("failed to create params dir!");

    let mut checkpoint_path = PathBuf::from(save_dir);
    checkpoint_path.push("checkpoint");
    let mut bak_checkpoint_path = PathBuf::from(save_dir);
    bak_checkpoint_path.push("checkpoint.0");

    copy(&checkpoint_path, &bak_checkpoint_path).ok();

    let mut checkpoint_file = OpenOptions::new()
      .create(true).truncate(true).write(true)
      .open(&checkpoint_path)
      .ok().expect("failed to open checkpoint file!");
    /*checkpoint_file.write_u64::<LittleEndian>(t as u64)
      .ok().expect("failed to write checkpoint!");*/
    writeln!(checkpoint_file, "{}", t);

    let mut blob = Vec::new();
    for layer in self.hidden_layers.iter_mut() {
      layer.save_params(&mut blob, ctx);
    }

    let mut blob_path = PathBuf::from(save_dir);
    blob_path.push(&format!("layer_params.t_{}.blob", t));
    let blob_filename = PathBuf::from(&blob_path.file_name().unwrap());

    let mut blob_file = OpenOptions::new()
      .create(true).truncate(true).write(true)
      .open(&blob_path)
      .ok().expect("failed to open blob file to save layer params!");
    blob_file.write_all(&blob)
      .ok().expect("failed to write to blob file!");

    let mut latest_path = PathBuf::from(save_dir);
    latest_path.push("layer_params.latest.blob");
    remove_file(&latest_path).ok();
    symlink(&blob_filename, &latest_path)
      .ok().expect("failed to symlink latest blob!");
  }

  fn input_layer(&mut self) -> &mut InputLayer {
    &mut *self.input_layer
  }

  fn loss_layer(&mut self) -> &mut LossLayer {
    &mut *self.loss_layer
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    self.input_layer.forward(batch_size, phase, ctx);
    for layer in self.hidden_layers.iter_mut() {
      layer.forward(batch_size, phase, ctx);
    }
    self.loss_layer.forward(batch_size, phase, ctx);
  }

  fn backward(&mut self, batch_size: usize, scale: f32, ctx: &DeviceCtxRef) {
    self.loss_layer.backward(batch_size, scale, ctx);
    for layer in self.hidden_layers.iter_mut().rev() {
      layer.backward(batch_size, scale, ctx);
    }
  }

  fn descend(&mut self, scale: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef) {
    for layer in self.hidden_layers.iter_mut() {
      layer.descend(scale, l2_reg_coef, ctx);
    }
  }

  fn reset_gradients(&mut self, momentum: f32, ctx: &DeviceCtxRef) {
    for layer in self.hidden_layers.iter_mut() {
      layer.reset_gradients(momentum, ctx);
    }
  }

  fn dev_allreduce_sum_gradients(&mut self, ctx: &DeviceCtxRef) {
    let num_workers = self.num_workers();
    if num_workers <= 1 {
      // Do nothing.
      //println!("DEBUG: arch: no allreduce");
    } else {
      //println!("DEBUG: arch: allreduce sum grads {}", self.num_workers());
      let mut offset = 0;
      for layer in self.hidden_layers.iter_mut() {
        //info!("arch: {}/{} load layer", self.local_tid, num_workers);
        layer.dev_allreduce_load(&mut self.dev_allreduce, offset, ctx);
        offset += layer.config().params_len();
      }
      self.dev_allreduce.communicate(ctx);
      let mut offset = 0;
      for layer in self.hidden_layers.iter_mut() {
        layer.dev_allreduce_store(&mut self.dev_allreduce, offset, ctx);
        offset += layer.config().params_len();
      }
    }
  }

  fn reduce_loss(&mut self, ctx: &DeviceCtxRef) -> f32 {
    // TODO(20151230)
    unimplemented!();
  }

  fn reset_loss(&mut self, ctx: &DeviceCtxRef) {
    // TODO(20151230)
    unimplemented!();
  }

  fn get_atomic_data(&self) -> &A {
    &*self.atomic_data
  }

  fn reset_atomic_data(&self) {
    self.atomic_data.reset();
  }

  fn update_atomic_data(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    let &mut PipelineArchWorker{
      ref atomic_data, ref mut loss_layer, .. } = self;
    atomic_data.update(batch_size, &mut **loss_layer, phase, ctx);
  }
}
