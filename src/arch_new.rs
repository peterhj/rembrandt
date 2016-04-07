use layer_new::{
  Layer, InputLayer, LossLayer,
  LayerConfig, Phase,
  Data3dLayerConfig,
  AffineLayerConfig,
  Conv2dLayerConfig,
  Conv2dLayerConfigV2,
  CategoricalLossLayerConfig,
  MultiCategoricalLossLayerConfig,
};

use array_cuda::device::{
  DeviceContext, DeviceCtxRef,
};
use array_cuda::device::comm::{
  for_all_devices,
};
use array_cuda::device::comm::allreduce::{
  DeviceAllReduceSharedData, DeviceAllReduceWorker,
};
use array_new::{NdArraySerialize, ArrayZeroExt, Array2d};
use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, SeedableRng, thread_rng};
use std::fs::{OpenOptions, copy, create_dir_all, read_link, remove_file};
use std::io::{Read, BufRead, Write, BufReader, Cursor};
use std::os::unix::fs::{symlink};
use std::path::{Path, PathBuf};
use std::sync::{Arc};
use std::sync::mpsc::{Receiver, Sender, channel};

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
  fn batch_capacity(&self) -> usize;
  fn initialize_layer_params(&mut self, ctx: &DeviceCtxRef);
  fn params_path(&self) -> PathBuf;
  fn load_layer_params(&mut self, maybe_t: Option<usize>, ctx: &DeviceCtxRef) -> Result<(), ()>;
  fn load_layer_params_dir(&mut self, save_dir: &Path, maybe_t: Option<usize>, ctx: &DeviceCtxRef) -> Result<(), ()>;
  fn load_layer_params_into_mem(&mut self, save_dir: &Path, maybe_t: Option<usize>) -> Result<Vec<u8>, ()>;
  fn load_layer_params_from_mem(&mut self, blob: &[u8], ctx: &DeviceCtxRef);
  fn save_layer_params(&mut self, t: usize, ctx: &DeviceCtxRef);
  fn save_layer_params_dir(&mut self, save_dir: &Path, t: usize, ctx: &DeviceCtxRef);
  fn save_layer_params_to_mem(&mut self, ctx: &DeviceCtxRef) -> Vec<u8>;
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

pub struct ReshapePipelineArchConfigs {
  pub old_arch: PipelineArchConfig,
  pub new_arch: PipelineArchConfig,
}

impl ReshapePipelineArchConfigs {
  pub fn new(old_arch: PipelineArchConfig, new_arch: PipelineArchConfig) -> ReshapePipelineArchConfigs {
    ReshapePipelineArchConfigs{
      old_arch: old_arch,
      new_arch: new_arch,
    }
  }

  pub fn reshape_params(&self, old_params_path: &Path, new_params_path: &Path) {
    assert_eq!(self.old_arch.hidden.len(), self.new_arch.hidden.len());

    let mut old_blob = vec![];
    let mut old_blob_file = match OpenOptions::new()
      .read(true)
      .open(old_params_path) 
    {
        Ok(file) => file,
        Err(_) => panic!(),
    };
    match old_blob_file.read_to_end(&mut old_blob) {
      Ok(_) => {}
      Err(_) => panic!(),
    };

    let mut new_blob = vec![];

    let mut reader = Cursor::new(&old_blob);
    for k in 0 .. self.old_arch.hidden.len() {
      //match (self.old_arch.hidden[k], self.new_arch.hidden[k]) {
        /*(LayerConfig::Affine{..}, LayerConfig::Affine{..}) => {
          unimplemented!();
        }*/
      match self.old_arch.hidden[k] {
        //(LayerConfig::Conv2d{layer_cfg}, LayerConfig::Conv2d{new_layer_cfg: layer_cfg}) => {
        LayerConfig::Conv2d(layer_cfg)=> {
          let old_layer_cfg = layer_cfg;
          match self.new_arch.hidden[k] {
            LayerConfig::Conv2d(layer_cfg)=> {
              let new_layer_cfg = layer_cfg;
              let weights: Array2d<f32> = Array2d::deserialize(&mut reader).unwrap();
              let bias: Array2d<f32> = Array2d::deserialize(&mut reader).unwrap();
              if old_layer_cfg == new_layer_cfg {
                println!("DEBUG: reshape: layer {}: unchanged", k);
                weights.serialize(&mut new_blob);
                bias.serialize(&mut new_blob);
              } else {
                let old_conv_size = old_layer_cfg.conv_size;
                let old_in_dims = old_layer_cfg.in_dims;
                let old_out_channels = old_layer_cfg.out_channels;
                let new_conv_size = new_layer_cfg.conv_size;
                let new_in_dims = new_layer_cfg.in_dims;
                let new_out_channels = new_layer_cfg.out_channels;
                if old_conv_size == new_conv_size && old_in_dims == new_in_dims {
                  assert!(new_out_channels <= old_out_channels);
                  let mut new_weights: Array2d<f32> = Array2d::zeros((new_conv_size * new_conv_size * new_in_dims.2, new_out_channels));
                  let old_cols = old_conv_size * old_conv_size * old_in_dims.2;
                  let new_cols = new_conv_size * new_conv_size * new_in_dims.2;
                  for i in 0 .. new_cols {
                    for j in 0 .. new_out_channels {
                      new_weights.as_mut_slice()[i + j * new_cols] = weights.as_slice()[i + j * old_cols];
                    }
                  }
                  let mut new_bias: Array2d<f32> = Array2d::zeros((1, new_out_channels));
                  for j in 0 .. new_out_channels {
                    new_bias.as_mut_slice()[j] = bias.as_slice()[j];
                  }
                  println!("DEBUG: reshape: layer {}: downsizing out channels: {} => {}",
                      k, old_out_channels, new_out_channels);
                  new_weights.serialize(&mut new_blob);
                  new_bias.serialize(&mut new_blob);
                } else {
                  unimplemented!();
                }
              }
            }
            _ => unimplemented!(),
          }
        }
        _ => unimplemented!(),
      }
    }

    let mut new_blob_file = OpenOptions::new()
      .create(true).truncate(true).write(true)
      .open(new_params_path)
      .ok().expect("failed to open blob file to save layer params!");
    new_blob_file.write_all(&new_blob)
      .ok().expect("failed to write to blob file!");
  }
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

  pub fn affine(&mut self, layer_cfg: AffineLayerConfig) -> &mut PipelineArchConfig {
    self.hidden.push(LayerConfig::Affine(layer_cfg));
    self
  }

  pub fn conv2d(&mut self, layer_cfg: Conv2dLayerConfig) -> &mut PipelineArchConfig {
    self.hidden.push(LayerConfig::Conv2d(layer_cfg));
    self
  }

  pub fn conv2d_v2(&mut self, layer_cfg: Conv2dLayerConfigV2) -> &mut PipelineArchConfig {
    self.hidden.push(LayerConfig::Conv2d_V2(layer_cfg));
    self
  }

  pub fn softmax_kl_loss(&mut self, layer_cfg: CategoricalLossLayerConfig) -> &mut PipelineArchConfig {
    self.loss = Some(LayerConfig::SoftmaxKLLoss(layer_cfg));
    self
  }

  pub fn softmax_ind_loss(&mut self, layer_cfg: CategoricalLossLayerConfig) -> &mut PipelineArchConfig {
    self.loss = Some(LayerConfig::SoftmaxIndicatorLoss(layer_cfg));
    self
  }

  pub fn logistic_ind_loss(&mut self, layer_cfg: CategoricalLossLayerConfig) -> &mut PipelineArchConfig {
    self.loss = Some(LayerConfig::LogisticIndicatorLoss(layer_cfg));
    self
  }

  /*pub fn antilogistic_kl_loss(&mut self, layer_cfg: CategoricalLossLayerConfig) -> &mut PipelineArchConfig {
    self.loss = Some(LayerConfig::AntilogisticKLLoss(layer_cfg));
    self
  }*/

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
pub struct PipelineArchWorkerBuilder {
  num_workers:  usize,
  batch_cap:    usize,
  arch_cfg:     PipelineArchConfig,
  params_path:  PathBuf,
  shared_seed:  [u64; 2],
  shared_data:  Arc<PipelineArchSharedData>,
}

impl PipelineArchWorkerBuilder {
  pub fn new(num_workers: usize, batch_capacity: usize, arch_cfg: PipelineArchConfig, params_path: PathBuf, shared_seed: [u64; 2]) -> PipelineArchWorkerBuilder {
    let shared_data = for_all_devices(num_workers, |contexts| {
      Arc::new(PipelineArchSharedData::new(num_workers, &arch_cfg, contexts))
    });
    PipelineArchWorkerBuilder{
      num_workers:  num_workers,
      batch_cap:    batch_capacity,
      arch_cfg:     arch_cfg,
      params_path:  params_path,
      shared_seed:  shared_seed,
      shared_data:  shared_data,
    }
  }

  pub fn into_worker<A>(self, tid: usize, extra_data: Arc<A>, ctx: &DeviceCtxRef) -> PipelineArchWorker<A> where A: AtomicData {
    let arch = PipelineArchWorker::new(
        self.batch_cap,
        self.arch_cfg,
        self.params_path,
        tid,
        //[thread_rng().next_u64(), thread_rng().next_u64()], // FIXME(20160313): should be shared seed.
        self.shared_seed,
        &self.shared_data,
        extra_data,
        ctx,
    );
    arch
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

  // FIXME(20160207)
  /*tid0_recver:      Option<Receiver<Vec<u8>>>,
  sender:           Sender<Vec<u8>>,*/
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
  fn batch_capacity(&self) -> usize {
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

  fn params_path(&self) -> PathBuf {
    self.params_dir.clone()
  }

  fn load_layer_params(&mut self, maybe_t: Option<usize>, ctx: &DeviceCtxRef) -> Result<(), ()> {
    let save_dir = self.params_dir.clone();
    self.load_layer_params_dir(&save_dir, maybe_t, ctx)
  }

  fn load_layer_params_dir(&mut self, save_dir: &Path, maybe_t: Option<usize>, ctx: &DeviceCtxRef) -> Result<(), ()> {
    let blob = match self.load_layer_params_into_mem(save_dir, maybe_t) {
      Ok(blob) => blob,
      Err(_) => return Err(()),
    };
    self.load_layer_params_from_mem(&blob, ctx);
    Ok(())
  }

  fn load_layer_params_into_mem(&mut self, save_dir: &Path, maybe_t: Option<usize>) -> Result<Vec<u8>, ()> {
    let blob_path = match maybe_t {
      Some(t) => {
        let mut checkpoint_path = PathBuf::from(save_dir);
        checkpoint_path.push("checkpoint");

        let checkpoint_file = match OpenOptions::new()
          .read(true)
          .open(&checkpoint_path)
        {
          Ok(file) => file,
          Err(_) => return Err(()),
        };
        let mut latest_t: usize = 0;
        for line in BufReader::new(checkpoint_file).lines() {
          let line = line.unwrap();
          latest_t = line.parse().unwrap();
          break;
        }
        //assert!(t <= latest_t, "failed to load layer params: t: {} checkpoint: {}", t, latest_t);
        if t > latest_t {
          return Err(());
        }

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

    let mut blob_file = match OpenOptions::new()
      .read(true)
      .open(&blob_path) 
    {
        Ok(file) => file,
        Err(_) => return Err(()),
    };
    let mut blob = Vec::new();
    match blob_file.read_to_end(&mut blob) {
      Ok(_) => {}
      Err(_) => return Err(()),
    };
    Ok(blob)
  }

  fn load_layer_params_from_mem(&mut self, blob: &[u8], ctx: &DeviceCtxRef) {
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

    let blob = self.save_layer_params_to_mem(ctx);

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

  fn save_layer_params_to_mem(&mut self, ctx: &DeviceCtxRef) -> Vec<u8> {
    let mut blob = Vec::new();
    for layer in self.hidden_layers.iter_mut() {
      layer.save_params(&mut blob, ctx);
    }
    blob
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
