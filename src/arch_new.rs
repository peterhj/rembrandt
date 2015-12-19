use layer_new::{
  Layer, InputLayer, LossLayer,
  LayerConfig, Phase,
};

use array_cuda::device::{
  DeviceContext, DeviceCtxRef,
};
use array_cuda::device::comm::{
  DeviceAllReduceSharedData, DeviceAllReduceWorker,
};
use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian};

use std::fs::{OpenOptions, copy, create_dir_all, read_link, remove_file};
use std::io::{Read, Write};
use std::os::unix::fs::{symlink};
use std::path::{PathBuf};
use std::sync::{Arc};

pub trait ArchWorker {
  fn initialize_layer_params(&mut self, ctx: &DeviceCtxRef);
  fn load_layer_params(&mut self, maybe_t: Option<usize>, ctx: &DeviceCtxRef);
  fn save_layer_params(&mut self, t: usize, ctx: &DeviceCtxRef);
  fn input_layer(&mut self, ctx: &DeviceCtxRef) -> &mut InputLayer;
  fn loss_layer(&mut self, ctx: &DeviceCtxRef) -> &mut LossLayer;
  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef);
  fn backward(&mut self, batch_size: usize, ctx: &DeviceCtxRef);
  fn descend(&mut self, learning_rate: f32, minibatch_size: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef);
}

#[derive(Clone)]
pub struct PipelineArchConfig {
  pub input:    LayerConfig,
  pub hidden:   Vec<LayerConfig>,
  pub loss:     LayerConfig,
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

pub struct PipelineArchWorker {
  batch_size:       usize,
  params_dir:       PathBuf,

  input_layer:      Box<InputLayer>,
  hidden_layers:    Vec<Box<Layer>>,
  loss_layer:       Box<LossLayer>,

  dev_allreduce:    DeviceAllReduceWorker<f32>,
}

impl PipelineArchWorker {
  pub fn new(
      batch_size:       usize,
      config:           PipelineArchConfig,
      params_dir:       PathBuf,
      local_tid:        usize,
      shared_data:      &PipelineArchSharedData,
      ctx:              &DeviceCtxRef)
      -> PipelineArchWorker
  {
    let input_layer = config.input.build_input_layer(batch_size, ctx);
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
    let loss_layer = config.loss.build_loss_layer(batch_size, Some(&*hidden_layers[hidden_layers.len()-1]), ctx);

    PipelineArchWorker{
      batch_size:       batch_size,
      params_dir:       params_dir,
      input_layer:      input_layer,
      hidden_layers:    hidden_layers,
      loss_layer:       loss_layer,
      dev_allreduce:    DeviceAllReduceWorker::new(local_tid, &shared_data.dev_allreduce, ctx),
    }
  }
}

impl ArchWorker for PipelineArchWorker {
  fn initialize_layer_params(&mut self, ctx: &DeviceCtxRef) {
    for layer in self.hidden_layers.iter_mut() {
      layer.initialize_params(ctx);
    }
  }

  fn load_layer_params(&mut self, maybe_t: Option<usize>, ctx: &DeviceCtxRef) {
    let mut latest_path = self.params_dir.clone();
    latest_path.push("layer_params.latest.blob");
    let blob_path = latest_path;

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
    create_dir_all(&self.params_dir)
      .ok().expect("failed to create params dir!");

    let mut blob = Vec::new();
    for layer in self.hidden_layers.iter_mut() {
      layer.save_params(&mut blob, ctx);
    }

    let mut checkpoint_path = self.params_dir.clone();
    checkpoint_path.push("checkpoint");
    let mut bak_checkpoint_path = self.params_dir.clone();
    bak_checkpoint_path.push("checkpoint.0");

    copy(&checkpoint_path, &bak_checkpoint_path).ok();

    let mut checkpoint_file = OpenOptions::new()
      .create(true).truncate(true).write(true)
      .open(&checkpoint_path)
      .ok().expect("failed to open checkpoint file!");
    checkpoint_file.write_u64::<LittleEndian>(t as u64)
      .ok().expect("failed to write checkpoint!");

    let mut blob_path = self.params_dir.clone();
    blob_path.push(&format!("layer_params.t_{}.blob", t));
    let blob_filename = PathBuf::from(&blob_path.file_name().unwrap());

    let mut blob_file = OpenOptions::new()
      .create(true).truncate(true).write(true)
      .open(&blob_path)
      .ok().expect("failed to open blob file to save layer params!");
    blob_file.write_all(&blob)
      .ok().expect("failed to write to blob file!");

    let mut latest_path = self.params_dir.clone();
    latest_path.push("layer_params.latest.blob");
    remove_file(&latest_path).ok();
    symlink(&blob_filename, &latest_path)
      .ok().expect("failed to symlink latest blob!");
  }

  fn input_layer(&mut self, ctx: &DeviceCtxRef) -> &mut InputLayer {
    &mut *self.input_layer
  }

  fn loss_layer(&mut self, ctx: &DeviceCtxRef) -> &mut LossLayer {
    &mut *self.loss_layer
  }

  fn forward(&mut self, batch_size: usize, phase: Phase, ctx: &DeviceCtxRef) {
    self.input_layer.forward(batch_size, phase, ctx);
    for layer in self.hidden_layers.iter_mut() {
      layer.forward(batch_size, phase, ctx);
    }
  }

  fn backward(&mut self, batch_size: usize, ctx: &DeviceCtxRef) {
    self.loss_layer.backward(batch_size, ctx);
    for layer in self.hidden_layers.iter_mut().rev() {
      layer.backward(batch_size, ctx);
    }
  }

  fn descend(&mut self, learning_rate: f32, minibatch_size: f32, l2_reg_coef: f32, ctx: &DeviceCtxRef) {
    for layer in self.hidden_layers.iter_mut() {
      layer.descend(learning_rate, minibatch_size, l2_reg_coef, ctx);
    }
  }
}
