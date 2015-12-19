use layer_new::{
  Layer, InputLayer, LossLayer,
  LayerConfig,
};

use array_cuda::device::{
  DeviceContext, DeviceCtxRef,
};
use array_cuda::device::comm::{
  DeviceAllReduceSharedData, DeviceAllReduceWorker,
};

use std::sync::{Arc};

pub trait ArchWorker {
  fn initialize_layer_params(&mut self, ctx: &DeviceCtxRef);
  fn load_layer_params(&mut self, ctx: &DeviceCtxRef);
  fn save_layer_params(&mut self, ctx: &DeviceCtxRef);
  fn input_layer(&mut self, ctx: &DeviceCtxRef) -> &mut InputLayer;
  fn loss_layer(&mut self, ctx: &DeviceCtxRef) -> &mut LossLayer;
  fn forward(&mut self, ctx: &DeviceCtxRef);
  fn backward(&mut self, ctx: &DeviceCtxRef);
  fn descend(&mut self, ctx: &DeviceCtxRef);
}

#[derive(Clone)]
pub struct PipelineArchConfig {
  pub input:    LayerConfig,
  pub hidden:   Vec<LayerConfig>,
  pub loss:     LayerConfig,
}

pub struct PipelineArchSharedData {
  num_workers:  usize,
  allreduce:    DeviceAllReduceSharedData<f32>,
}

impl PipelineArchSharedData {
  pub fn new(num_workers: usize, config: &PipelineArchConfig, ctxs: &[DeviceContext]) -> PipelineArchSharedData {
    let mut num_params = 0;
    for layer in config.hidden.iter() {
      num_params += layer.params_len();
    }
    PipelineArchSharedData{
      num_workers:  num_workers,
      allreduce:    DeviceAllReduceSharedData::new(num_workers, num_params, ctxs),
    }
  }
}

pub struct PipelineArchWorker {
  input_layer:      Box<InputLayer>,
  hidden_layers:    Vec<Box<Layer>>,
  loss_layer:       Box<LossLayer>,
}

impl PipelineArchWorker {
  pub fn new(
      input_layer:      Box<InputLayer>,
      hidden_layers:    Vec<Box<Layer>>,
      loss_layer:       Box<LossLayer>,
      shared_data:      Arc<PipelineArchSharedData>)
      -> PipelineArchWorker
  {
    // TODO(20151217)
    //unimplemented!();

    PipelineArchWorker{
      input_layer:      input_layer,
      hidden_layers:    hidden_layers,
      loss_layer:       loss_layer,
    }
  }
}

impl ArchWorker for PipelineArchWorker {
  fn initialize_layer_params(&mut self, ctx: &DeviceCtxRef) {
    for layer in self.hidden_layers.iter_mut() {
      layer.initialize_params(ctx);
    }
  }

  fn load_layer_params(&mut self, ctx: &DeviceCtxRef) {
    // TODO(20151218)
    unimplemented!();
  }

  fn save_layer_params(&mut self, ctx: &DeviceCtxRef) {
    // TODO(20151218)
    unimplemented!();
  }

  fn input_layer(&mut self, ctx: &DeviceCtxRef) -> &mut InputLayer {
    &mut *self.input_layer
  }

  fn loss_layer(&mut self, ctx: &DeviceCtxRef) -> &mut LossLayer {
    &mut *self.loss_layer
  }

  fn forward(&mut self, ctx: &DeviceCtxRef) {
    for layer in self.hidden_layers.iter_mut() {
    }
  }

  fn backward(&mut self, ctx: &DeviceCtxRef) {
    for layer in self.hidden_layers.iter_mut() {
    }
  }

  fn descend(&mut self, ctx: &DeviceCtxRef) {
    for layer in self.hidden_layers.iter_mut() {
    }
  }
}
