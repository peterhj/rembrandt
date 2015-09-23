use data::{SampleDatum, SampleLabel};
use layer::*;
use opt::{DescentConfig};

use async_cuda::context::{DeviceContext};

pub trait NetArch {
  fn initialize_params(&mut self, ctx: &DeviceContext) {}
  fn load_sample(&mut self, datum: &SampleDatum, maybe_label: Option<SampleLabel>, ctx: &DeviceContext) {}
  fn reset_gradients(&mut self, ctx: &DeviceContext) {}
  fn forward(&mut self, ctx: &DeviceContext) {}
  fn backward(&mut self, descent: &DescentConfig, ctx: &DeviceContext) {}
  fn descend(&mut self, descent: &DescentConfig, ctx: &DeviceContext) {}
}

pub struct LinearNetArch {
  pub data_layer:     DataLayer,
  pub loss_layer:     SoftmaxLossLayer,
  pub hidden_layers:  Vec<Box<Layer>>,
}

impl LinearNetArch {
  pub fn new(data_layer: DataLayer, loss_layer: SoftmaxLossLayer, hidden_layers: Vec<Box<Layer>>) -> LinearNetArch {
    LinearNetArch{
      //data_layer:     DataLayer::new(data_layer_cfg),
      //loss_layer:     SoftmaxLossLayer::new(Some(&*hidden_layers[hidden_layers.len()-1]), num_categories),
      data_layer:     data_layer,
      loss_layer:     loss_layer,
      hidden_layers:  hidden_layers,
    }
  }
}

impl NetArch for LinearNetArch {
  fn initialize_params(&mut self, ctx: &DeviceContext) {
    for layer in self.hidden_layers.iter_mut() {
      layer.initialize_params(ctx);
    }
  }

  fn load_sample(&mut self, datum: &SampleDatum, maybe_label: Option<SampleLabel>, ctx: &DeviceContext) {
    self.data_layer.load_sample(datum, ctx);
    self.loss_layer.load_sample(maybe_label, ctx);
  }

  fn reset_gradients(&mut self, ctx: &DeviceContext) {
    for layer in self.hidden_layers.iter_mut() {
      layer.reset_gradients(ctx);
    }
  }

  fn forward(&mut self, ctx: &DeviceContext) {
    self.data_layer.forward(ctx);
    for layer in self.hidden_layers.iter_mut() {
      layer.forward(ctx);
    }
    self.loss_layer.forward(ctx);
  }

  fn backward(&mut self, descent: &DescentConfig, ctx: &DeviceContext) {
    self.loss_layer.backward(descent, ctx);
    for layer in self.hidden_layers.iter_mut().rev() {
      layer.backward(descent, ctx);
    }
  }

  fn descend(&mut self, descent: &DescentConfig, ctx: &DeviceContext) {
    for layer in self.hidden_layers.iter_mut() {
      layer.descend(descent, ctx);
    }
  }
}
