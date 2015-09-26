use data::{SampleDatum, SampleLabel};
use graph::{Graph};
use layer::*;
use opt::{DescentConfig, DescentSchedule};

use async_cuda::context::{DeviceContext};

use std::collections::{BTreeMap};

pub trait NetArch {
  fn data_layer(&mut self) -> &mut DataLayer {
    unimplemented!();
  }

  fn hidden_layers(&mut self) -> &mut [Box<Layer>] {
    unimplemented!();
  }

  fn loss_layer(&mut self) -> &mut SoftmaxLossLayer {
    unimplemented!();
  }
}

pub struct LinearNetArch {
  pub data_layer:     DataLayer,
  pub loss_layer:     SoftmaxLossLayer,
  pub hidden_layers:  Vec<Box<Layer>>,
}

impl LinearNetArch {
  pub fn new(data_layer: DataLayer, loss_layer: SoftmaxLossLayer, hidden_layers: Vec<Box<Layer>>) -> LinearNetArch {
    LinearNetArch{
      data_layer:     data_layer,
      loss_layer:     loss_layer,
      hidden_layers:  hidden_layers,
    }
  }
}

impl NetArch for LinearNetArch {
  fn data_layer(&mut self) -> &mut DataLayer {
    &mut self.data_layer
  }

  fn hidden_layers(&mut self) -> &mut [Box<Layer>] {
    &mut self.hidden_layers
  }

  fn loss_layer(&mut self) -> &mut SoftmaxLossLayer {
    &mut self.loss_layer
  }
}

pub struct DagNetArch {
  layer_graph:  Graph<Box<LayerConfig>>,
  layers:       BTreeMap<usize, Box<Layer>>,
}

impl DagNetArch {
}

impl NetArch for DagNetArch {
}
