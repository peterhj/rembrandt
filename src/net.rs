use data::{SampleDatum, SampleLabel};
use graph::{Graph};
use layer::*;
use opt::{DescentConfig, DescentSchedule};

use async_cuda::context::{DeviceContext};

use std::collections::{BTreeMap};

pub trait NetArch {
  fn batch_size(&self) -> usize;

  fn data_layer(&mut self) -> &mut DataLayer {
    unimplemented!();
  }

  /*fn hidden_layers(&mut self) -> &mut [Box<Layer>] {
    unimplemented!();
  }*/

  fn hidden_layers_forward(&mut self) -> Vec<&mut Box<Layer>> {
    unimplemented!();
  }

  fn hidden_layers_backward(&mut self) -> Vec<&mut Box<Layer>> {
    unimplemented!();
  }

  fn loss_layer(&mut self) -> &mut SoftmaxLossLayer {
    unimplemented!();
  }
}

pub struct LinearNetArch {
  batch_size:     usize,
  data_layer:     DataLayer,
  loss_layer:     SoftmaxLossLayer,
  hidden_layers:  Vec<Box<Layer>>,
}

impl LinearNetArch {
  pub fn new(batch_size: usize, data_layer: DataLayer, loss_layer: SoftmaxLossLayer, hidden_layers: Vec<Box<Layer>>) -> LinearNetArch {
    LinearNetArch{
      batch_size:     batch_size,
      data_layer:     data_layer,
      loss_layer:     loss_layer,
      hidden_layers:  hidden_layers,
    }
  }

  pub fn with_graph(layer_graph: Graph<Box<LayerConfig>>) -> LinearNetArch {
    for &layer_id in layer_graph.topological_order().iter() {
      if layer_graph.in_edges[&layer_id].len() <= 1 {
        // TODO
        //layer_graph.vertexes[&layer_id]
      } else {
        panic!("PANIC: layer graph is not a linear chain!");
      }
    }

    // TODO
    unimplemented!();
  }
}

impl NetArch for LinearNetArch {
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn data_layer(&mut self) -> &mut DataLayer {
    &mut self.data_layer
  }

  /*fn hidden_layers(&mut self) -> &mut [Box<Layer>] {
    &mut self.hidden_layers
  }*/

  fn hidden_layers_forward(&mut self) -> Vec<&mut Box<Layer>> {
    self.hidden_layers.iter_mut().collect()
  }

  fn hidden_layers_backward(&mut self) -> Vec<&mut Box<Layer>> {
    self.hidden_layers.iter_mut().rev().collect()
  }

  fn loss_layer(&mut self) -> &mut SoftmaxLossLayer {
    &mut self.loss_layer
  }
}

pub struct DagNetArch {
  layer_graph:    Graph<Box<LayerConfig>>,
  layers:         BTreeMap<usize, Box<Layer>>,
  data_layer_id:  usize,
  loss_layer_id:  usize,
  fwd_layer_ids:  Vec<usize>,
  bwd_layer_ids:  Vec<usize>,
  fwd_inv_ids:    Vec<usize>,
  bwd_inv_ids:    Vec<usize>,
  fwd_order:      BTreeMap<usize, usize>,
  bwd_order:      BTreeMap<usize, usize>,
}

impl DagNetArch {
  pub fn new(init_layers: Graph<Box<LayerConfig>>) -> DagNetArch {
    // TODO:
    // 1. fixup layer configs by inserting Split and Join layer configs.
    unimplemented!();
  }
}

//impl NetArch for DagNetArch {
impl DagNetArch {
  fn data_layer(&mut self) -> &mut Layer {
    //self.layers.get_mut(&self.data_layer_id).unwrap()
    // TODO
    unimplemented!();
  }

  //fn hidden_layers_forward(&mut self) -> (Vec<&mut Box<Layer>>, &[usize]) {
  fn hidden_layers_forward(&mut self) -> Vec<&mut Box<Layer>> {
    let &mut DagNetArch{ref mut layers, ref fwd_order, ..} = self;
    let mut ordered_layers: Vec<_> = layers.iter_mut()
      .map(|(&layer_id, layer)| (fwd_order[&layer_id], layer))
      .collect();
    (&mut ordered_layers).sort_by(|a, b| a.0.cmp(&b.0));
    let flat_layers: Vec<_> = ordered_layers.into_iter()
      .map(|(_, layer)| layer)
      .collect();
    flat_layers
  }

  fn loss_layer(&mut self) -> &mut Layer {
    //self.layers.get_mut(&self.loss_layer_id).unwrap()
    // TODO
    unimplemented!();
  }
}
