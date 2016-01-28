use data::{SampleDatum, SampleLabel};
use graph::{Graph};
use layer::*;
use opt::{DescentConfig, DescentSchedule, OptPhase};

use async_cuda::context::{DeviceContext};
use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian};

use std::collections::{BTreeMap};
use std::fs::{OpenOptions, copy, create_dir_all, read_link, remove_file};
use std::io::{Read, Write};
use std::os::unix::fs::{symlink};
use std::path::{PathBuf};

pub trait NetArch {
  fn batch_size(&self) -> usize;

  fn initialize_layer_params(&mut self, ctx: &DeviceContext) {
    unimplemented!();
  }

  fn load_layer_params(&mut self, maybe_t: Option<usize>, ctx: &DeviceContext) {
    unimplemented!();
  }

  fn save_layer_params(&mut self, t: usize, ctx: &DeviceContext) {
    unimplemented!();
  }

  fn data_layer(&mut self) -> &mut DataLayer {
    unimplemented!();
  }

  fn hidden_layers_forward(&mut self) -> Vec<&mut Box<Layer>> {
    unimplemented!();
  }

  fn hidden_layers_backward(&mut self) -> Vec<&mut Box<Layer>> {
    unimplemented!();
  }

  //fn loss_layer(&mut self) -> &mut SoftmaxLossLayer {
  fn loss_layer(&mut self) -> &mut Layer {
    unimplemented!();
  }

  fn evaluate(&mut self, phase: OptPhase, ctx: &DeviceContext) {
    unimplemented!();
  }

  fn evaluate_gradients(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    unimplemented!();
  }
}

pub struct LinearNetArch {
  params_path:    PathBuf,
  batch_size:     usize,
  data_layer:     DataLayer,
  loss_layer:     Box<Layer>,
  hidden_layers:  Vec<Box<Layer>>,
}

impl LinearNetArch {
  pub fn new(params_path: PathBuf, batch_size: usize, data_layer: DataLayer, loss_layer: Box<Layer>, hidden_layers: Vec<Box<Layer>>) -> LinearNetArch {
    LinearNetArch{
      params_path:    params_path,
      batch_size:     batch_size,
      data_layer:     data_layer,
      loss_layer:     loss_layer,
      hidden_layers:  hidden_layers,
    }
  }

  /*pub fn with_graph(layer_graph: Graph<Box<LayerConfig>>) -> LinearNetArch {
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
  }*/
}

impl NetArch for LinearNetArch {
  fn batch_size(&self) -> usize {
    self.batch_size
  }

  fn initialize_layer_params(&mut self, ctx: &DeviceContext) {
    for layer in self.hidden_layers_forward() {
      layer.initialize_params(ctx);
    }
  }

  fn load_layer_params(&mut self, maybe_t: Option<usize>, ctx: &DeviceContext) {
    /*let mut blob_path = self.params_path.clone();

    // FIXME(20151023): if t is None, should load the _newest_ params
    // available.
    let t = if let Some(t) = maybe_t {
      t
    } else {
      // FIXME(20151120): load from symlink.
      let mut checkpoint_path = self.params_path.clone();
      checkpoint_path.push("checkpoint");
      let mut checkpoint_file = OpenOptions::new()
        .read(true)
        .open(&checkpoint_path)
        .ok().expect("LinearNetArch failed to open checkpoint file!");
      let t = checkpoint_file.read_u64::<LittleEndian>()
        .ok().expect("LinearNetArch failed to read checkpoint!") as usize;
      t
    };

    blob_path.push(&format!("layer_params.t_{}.blob", t));*/

    let mut latest_path = self.params_path.clone();
    latest_path.push("layer_params.latest.blob");
    /*let blob_path = read_link(&latest_path)
      .ok().expect("LinearNetArch failed to read latest symlink!");*/
    let blob_path = latest_path;

    let mut blob_file = match OpenOptions::new().read(true).open(&blob_path) {
      //.ok().expect("LinearNetArch failed to open blob file to load layer params!");
      Ok(file) => file,
      Err(e) => {
        panic!("LinearNetArch failed to open blob path {:?} to load layer params: {:?}",
            blob_path, e);
      }
    };
    let mut blob = Vec::new();
    blob_file.read_to_end(&mut blob)
      .ok().expect("LinearNetArch failed to read from blob file!");

    let mut cursor_offset = 0;
    for layer in self.hidden_layers_forward() {
      let progress = layer.load_params(&blob[cursor_offset ..], ctx);
      cursor_offset += progress;
    }
  }

  fn save_layer_params(&mut self, t: usize, ctx: &DeviceContext) {
    create_dir_all(&self.params_path)
      .ok().expect("LinearNetArch failed to create params dir!");

    let mut blob = Vec::new();
    for layer in self.hidden_layers_forward() {
      let layer_blob = layer.save_params(ctx);
      blob.extend(&layer_blob);
    }

    let mut checkpoint_path = self.params_path.clone();
    checkpoint_path.push("checkpoint");
    let mut bak_checkpoint_path = self.params_path.clone();
    bak_checkpoint_path.push("checkpoint.0");

    copy(&checkpoint_path, &bak_checkpoint_path).ok();

    let mut checkpoint_file = OpenOptions::new()
      .create(true).truncate(true).write(true)
      .open(&checkpoint_path)
      .ok().expect("LinearNetArch failed to open checkpoint file!");
    checkpoint_file.write_u64::<LittleEndian>(t as u64)
      .ok().expect("LinearNetArch failed to write checkpoint!");

    let mut blob_path = self.params_path.clone();
    blob_path.push(&format!("layer_params.t_{}.blob", t));
    let blob_filename = PathBuf::from(&blob_path.file_name().unwrap());

    let mut blob_file = OpenOptions::new()
      .create(true).truncate(true).write(true)
      .open(&blob_path)
      .ok().expect("LinearNetArch failed to open blob file to save layer params!");
    blob_file.write_all(&blob)
      .ok().expect("LinearNetArch failed to write to blob file!");

    let mut latest_path = self.params_path.clone();
    latest_path.push("layer_params.latest.blob");
    remove_file(&latest_path).ok();
    symlink(&blob_filename, &latest_path)
      .ok().expect("LinearNetArch failed to symlink latest blob!");
  }

  fn data_layer(&mut self) -> &mut DataLayer {
    &mut self.data_layer
  }

  fn hidden_layers_forward(&mut self) -> Vec<&mut Box<Layer>> {
    self.hidden_layers.iter_mut().collect()
  }

  fn hidden_layers_backward(&mut self) -> Vec<&mut Box<Layer>> {
    self.hidden_layers.iter_mut().rev().collect()
  }

  fn loss_layer(&mut self) -> &mut Layer {
    &mut *self.loss_layer
  }

  fn evaluate(&mut self, phase: OptPhase, ctx: &DeviceContext) {
    let batch_size = self.batch_size();
    for layer in self.hidden_layers_forward() {
      layer.forward(phase, batch_size, ctx);
    }
    self.loss_layer().forward(phase, batch_size, ctx);
  }

  fn evaluate_gradients(&mut self, descent: &DescentSchedule, ctx: &DeviceContext) {
    let batch_size = self.batch_size();
    self.loss_layer().backward(descent, batch_size, ctx);
    for layer in self.hidden_layers_backward() {
      layer.backward(descent, batch_size, ctx);
    }
  }
}

/*pub struct DagNetArch {
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
}*/
