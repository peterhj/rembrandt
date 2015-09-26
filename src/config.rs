use graph::{Graph};
use layer::*;

use toml::{Parser, Value};

use std::collections::{HashMap};
use std::fs::{File};
use std::io::{Read};
use std::path::{PathBuf};

pub struct ModelConfigFile {
  pub layer_graph: Graph<Box<LayerConfig>>,
}

impl ModelConfigFile {
  pub fn open(path: &PathBuf) -> ModelConfigFile {
    let mut file = File::open(path)
      .ok().expect("failed to open config file");
    let mut s = String::new();
    file.read_to_string(&mut s)
      .ok().expect("failed to read config file as string!");
    let mut parser = Parser::new(&s);
    let root = parser.parse()
      .expect("failed to parse config file as toml!");

    //let mut data_layer: Option<DataLayer> = None;
    //let mut loss_layer: Option<SoftmaxLossLayer> = None;
    let mut layer_graph: Graph<Box<LayerConfig>> = Graph::new();
    let mut layer_name_ids: HashMap<String, usize> = HashMap::new();
    let mut prev_layer_id: Option<usize> = None;

    let layer_nodes = &root[&"layer".to_string()];
    for node in unwrap!(layer_nodes.as_slice()).iter() {
      let layer_name = node.lookup("name").expect("'layer' missing name!")
        .as_str().expect("'name' should be a string!");
      let layer_id = match node.lookup("type").expect("'layer' missing 'type' field!")
        .as_str().expect("'type' should be a string!")
      {
        "data" => {
          //println!("DEBUG: data layer");
          assert!(prev_layer_id.is_none());
          let raw_width = unwrap!(unwrap!(node.lookup("raw_width")).as_integer()) as usize;
          let raw_height = unwrap!(unwrap!(node.lookup("raw_height")).as_integer()) as usize;
          let crop_width = unwrap!(unwrap!(node.lookup("crop_width")).as_integer()) as usize;
          let crop_height = unwrap!(unwrap!(node.lookup("crop_height")).as_integer()) as usize;
          let channels = unwrap!(unwrap!(node.lookup("channels")).as_integer()) as usize;
          let layer = DataLayerConfig{
            raw_width: raw_width,
            raw_height: raw_height,
            crop_width: crop_width,
            crop_height: crop_height,
            channels: channels,
          };
          layer_graph.insert_vertex(Box::new(layer))
        }
        "fully_conn" => {
          //println!("DEBUG: fully conn layer");
          let in_channels = if let Some(prev_layer_id) = prev_layer_id {
            let in_dims = layer_graph.vertexes[&prev_layer_id].get_out_dims();
            in_dims.0 * in_dims.1 * in_dims.2
          } else {
            unimplemented!();
          };
          let out_channels = unwrap!(unwrap!(node.lookup("out_channels")).as_integer()) as usize;
          let act_fun = match unwrap!(unwrap!(node.lookup("act_func")).as_str()) {
            "identity"  => ActivationFunction::Identity,
            "rect"      => ActivationFunction::Rect,
            "sigmoid"   => ActivationFunction::Sigmoid,
            "tanh"      => ActivationFunction::Tanh,
            _ => unimplemented!(),
          };
          let layer = FullyConnLayerConfig{
            in_channels: in_channels,
            out_channels: out_channels,
            act_fun: act_fun,
          };
          layer_graph.insert_vertex(Box::new(layer))
        }
        "conv2d" => {
          //println!("DEBUG: conv2d layer");
          let (in_width, in_height, in_channels) = if let Some(prev_layer_id) = prev_layer_id {
            let in_dims = layer_graph.vertexes[&prev_layer_id].get_out_dims();
            in_dims
          } else {
            unimplemented!();
          };
          let conv_size = unwrap!(unwrap!(node.lookup("conv_size")).as_integer()) as usize;
          let conv_stride = unwrap!(unwrap!(node.lookup("conv_stride")).as_integer()) as usize;
          let conv_pad = unwrap!(unwrap!(node.lookup("conv_pad")).as_integer()) as usize;
          let out_channels = unwrap!(unwrap!(node.lookup("out_channels")).as_integer()) as usize;
          let act_fun = match unwrap!(unwrap!(node.lookup("act_func")).as_str()) {
            "identity"  => ActivationFunction::Identity,
            "rect"      => ActivationFunction::Rect,
            "sigmoid"   => ActivationFunction::Sigmoid,
            "tanh"      => ActivationFunction::Tanh,
            _ => unimplemented!(),
          };
          let layer = Conv2dLayerConfig{
            in_width: in_width,
            in_height: in_height,
            in_channels: in_channels,
            conv_size: conv_size,
            conv_stride: conv_stride,
            conv_pad: conv_pad,
            out_channels: out_channels,
            act_fun: act_fun,
          };
          layer_graph.insert_vertex(Box::new(layer))
        }
        "pool" => {
          let (in_width, in_height, channels) = if let Some(prev_layer_id) = prev_layer_id {
            let in_dims = layer_graph.vertexes[&prev_layer_id].get_out_dims();
            in_dims
          } else {
            unimplemented!();
          };
          let pool_size = unwrap!(unwrap!(node.lookup("pool_size")).as_integer()) as usize;
          let pool_stride = unwrap!(unwrap!(node.lookup("pool_stride")).as_integer()) as usize;
          let pool_pad = unwrap!(unwrap!(node.lookup("pool_pad")).as_integer()) as usize;
          let pool_kind = match unwrap!(unwrap!(node.lookup("pool_kind")).as_str()) {
            "max"     => PoolKind::Max,
            "average" => PoolKind::Average,
            _ => unimplemented!(),
          };
          let layer = PoolLayerConfig{
            in_width: in_width,
            in_height: in_height,
            channels: channels,
            pool_size: pool_size,
            pool_stride: pool_stride,
            pool_pad: pool_pad,
            pool_kind: pool_kind,
          };
          layer_graph.insert_vertex(Box::new(layer))
        }
        "dropout" => {
          let channels = if let Some(prev_layer_id) = prev_layer_id {
            let in_dims = layer_graph.vertexes[&prev_layer_id].get_out_dims();
            in_dims.0 * in_dims.1 * in_dims.2
          } else {
            unimplemented!();
          };
          let drop_ratio = unwrap!(unwrap!(node.lookup("drop_ratio")).as_float()) as f32;
          let layer = DropoutLayerConfig{
            channels: channels,
            drop_ratio: drop_ratio,
          };
          layer_graph.insert_vertex(Box::new(layer))
        }
        "softmax_loss" => {
          let in_channels = if let Some(prev_layer_id) = prev_layer_id {
            let in_dims = layer_graph.vertexes[&prev_layer_id].get_out_dims();
            in_dims.0 * in_dims.1 * in_dims.2
          } else {
            unimplemented!();
          };
          let num_categories = unwrap!(unwrap!(node.lookup("num_categories")).as_integer()) as usize;
          assert_eq!(in_channels, num_categories);
          let layer = SoftmaxLossLayerConfig{
            num_categories: num_categories,
          };
          layer_graph.insert_vertex(Box::new(layer))
        }
        _ => unimplemented!(),
      };
      layer_name_ids.insert(layer_name.to_string(), layer_id);
      if let Some(prev_layer_id) = prev_layer_id {
        layer_graph.insert_edge(prev_layer_id, layer_id);
      }
      prev_layer_id = Some(layer_id);
    }

    ModelConfigFile{
      layer_graph: layer_graph,
    }
  }
}
