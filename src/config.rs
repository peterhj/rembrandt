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

    let mut layer_graph: Graph<Box<LayerConfig>> = Graph::new();
    let mut layer_name_ids: HashMap<String, usize> = HashMap::new();
    let mut prev_layer_ids: Option<Vec<usize>> = None;

    let layer_nodes = &root[&"layer".to_string()];
    for node in unwrap!(layer_nodes.as_slice()).iter() {
      let layer_names: Vec<String> = if let Some(name) = node.lookup("name") {
        vec![name.as_str().expect("'name' should be a string!").to_string()]
      } else if let Some(names) = node.lookup("names") {
        names.as_slice().expect("'names' should be an array!")
          .iter()
          .map(|s| s.as_str().expect("'names' should be an array of strings!")
            .to_string())
          .collect()
      } else {
        unimplemented!();
      };

      let input_layer_names: Vec<String> = if let Some(name) = node.lookup("input") {
        vec![name.as_str().expect("'name' should be a string!").to_string()]
      } else if let Some(names) = node.lookup("inputs") {
        names.as_slice().expect("'names' should be an array!")
          .iter()
          .map(|s| s.as_str().expect("'names' should be an array of strings!")
            .to_string())
          .collect()
      } else {
        Vec::new()
      };

      let in_dims: Option<(usize, usize, usize)> = if let Some(prev_layer_ids) = prev_layer_ids.as_ref() {
        let mut in_width = 0;
        let mut in_height = 0;
        let mut in_channels = 0;
        for &layer_id in prev_layer_ids.iter() {
          let out_dims = layer_graph.vertexes[&layer_id].get_out_dims();
          if in_width == 0 {
            in_width = out_dims.0;
          } else {
            assert_eq!(in_width, out_dims.0);
          }
          if in_height == 0 {
            in_height = out_dims.0;
          } else {
            assert_eq!(in_height, out_dims.0);
          }
          in_channels += out_dims.2;
        }
        Some((in_width, in_height, in_channels))
      } else if input_layer_names.len() > 0 {
        let mut in_width = 0;
        let mut in_height = 0;
        let mut in_channels = 0;
        for input_layer_name in input_layer_names.iter() {
          let layer_id = layer_name_ids[input_layer_name];
          let out_dims = layer_graph.vertexes[&layer_id].get_out_dims();
          if in_width == 0 {
            in_width = out_dims.0;
          } else {
            assert_eq!(in_width, out_dims.0);
          }
          if in_height == 0 {
            in_height = out_dims.0;
          } else {
            assert_eq!(in_height, out_dims.0);
          }
          in_channels += out_dims.2;
        }
        Some((in_width, in_height, in_channels))
      } else {
        None
      };

      let layers: Vec<Box<LayerConfig>> = match node.lookup("type").expect("'layer' missing 'type' field!")
        .as_str().expect("'type' should be a string!")
      {
        "data" => {
          //println!("DEBUG: data layer");
          assert!(prev_layer_ids.is_none());
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
          (0 .. layer_names.len()).map(|_| Box::new(layer) as Box<LayerConfig>).collect()
        }
        "fully_conn" => {
          //println!("DEBUG: fully conn layer");
          let (_, _, in_channels) = in_dims.unwrap();
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
          (0 .. layer_names.len()).map(|_| Box::new(layer) as Box<LayerConfig>).collect()
        }
        "conv2d" => {
          //println!("DEBUG: conv2d layer");
          let (in_width, in_height, in_channels) = in_dims.unwrap();
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
          let init_weights = {
            let init_params = unwrap!(node.lookup("init_weights"));
            match unwrap!(unwrap!(init_params.lookup("kind")).as_str()) {
              "normal"  => ParamsInitialization::Normal{
                mean: unwrap!(unwrap!(init_params.lookup("mean")).as_float()) as f32,
                std:  unwrap!(unwrap!(init_params.lookup("std")).as_float()) as f32,
              },
              _ => unimplemented!(),
            }
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
            init_weights: init_weights,
          };
          (0 .. layer_names.len()).map(|_| Box::new(layer) as Box<LayerConfig>).collect()
        }
        "pool" => {
          let (in_width, in_height, channels) = in_dims.unwrap();
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
          (0 .. layer_names.len()).map(|_| Box::new(layer) as Box<LayerConfig>).collect()
        }
        "local_res_norm" => {
          let (in_width, in_height, channels) = in_dims.unwrap();
          let cross_size = unwrap!(unwrap!(node.lookup("cross_size")).as_integer()) as usize;
          let layer = LocalResNormLayerConfig{
            channels: channels,
            cross_size: cross_size,
          };
          (0 .. layer_names.len()).map(|_| Box::new(layer) as Box<LayerConfig>).collect()
        }
        "dropout" => {
          let (_, _, channels) = in_dims.unwrap();
          let drop_ratio = unwrap!(unwrap!(node.lookup("drop_ratio")).as_float()) as f32;
          let layer = DropoutLayerConfig{
            channels: channels,
            drop_ratio: drop_ratio,
          };
          (0 .. layer_names.len()).map(|_| Box::new(layer) as Box<LayerConfig>).collect()
        }
        "softmax_loss" => {
          let in_length = in_dims.unwrap().0 * in_dims.unwrap().1 * in_dims.unwrap().2;
          let num_categories = unwrap!(unwrap!(node.lookup("num_categories")).as_integer()) as usize;
          assert_eq!(in_length, num_categories);
          let layer = SoftmaxLossLayerConfig{
            num_categories: num_categories,
            do_mask: false, // FIXME(20151022)
          };
          (0 .. layer_names.len()).map(|_| Box::new(layer) as Box<LayerConfig>).collect()
        }
        _ => unimplemented!(),
      };

      for (idx, layer) in layers.into_iter().enumerate() {
        let layer_name = &layer_names[idx];
        let layer_id = layer_graph.insert_vertex(layer);
        layer_name_ids.insert(layer_name.clone(), layer_id);

        if prev_layer_ids.is_some() && input_layer_names.len() == 0 {
          // This is the "implicit" notation for specifying input layers.
          // When 'input' or 'inputs' are unspecified and there exist previous
          // layer definitions, assume a one-to-one mapping b/w them.
          if layer_names.len() == prev_layer_ids.as_ref().unwrap().len() {
            layer_graph.insert_edge(prev_layer_ids.as_ref().unwrap()[idx], layer_id);
          } else {
            unimplemented!();
          }
        } else if input_layer_names.len() > 0 {
          // This is the "explicit" notation for specifying input layers.
          // When 'input' or 'inputs' are provided, use them for each layer in
          // the layer definition.
          for input_layer_name in input_layer_names.iter() {
            if let Some(&input_layer_id) = layer_name_ids.get(input_layer_name) {
              layer_graph.insert_edge(input_layer_id, layer_id);
            }
          }
        }
      }

      prev_layer_ids = None;
      for layer_name in layer_names.iter() {
        if prev_layer_ids.is_none() {
          prev_layer_ids = Some(Vec::new());
        }
        prev_layer_ids.as_mut().unwrap().push(*layer_name_ids.get(layer_name).unwrap());
      }
    }

    ModelConfigFile{
      layer_graph: layer_graph,
    }
  }
}
