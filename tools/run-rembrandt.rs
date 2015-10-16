extern crate async_cuda;
extern crate rembrandt;
extern crate rand;

use rembrandt::config::{ModelConfigFile};
use rembrandt::data::{DatasetConfiguration, DataSourceBuilder};
use rembrandt::layer::*;
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{
  OptConfig, DescentSchedule, AnnealingPolicy,
  OptState, Optimizer, SgdOptimizer,
};

use async_cuda::context::{DeviceContext};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};

fn main() {
  //train_mnist();
  //train_imagenet();
}

fn train_mnist() {
  let mut rng = thread_rng();
  let ctx = DeviceContext::new(0);

  /*let model_conf = ModelConfigFile::open(&PathBuf::from("examples/mnist_lenet.model"));
  for &key in model_conf.layer_graph.topological_order().iter() {
    println!("DEBUG: layer: {:?}", model_conf.layer_graph.vertexes[&key]);
  }

  let model_conf = ModelConfigFile::open(&PathBuf::from("examples/imagenet_nin.model"));
  for &key in model_conf.layer_graph.topological_order().iter() {
    println!("DEBUG: layer: {:?}", model_conf.layer_graph.vertexes[&key]);
  }*/

  let opt_cfg = OptConfig{
    minibatch_size: 50,
    max_iters:      1200,
    init_step_size: 0.01,
    momentum:       0.0,
    l2_reg_coef:    0.0,
    anneal:         AnnealingPolicy::None,
    interval_size:  1000,
  };
  let descent = DescentSchedule::new(opt_cfg);

  /*let data_layer_cfg = DataLayerConfig{
    raw_width: 28, raw_height: 28,
    crop_width: 28, crop_height: 28,
    channels: 1,
  };
  let fc1_layer_cfg = FullyConnLayerConfig{
    in_channels: 28 * 28, out_channels: 10,
    act_fun: ActivationFunction::Identity,
  };
  let data_layer = DataLayer::new(0, data_layer_cfg);
  let fc1_layer = FullyConnLayer::new(0, fc1_layer_cfg, Some(&data_layer));
  let softmax_layer = SoftmaxLossLayer::new(0, 10, Some(&fc1_layer));*/

  let data_layer_cfg = DataLayerConfig{
    raw_width: 28, raw_height: 28,
    crop_width: 28, crop_height: 28,
    channels: 1,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 28, in_height: 28, in_channels: 1,
    conv_size: 5, conv_stride: 1, conv_pad: 2,
    out_channels: 16,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let pool1_layer_cfg = PoolLayerConfig{
    in_width: 28, in_height: 28, channels: 16,
    pool_size: 2, pool_stride: 2, pool_pad: 0,
    pool_kind: PoolKind::Max,
  };
  let fc1_layer_cfg = FullyConnLayerConfig{
    in_channels: 14 * 14 * 16, out_channels: 100,
    act_fun: ActivationFunction::Rect,
  };
  let drop1_layer_cfg = DropoutLayerConfig{
    channels:   100,
    drop_ratio: 0.1,
  };
  let fc2_layer_cfg = FullyConnLayerConfig{
    in_channels: 100, out_channels: 10,
    act_fun: ActivationFunction::Identity,
  };
  let data_layer = DataLayer::new(0, data_layer_cfg, 1);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, 1, Some(&data_layer), &ctx);
  let pool1_layer = PoolLayer::new(0, pool1_layer_cfg, 1, Some(&conv1_layer));
  let fc1_layer = FullyConnLayer::new(0, fc1_layer_cfg, 1, Some(&pool1_layer));
  let drop1_layer = DropoutLayer::new(0, drop1_layer_cfg, 1, Some(&fc1_layer));
  let fc2_layer = FullyConnLayer::new(0, fc2_layer_cfg, 1, Some(&drop1_layer));
  let softmax_layer = SoftmaxLossLayer::new(0, 10, 1, Some(&fc2_layer));

  let mut arch = LinearNetArch::new(data_layer, softmax_layer, vec![
      //Box::new(fc1_layer),
      Box::new(conv1_layer),
      Box::new(pool1_layer),
      Box::new(fc1_layer),
      Box::new(drop1_layer),
      Box::new(fc2_layer),
  ]);
  for layer in arch.hidden_layers() {
    layer.initialize_params(&ctx);
  }

  let mut state = OptState{epoch: 0, t: 0};

  let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("examples/mnist.data"));
  let mut train_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("train") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing train data source!");
  };
  let mut test_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("test") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing test data source!");
  };

  let sgd = SgdOptimizer;
  sgd.train(&opt_cfg, &mut state, &mut arch, &mut *train_data_source, &mut *test_data_source, &ctx);
}

fn train_imagenet() {
  let mut rng = thread_rng();
  let ctx = DeviceContext::new(0);

  // TODO
  let data_layer_cfg = DataLayerConfig{
    raw_width: 256, raw_height: 256,
    crop_width: 224, crop_height: 224,
    channels: 3,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 224, in_height: 224, in_channels: 3,
    conv_size: 11, conv_stride: 4, conv_pad: 0,
    out_channels: 96,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let cccp1_layer_cfg = Conv2dLayerConfig{
    in_width: 54, in_height: 54, in_channels: 96,
    conv_size: 1, conv_stride: 1, conv_pad: 0,
    out_channels: 96,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.05},
  };
  let cccp2_layer_cfg = Conv2dLayerConfig{
    in_width: 54, in_height: 54, in_channels: 96,
    conv_size: 1, conv_stride: 1, conv_pad: 0,
    out_channels: 96,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.05},
  };
  let pool1_layer_cfg = PoolLayerConfig{
    in_width: 54, in_height: 54, channels: 96,
    pool_size: 3, pool_stride: 2, pool_pad: 0,
    pool_kind: PoolKind::Max,
  };
  let conv2_layer_cfg = Conv2dLayerConfig{
    in_width: 27, in_height: 27, in_channels: 96,
    conv_size: 5, conv_stride: 1, conv_pad: 2,
    out_channels: 256,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.05},
  };
  let cccp3_layer_cfg = Conv2dLayerConfig{
    in_width: 27, in_height: 27, in_channels: 256,
    conv_size: 1, conv_stride: 1, conv_pad: 0,
    out_channels: 256,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.05},
  };
  let cccp4_layer_cfg = Conv2dLayerConfig{
    in_width: 27, in_height: 27, in_channels: 256,
    conv_size: 1, conv_stride: 1, conv_pad: 0,
    out_channels: 256,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.05},
  };
  let pool2_layer_cfg = PoolLayerConfig{
    in_width: 27, in_height: 27, channels: 256,
    pool_size: 3, pool_stride: 2, pool_pad: 0,
    pool_kind: PoolKind::Max,
  };
  let conv3_layer_cfg = Conv2dLayerConfig{
    in_width: 13, in_height: 13, in_channels: 256,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 384,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let cccp5_layer_cfg = Conv2dLayerConfig{
    in_width: 13, in_height: 13, in_channels: 384,
    conv_size: 1, conv_stride: 1, conv_pad: 0,
    out_channels: 384,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.05},
  };
  let cccp6_layer_cfg = Conv2dLayerConfig{
    in_width: 13, in_height: 13, in_channels: 384,
    conv_size: 1, conv_stride: 1, conv_pad: 0,
    out_channels: 384,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.05},
  };
  let pool3_layer_cfg = PoolLayerConfig{
    in_width: 13, in_height: 13, channels: 384,
    pool_size: 3, pool_stride: 2, pool_pad: 0,
    pool_kind: PoolKind::Max,
  };
  let drop_layer_cfg = DropoutLayerConfig{
    channels:   384,
    drop_ratio: 0.5,
  };
  let conv4_layer_cfg = Conv2dLayerConfig{
    in_width: 6, in_height: 6, in_channels: 384,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 1024,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.05},
  };
  let cccp7_layer_cfg = Conv2dLayerConfig{
    in_width: 6, in_height: 6, in_channels: 1024,
    conv_size: 1, conv_stride: 1, conv_pad: 0,
    out_channels: 1024,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.05},
  };
  let cccp8_layer_cfg = Conv2dLayerConfig{
    in_width: 6, in_height: 6, in_channels: 1024,
    conv_size: 1, conv_stride: 1, conv_pad: 0,
    out_channels: 1000,
    act_fun: ActivationFunction::Rect,
    init_weights: ParamsInitialization::Normal{mean: 0.0, std: 0.01},
  };
  let pool4_layer_cfg = PoolLayerConfig{
    in_width: 6, in_height: 6, channels: 1000,
    pool_size: 6, pool_stride: 1, pool_pad: 0,
    pool_kind: PoolKind::Average,
  };
  let data_layer = DataLayer::new(0, data_layer_cfg, 1);
  let conv1_layer = Conv2dLayer::new(0, conv1_layer_cfg, 1, Some(&data_layer), &ctx);
  let cccp1_layer = Conv2dLayer::new(0, cccp1_layer_cfg, 1, Some(&conv1_layer), &ctx);
  let cccp2_layer = Conv2dLayer::new(0, cccp2_layer_cfg, 1, Some(&cccp1_layer), &ctx);
  let pool1_layer = PoolLayer::new(0, pool1_layer_cfg, 1, Some(&cccp2_layer));
  let conv2_layer = Conv2dLayer::new(0, conv2_layer_cfg, 1, Some(&pool1_layer), &ctx);
  let cccp3_layer = Conv2dLayer::new(0, cccp3_layer_cfg, 1, Some(&conv2_layer), &ctx);
  let cccp4_layer = Conv2dLayer::new(0, cccp4_layer_cfg, 1, Some(&cccp3_layer), &ctx);
  let pool2_layer = PoolLayer::new(0, pool2_layer_cfg, 1, Some(&cccp4_layer));
  let conv3_layer = Conv2dLayer::new(0, conv3_layer_cfg, 1, Some(&pool2_layer), &ctx);
  let cccp5_layer = Conv2dLayer::new(0, cccp5_layer_cfg, 1, Some(&conv3_layer), &ctx);
  let cccp6_layer = Conv2dLayer::new(0, cccp6_layer_cfg, 1, Some(&cccp5_layer), &ctx);
  let pool3_layer = PoolLayer::new(0, pool3_layer_cfg, 1, Some(&cccp6_layer));
  let drop_layer = DropoutLayer::new(0, drop_layer_cfg, 1, Some(&pool3_layer));
  let conv4_layer = Conv2dLayer::new(0, conv4_layer_cfg, 1, Some(&drop_layer), &ctx);
  let cccp7_layer = Conv2dLayer::new(0, cccp7_layer_cfg, 1, Some(&conv4_layer), &ctx);
  let cccp8_layer = Conv2dLayer::new(0, cccp8_layer_cfg, 1, Some(&cccp7_layer), &ctx);
  let pool4_layer = PoolLayer::new(0, pool4_layer_cfg, 1, Some(&cccp8_layer));
  let softmax_layer = SoftmaxLossLayer::new(0, 1000, 1, Some(&pool4_layer));

  let mut arch = LinearNetArch::new(data_layer, softmax_layer, vec![
      Box::new(conv1_layer),
      Box::new(cccp1_layer),
      Box::new(cccp2_layer),
      Box::new(pool1_layer),
      Box::new(conv2_layer),
      Box::new(cccp3_layer),
      Box::new(cccp4_layer),
      Box::new(pool2_layer),
      Box::new(conv3_layer),
      Box::new(cccp5_layer),
      Box::new(cccp6_layer),
      Box::new(pool3_layer),
      Box::new(drop_layer),
      Box::new(conv4_layer),
      Box::new(cccp7_layer),
      Box::new(cccp8_layer),
      Box::new(pool4_layer),
  ]);
  for layer in arch.hidden_layers() {
    layer.initialize_params(&ctx);
  }

  let mut state = OptState{epoch: 0, t: 0};

  let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("examples/imagenet.data"));
  let mut train_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("train") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing train data source!");
  };
  let mut test_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("valid") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing test data source!");
  };

  let opt_cfg = OptConfig{
    minibatch_size: 64,
    max_iters:      440396,
    init_step_size: 0.01,
    momentum:       0.9,
    l2_reg_coef:    0.0005,
    anneal:         AnnealingPolicy::Step{step_iters: 200180, decay: 0.1},
    interval_size:  1000,
  };

  let sgd = SgdOptimizer;
  sgd.train(&opt_cfg, &mut state, &mut arch, &mut *train_data_source, &mut *test_data_source, &ctx);
}
