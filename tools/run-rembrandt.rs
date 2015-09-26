extern crate async_cuda;
extern crate rembrandt;
extern crate rand;

use rembrandt::config::{ModelConfigFile};
use rembrandt::data::{DatasetConfiguration, DataSourceBuilder};
use rembrandt::layer::*;
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{
  DescentConfig, DescentSchedule, AnnealingPolicy,
  OptState, Optimizer, SgdOptimizer,
};

use async_cuda::context::{DeviceContext};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};

fn main() {
  let mut rng = thread_rng();
  let ctx = DeviceContext::new(0);

  let descent_cfg = DescentConfig{
    minibatch_size: 50,
    max_iters:      1200,
    init_step_size: 0.01,
    momentum:       0.0,
    l2_reg_coef:    0.0,
    anneal:         AnnealingPolicy::None,
  };
  let descent = DescentSchedule::new(descent_cfg);

  let model_conf = ModelConfigFile::open(&PathBuf::from("examples/mnist_lenet.model"));

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

  let data_layer = DataLayer::new(data_layer_cfg);
  let conv1_layer = Conv2dLayer::new(Some(&data_layer), conv1_layer_cfg, &ctx);
  let pool1_layer = PoolLayer::new(Some(&conv1_layer), pool1_layer_cfg);
  let fc1_layer = FullyConnLayer::new(Some(&pool1_layer), fc1_layer_cfg);
  let drop1_layer = DropoutLayer::new(Some(&fc1_layer), drop1_layer_cfg);
  let fc2_layer = FullyConnLayer::new(Some(&drop1_layer), fc2_layer_cfg);
  let softmax_layer = SoftmaxLossLayer::new(Some(&fc2_layer), 10);

  let mut arch = LinearNetArch::new(data_layer, softmax_layer, vec![
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

  //let num_minibatches = train_data_source.len() / descent_cfg.minibatch_size;
  //let num_epoch_samples = descent_cfg.minibatch_size * num_minibatches;
  //let interval_size = 1000;

  //let mut train_data: Vec<_> = train_data_source.collect();
  //rng.shuffle(&mut train_data);

  let sgd = SgdOptimizer;
  sgd.train(&descent_cfg, &mut state, &mut arch, &mut *train_data_source, &mut *test_data_source, &ctx);
}
