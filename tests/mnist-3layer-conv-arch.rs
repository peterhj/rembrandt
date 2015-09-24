extern crate async_cuda;
extern crate rembrandt;
extern crate rand;

use rembrandt::data::{DatasetConfiguration, DataSourceBuilder};
use rembrandt::layer::*;
use rembrandt::net::{NetArch, LinearNetArch};
use rembrandt::opt::{DescentConfig, DescentSchedule, AnnealingPolicy};

use async_cuda::context::{DeviceContext};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};

fn main() {
  let mut rng = thread_rng();
  let ctx = DeviceContext::new(0);

  let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("examples/mnist.data"));
  let train_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("train") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing data source!");
  };

  let mut train_data: Vec<_> = train_data_source.collect();
  rng.shuffle(&mut train_data);

  let descent_cfg = DescentConfig{
    minibatch_size: 50,
    init_step_size: 0.01,
    momentum:       0.0,
    anneal:         AnnealingPolicy::None,
  };
  let descent = DescentSchedule::new(descent_cfg);

  let num_minibatches = train_data.len() / descent_cfg.minibatch_size;
  let num_epoch_samples = descent_cfg.minibatch_size * num_minibatches;
  let interval_size = 1000;

  let data_layer_cfg = DataLayerConfig{
    width: 28, height: 28, channels: 1,
  };
  let conv1_layer_cfg = Conv2dLayerConfig{
    in_width: 28, in_height: 28, in_channels: 1,
    conv_size: 3, conv_stride: 1, conv_pad: 1,
    out_channels: 16,
    act_fun: ActivationFunction::Rect,
  };
  let fc1_layer_cfg = FullyConnLayerConfig{
    in_channels: 28 * 28 * 16, out_channels: 100,
    act_fun: ActivationFunction::Rect,
  };
  let fc2_layer_cfg = FullyConnLayerConfig{
    in_channels: 100, out_channels: 10,
    act_fun: ActivationFunction::Identity,
  };

  let data_layer = DataLayer::new(data_layer_cfg);
  let conv1_layer = Conv2dLayer::new(Some(&data_layer), conv1_layer_cfg, &ctx);
  let fc1_layer = FullyConnLayer::new(Some(&conv1_layer), fc1_layer_cfg);
  let fc2_layer = FullyConnLayer::new(Some(&fc1_layer), fc2_layer_cfg);
  let softmax_layer = SoftmaxLossLayer::new(Some(&fc2_layer), 10);

  let mut arch = LinearNetArch::new(data_layer, softmax_layer, vec![
      Box::new(conv1_layer),
      Box::new(fc1_layer),
      Box::new(fc2_layer),
  ]);
  arch.initialize_params(&ctx);

  let mut epoch: usize = 0;
  let mut t: usize = 0;
  loop {
    let mut epoch_correct = 0;
    let mut interval_correct = 0;
    arch.reset_gradients(&ctx);
    //rng.shuffle(&mut train_data);
    for (idx, &(ref datum, maybe_label)) in train_data.iter().enumerate() {
      arch.load_sample(datum, maybe_label, &ctx);
      arch.forward(&ctx);
      if arch.loss_layer.correct_guess(&ctx) {
        epoch_correct += 1;
        interval_correct += 1;
      }
      arch.backward(&descent_cfg, &ctx);
      let next_idx = idx + 1;
      if next_idx % interval_size == 0 {
        println!("DEBUG: interval: {}/{} accuracy: {:.3}",
            next_idx, num_epoch_samples,
            interval_correct as f32 / interval_size as f32);
        interval_correct = 0;
      }
      if next_idx >= num_epoch_samples {
        break;
      } else if next_idx % descent_cfg.minibatch_size == 0 {
        arch.descend(&descent, t, &ctx);
        arch.reset_gradients(&ctx);
        t += 1;
      }
    }
    println!("DEBUG: epoch: {} correct: {} accuracy: {:.3}", epoch, epoch_correct, epoch_correct as f32 / num_epoch_samples as f32);
    epoch += 1;
  }
}
