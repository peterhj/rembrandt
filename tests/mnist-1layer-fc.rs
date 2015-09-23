extern crate async_cuda;
extern crate rembrandt;
extern crate rand;

use rembrandt::data::{DatasetConfiguration, DataSourceBuilder};
use rembrandt::layer::*;
use rembrandt::opt::{DescentConfig};

use async_cuda::context::{DeviceContext};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};

fn main() {
  let mut rng = thread_rng();
  let ctx = DeviceContext::new(0);

  let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("../data/mnist.data"));
  let train_data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("train") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing data source!");
  };

  let mut train_data: Vec<_> = train_data_source.collect();
  rng.shuffle(&mut train_data);

  let descent = DescentConfig{
    minibatch_size: 50,
    step_size: 0.001,
  };

  let num_minibatches = train_data.len() / descent.minibatch_size;
  let num_epoch_samples = descent.minibatch_size * num_minibatches;
  let interval_size = 1000;

  let (width, height) = (28, 28);

  let ip1_layer_cfg = FullyConnLayerConfig{in_channels: width * height, out_channels: 10, act_fun: ActivationFunction::Identity};

  let mut data_layer = DataLayer::new(width * height);
  let mut ip1_layer = FullyConnLayer::new(Some(&data_layer), ip1_layer_cfg);
  let mut softmax_layer = SoftmaxLossLayer::new(Some(&ip1_layer), 10);

  ip1_layer.initialize_params(&ctx);

  let mut epoch: usize = 0;
  let mut t: usize = 0;
  loop {
    let mut epoch_correct = 0;
    let mut interval_correct = 0;

    ip1_layer.reset_gradients(&ctx);

    rng.shuffle(&mut train_data);
    for (idx, &(ref datum, maybe_label)) in train_data.iter().enumerate() {
      data_layer.load_sample(datum, &ctx);
      softmax_layer.load_sample(maybe_label, &ctx);

      ip1_layer.forward(&ctx);
      softmax_layer.forward(&ctx);

      if softmax_layer.correct_guess(&ctx) {
        epoch_correct += 1;
        interval_correct += 1;
      }

      softmax_layer.backward(&descent, &ctx);
      ip1_layer.backward(&descent, &ctx);

      t += 1;

      let next_idx = idx + 1;
      if next_idx % interval_size == 0 {
        println!("DEBUG: interval ({}/{}): accuracy {:.3}",
            next_idx, num_epoch_samples,
            interval_correct as f32 / interval_size as f32);
        interval_correct = 0;
      }
      if next_idx >= num_epoch_samples {
        break;
      } else if next_idx % descent.minibatch_size == 0 {
        ip1_layer.descend(&descent, &ctx);
        ip1_layer.reset_gradients(&ctx);
      }
    }

    println!("DEBUG: epoch: {} correct: {} accuracy {:.3}", epoch, epoch_correct, epoch_correct as f32 / num_epoch_samples as f32);
    epoch += 1;
  }
}
