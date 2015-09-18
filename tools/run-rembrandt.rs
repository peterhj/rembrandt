extern crate async_cuda;
extern crate rembrandt;
extern crate rand;

use rembrandt::data::{DatasetConfiguration, DataSourceBuilder};
use rembrandt::layer::{Layer, DataLayer, InnerProdLayer, SoftmaxLossLayer};

use async_cuda::context::{DeviceContext};

use rand::{Rng, thread_rng};
use std::path::{PathBuf};

fn main() {
  let mut rng = thread_rng();
  let ctx = DeviceContext::new(0);

  let dataset_cfg = DatasetConfiguration::open(&PathBuf::from("../data/mnist.data"));
  let data_source = if let Some(&(ref name, ref cfg)) = dataset_cfg.datasets.get("train") {
    DataSourceBuilder::build(name, cfg.clone())
  } else {
    panic!("missing data source!");
  };

  let mut train_data: Vec<_> = data_source.collect();
  rng.shuffle(&mut train_data);

  //let minibatch_size: usize = 1;
  let minibatch_size: usize = 50;
  let learning_rate: f32 = 0.01;

  let num_samples = train_data.len();
  let num_minibatches = num_samples / minibatch_size;
  let max_epoch_idx = minibatch_size * num_minibatches;

  //let (width, height) = (data_source.image_width, data_source.image_height);
  let (width, height) = (28, 28);

  let mut data_layer = DataLayer::new(width * height);
  let mut ip1_layer = InnerProdLayer::new(Some(&data_layer), width * height, 10);
  let mut softmax_layer = SoftmaxLossLayer::new(Some(&ip1_layer), 10);

  ip1_layer.initialize_params(&ctx); // TODO

  let mut epoch: usize = 0;
  loop {
    let mut epoch_correct = 0;
    //let mut batch_correct = 0;
    ip1_layer.reset_gradients(&ctx);

    for (idx, &(ref datum, maybe_label)) in train_data.iter().enumerate() {
      data_layer.load_sample(datum, &ctx);
      softmax_layer.load_sample(maybe_label, &ctx);

      ip1_layer.forward(&ctx);
      softmax_layer.forward(&ctx);

      if softmax_layer.correct_guess(&ctx) {
        epoch_correct += 1;
      }

      softmax_layer.backward(&ctx);
      ip1_layer.backward(&ctx);

      let next_idx = idx + 1;
      if next_idx >= max_epoch_idx {
        break;
      } else if minibatch_size == 1 || next_idx % minibatch_size == 0 {
        ip1_layer.update_params(learning_rate, &ctx);
        ip1_layer.reset_gradients(&ctx);
      }
    }

    println!("DEBUG: epoch: {} correct: {}", epoch, epoch_correct);
    epoch += 1;
  }
}
