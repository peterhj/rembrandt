extern crate array_cuda;
extern crate rembrandt;

#[macro_use]
extern crate log;
extern crate env_logger;
extern crate rand;

use array_cuda::device::context::{DeviceContext};
use rembrandt::data::{
  AsyncQueueDataIter,
  CyclicSampleDataIter,
  RandomSampleDataIter,
};
use rembrandt::data::codec::{
  Array3dDataCodec,
  TurboJpegDataCodec,
};
use rembrandt::data::varraydb_data::{VarrayDbShard};
use rembrandt::data_new::{
  SampleDatumConfig, SampleLabelConfig,
  DatasetConfig,
  SampleIterator, RandomEpisodeIterator,
  PartitionDataSource,
};
use rembrandt::operator::{
  OpCapability,
};
use rembrandt::operator::graph::{GraphOperator};
use rembrandt::operator::seq::{SequentialOperator};
use rembrandt::opt::sgd::{
  SgdOptConfig,
  InitBehavior,
  StepSizeSchedule,
  StepSizeReference,
  Momentum,
  SyncOrder,
  CheckpointBehavior,
  //OptSharedData,
  //SgdOpt,
  SerialSgdOptConfig,
  SerialSgdOpt,
};
use rembrandt::opt::sgd::new::{
  //SgdOpt,
};
use rembrandt::templates::resnet_graph::{
  //build_resnet18pool_var224x224,
};
use rembrandt::templates::resnet_new::{
  //build_resnet18_var224x224,
  build_resnet18pool_var224x224,
};

use rand::{Rng, thread_rng};
use std::cell::{RefCell};
use std::rc::{Rc};
use std::path::{PathBuf};
use std::sync::{Arc, Barrier};

// XXX(20160423): These are Torch (nn/THNN) defaults.
//const BNORM_EMA_FACTOR: f64 = 0.1;
//const BNORM_EPSILON:    f64 = 1.0e-5;

fn main() {
  env_logger::init().unwrap();

  //let num_workers = 1;
  let batch_size = 32;
  let minibatch_size = 256;
  /*let batch_size = 32;
  let minibatch_size = 32;*/
  /*let batch_size = 64;
  let minibatch_size = 64;*/
  //info!("num workers: {} batch size: {}", num_workers, batch_size);
  info!("batch size: {}", batch_size);

  let sgd_opt_cfg = SerialSgdOptConfig{
    init:           InitBehavior::InitOrResume,
    minibatch_size: minibatch_size,
    step_size:      StepSizeSchedule::Anneal2{
      step0: 0.1,
      step1: 0.01,  step1_iters: 150_000,
      step2: 0.001, step2_iters: 300_000,
    },
    //momentum:       MomentumStyle::Zero,
    momentum:       Momentum::UpdateNesterov{mu: 0.9},
    //momentum:       Momentum::GradientNesterov{mu: 0.9},
    l2_reg_coef:    1.0e-4,
    display_iters:      1,
    checkpoint_iters:   625,
    save_iters:         625,
    valid_iters:        625,

    checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-resnet18pool_test_seq"),
  };
  info!("sgd: {:?}", sgd_opt_cfg);

  /*let datum_cfg = SampleDatumConfig::Bytes3d;
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 1000,
  };*/

  let shared_seed = [thread_rng().next_u64(), thread_rng().next_u64()];
  let context = Rc::new(DeviceContext::new(0));
  let operator_cfg = build_resnet18pool_var224x224();
  info!("operator: {:?}", operator_cfg);
  let mut operator = SequentialOperator::new(operator_cfg, batch_size, OpCapability::Backward, shared_seed, context);
  //let mut operator = GraphOperator::new(operator_cfg, batch_size, OpCapability::Backward, context);

  let mut train_data =
      AsyncQueueDataIter::new(
      CyclicSampleDataIter::new(
      //RandomSampleDataIter::new(
      VarrayDbShard::open_partition(
          &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_shuf/ilsvrc2012_maxscale480_shuf_train_data.varraydb"),
          &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_shuf/ilsvrc2012_maxscale480_shuf_train_labels.varraydb"),
          TurboJpegDataCodec::new(),
          //worker.worker_rank(), worker.num_workers(),
          0, 1,
      )));

  let mut valid_data =
      //AsyncQueueDataIter::new(
      CyclicSampleDataIter::new(
      VarrayDbShard::open_partition(
          &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_orig/ilsvrc2012_scale256_orig_valid_data.varraydb"),
          &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_orig/ilsvrc2012_scale256_orig_valid_labels.varraydb"),
          Array3dDataCodec::new(),
          //worker.worker_rank(), worker.num_workers(),
          0, 1,
      ));

  let mut sgd_opt = SerialSgdOpt::new(sgd_opt_cfg);
  sgd_opt.train(&mut train_data, Some(&mut valid_data), &mut operator);
}
