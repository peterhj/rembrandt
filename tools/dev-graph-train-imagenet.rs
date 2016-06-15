extern crate array_cuda;
extern crate rembrandt;

#[macro_use]
extern crate log;
extern crate crossbeam;
extern crate env_logger;
extern crate rand;

use array_cuda::device::context::{DeviceContext};
use crossbeam::{scope};
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
use rembrandt::opt::sgd::parallel::{
  ParallelSgdOptConfig,
  ParallelSgdOpt,
};
use rembrandt::opt::sgd::parallel::dev_allreduce::{
  DeviceAllreduceSgdOptWorkerBuilder,
};
use rembrandt::templates::resnet_graph::{
  build_resnet18pool_var224x224,
};
/*use rembrandt::templates::resnet_new::{
  build_resnet18_var224x224,
  build_resnet18pool_var224x224,
};*/

//use rand::{thread_rng};
use std::cell::{RefCell};
use std::rc::{Rc};
use std::path::{PathBuf};
use std::sync::{Arc, Barrier};

// XXX(20160423): These are Torch (nn/THNN) defaults.
//const BNORM_EMA_FACTOR: f64 = 0.1;
//const BNORM_EPSILON:    f64 = 1.0e-5;

fn main() {
  env_logger::init().unwrap();

  let num_local_workers = 4;
  let batch_size = 32;
  let minibatch_size = 64;
  //info!("batch size: {}", batch_size);
  info!("num workers: {} batch size: {}", num_local_workers, batch_size);

  let sgd_opt_cfg = ParallelSgdOptConfig{
    init:           InitBehavior::InitOrResume,
    minibatch_size: minibatch_size,
    step_size:      StepSizeSchedule::Anneal2{
      step0: 0.1,
      step1: 0.01,  step1_iters: 150_000,
      step2: 0.001, step2_iters: 300_000,
    },
    //momentum:       Momentum::Zero,
    //momentum:       Momentum::Update{mu: 0.9},
    momentum:       Momentum::UpdateNesterov{mu: 0.9},
    l2_reg_coef:    1.0e-4,
    display_iters:      25,
    checkpoint_iters:   250,
    save_iters:         250,
    valid_iters:        250,

    checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-resnet18pool_dev_x4_test"),
  };
  info!("sgd: {:?}", sgd_opt_cfg);

  /*let datum_cfg = SampleDatumConfig::Bytes3d;
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 1000,
  };*/

  scope(|scope| {
    let builder = DeviceAllreduceSgdOptWorkerBuilder::new(num_local_workers);
    let mut guards = vec![];
    for tid in 0 .. num_local_workers {
      let sgd_opt_cfg = sgd_opt_cfg.clone();
      let builder = builder.clone();
      let guard = scope.spawn(move || {
        let context = Rc::new(DeviceContext::new(tid));
        let operator_cfg = build_resnet18pool_var224x224();
        if tid == 0 {
          info!("operator: {:?}", operator_cfg);
        }
        let operator = Box::new(GraphOperator::new(operator_cfg, batch_size, OpCapability::Backward, context.clone()));
        let mut worker = builder.into_worker(tid, context, operator);

        let mut train_data =
            AsyncQueueDataIter::new(
            //CyclicSampleDataIter::new(
            RandomSampleDataIter::new(
            VarrayDbShard::open_partition(
                &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_shuf/ilsvrc2012_maxscale480_shuf_train_data.varraydb"),
                &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_shuf/ilsvrc2012_maxscale480_shuf_train_labels.varraydb"),
                TurboJpegDataCodec::new(),
                //worker.worker_rank(), worker.num_workers(),
                //tid, num_local_workers,
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
                tid, num_local_workers,
            ));

        let mut sgd_opt = ParallelSgdOpt::new(sgd_opt_cfg);
        sgd_opt.train(&mut train_data, Some(&mut valid_data), &mut worker);
      });
      guards.push(guard);
    }
    for guard in guards {
      guard.join();
    }
  });
}
