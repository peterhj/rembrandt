extern crate array_cuda;
extern crate rembrandt;

#[macro_use]
extern crate log;
extern crate env_logger;
extern crate rand;
extern crate threadpool;

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
  ActivationFunction, ParamsInit,
  Data3dPreproc, AffineBackend, Conv2dFwdBackend, Conv2dBwdBackend, PoolOperation,
  Data3dOperatorConfig,
  AffineOperatorConfig,
  Conv2dOperatorConfig,
  Pool2dOperatorConfig,
  DropoutOperatorConfig,
};
use rembrandt::operator::loss::{
  CategoricalLossConfig,
};
use rembrandt::operator::comm::{
  ParameterServerConfig,
  GossipConfig,
  DeviceSyncGossipCommWorkerBuilder,
};
use rembrandt::operator::conv::{
  BNormMovingAverage,
  BNormConv2dOperatorConfig,
  StackResConv2dOperatorConfig,
  ProjStackResConv2dOperatorConfig,
};
use rembrandt::operator::worker::{
  OperatorWorkerBuilder,
  OperatorWorker,
  SequentialOperatorConfig,
  SequentialOperatorWorkerBuilder,
};
use rembrandt::opt::sgd::{
  SgdOptConfig,
  InitBehavior,
  StepSizeSchedule,
  StepSizeReference,
  Momentum,
  SyncOrder,
  CheckpointBehavior,
  OptSharedData,
  //SgdOpt,
};
use rembrandt::opt::sgd::new::{
  SgdOpt,
};
use rembrandt::templates::resnet_new::{
  build_resnet18_var224x224,
};
/*use rembrandt::templates::resnet::{
  build_resnet18_warp256x256,
  build_resnet34_warp256x256,
};
use rembrandt::templates::vgg::{
  build_vgg_a,
};*/
use rembrandt::worker::allreduce_dist::{
  MpiDistSyncAllreduceCommWorker,
};
use rembrandt::worker::elasticserver_dist::{
  MpiDistElasticServerCommWorker,
};
use rembrandt::worker::gossip_pull_dist_rdma::{
  MpiDistAsyncPullGossipCommWorker,
};
use rembrandt::worker::gossip_dist::{
  MpiDistAsyncPushGossipCommWorker,
  ExperimentConfig,
  MpiDistSequentialOperatorWorkerBuilder,
  MpiDistSequentialOperatorWorker,
};
use threadpool::{ThreadPool};

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

  //let num_workers = 1;
  let batch_size = 32;
  let minibatch_size = 128;
  /*let batch_size = 32;
  let minibatch_size = 32;*/
  /*let batch_size = 64;
  let minibatch_size = 64;*/
  //info!("num workers: {} batch size: {}", num_workers, batch_size);
  info!("batch size: {}", batch_size);

  let sgd_opt_cfg = SgdOptConfig{
    init:           InitBehavior::InitOrResume,
    //init_t:         None,
    //init_t:         Some(150_000),
    minibatch_size: minibatch_size,
    //step_size:      StepSizeSchedule::Constant{step_size: 0.1},
    step_size:      StepSizeSchedule::Anneal2{
      /*step0: 0.1,
      step1: 0.01,  step1_iters: 150_000,
      step2: 0.001, step2_iters: 300_000,*/
      step0: 0.1,
      step1: 0.01,  step1_iters: 18_750,
      step2: 0.001, step2_iters: 37_500,
    },
    step_ref:       StepSizeReference::Checkpoint,
    //momentum:       MomentumStyle::Zero,
    momentum:       Momentum::UpdateNesterov{mu: 0.9},
    //momentum:       Momentum::GradientNesterov{mu: 0.9},
    l2_reg_coef:    1.0e-4,
    //sync_order:     SyncOrder::StepThenSyncParams,  // XXX: for gossip.
    //sync_order:     SyncOrder::SyncParamsThenStep,  // XXX: for elastic averaging.
    sync_order:     SyncOrder::SyncGradsThenStep,   // XXX: for allreduce.
    checkpoint:     CheckpointBehavior::Discard,
    comm_interval:  1,
    display_iters:      25,
    /*checkpoint_iters:   50,
    save_iters:         50,
    valid_iters:        50,*/
    checkpoint_iters:   625,
    save_iters:         625,
    valid_iters:        625,
    /*display_iters:      50,
    checkpoint_iters:   1250,
    save_iters:         1250,
    valid_iters:        1250,*/

    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-async_push_gossip_x8-run1-step2_120k"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-async_push_gossip_x8-run2"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-async_push_gossip_x8-run4"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-sync_x8-run0"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-async_elastic_server_x8-run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-sync_x8-run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-sync_x8-run2"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-gossip_x16-run2"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-gossip_x16-run3"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-elastic_x16_run2"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-elastic_x16_run4_tau10"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-sync_x16_run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-test"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-pushgossip_x8_run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-pushgossip_x8_run2_lr0.1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-pushgossip_x8_test"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-pushgossip_x16_run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-pushgossip_x16_run2_minibatch128"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-pullgossip_x8_run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-sync_x8_run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-sync_x8_run2"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-sync_x16_run1"),
    checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-sync_x16_run2_minibatch128"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-elastic_x8_run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-elastic_x8_run1_pt2"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-elastic_x8_run2"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-elastic_x16_run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-elastic_x16_run2_minibatch128"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-test"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-x1_run1"),
  };
  info!("sgd: {:?}", sgd_opt_cfg);

  let datum_cfg = SampleDatumConfig::Bytes3d;
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 1000,
  };

  //let worker_cfg = build_vgg_a_arch();
  //let worker_cfg = build_vgg_a_avgpool_arch();

  //let worker_cfg = build_resnet18_warp256x256();
  let worker_cfg = build_resnet18_var224x224();

  info!("operator: {:?}", worker_cfg);

  let exp_cfg = ExperimentConfig{
    straggler_ranks:        vec![],
    straggler_max_delay_ms: 300,
  };

  // FIXME(20160331)
  //let comm_worker_builder = DeviceSyncGossipCommWorkerBuilder::new(num_workers, 1, worker_cfg.params_len());
  //let worker_builder = SequentialOperatorWorkerBuilder::new(num_workers, batch_size, worker_cfg, OpCapability::Backward);
  let worker_builder = MpiDistSequentialOperatorWorkerBuilder::new(batch_size, worker_cfg.clone(), OpCapability::Backward, exp_cfg);
  let opt_shared = Arc::new(OptSharedData::new(1));

  let context = Rc::new(DeviceContext::new(0));
  let gossip_cfg = GossipConfig{
    num_rounds: 1,
    buf_size:   worker_cfg.params_len(),
  };
  let paramserver_cfg = ParameterServerConfig{
    com_interval: 1,
    buf_size:     worker_cfg.params_len(),
  };

  let msg_len = 32 * 1024;
  //let comm_worker = Rc::new(RefCell::new(MpiDistAsyncPushGossipCommWorker::new(gossip_cfg, context.clone())));
  //let comm_worker = Rc::new(RefCell::new(MpiDistAsyncPullGossipCommWorker::new(msg_len, gossip_cfg, context.clone())));
  //let comm_worker = Rc::new(RefCell::new(MpiDistElasticServerCommWorker::new(paramserver_cfg, context.clone())));
  let comm_worker = Rc::new(RefCell::new(MpiDistSyncAllreduceCommWorker::new(paramserver_cfg, context.clone())));
  let mut worker = worker_builder.into_worker(context, comm_worker);

  /*let dataset_cfg = DatasetConfig::open(&PathBuf::from("examples/imagenet.data"));
  let mut train_data =
      dataset_cfg.build_partition_iterator("train", worker.worker_rank(), worker.num_workers());
  let mut valid_data =
      dataset_cfg.build_partition_iterator("valid", worker.worker_rank(), worker.num_workers());*/

  let mut train_data =
      AsyncQueueDataIter::new(
      RandomSampleDataIter::new(
      VarrayDbShard::open_partition(
          &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_shuf/ilsvrc2012_maxscale480_shuf_train_data.varraydb"),
          &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_shuf/ilsvrc2012_maxscale480_shuf_train_labels.varraydb"),
          TurboJpegDataCodec::new(),
          worker.worker_rank(), worker.num_workers(),
      )));

  let mut valid_data =
      //AsyncQueueDataIter::new(
      CyclicSampleDataIter::new(
      VarrayDbShard::open_partition(
          &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_orig/ilsvrc2012_scale256_orig_valid_data.varraydb"),
          &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_orig/ilsvrc2012_scale256_orig_valid_labels.varraydb"),
          Array3dDataCodec::new(),
          worker.worker_rank(), worker.num_workers(),
      ));


  let mut sgd_opt = SgdOpt::new(sgd_opt_cfg, Some(worker.worker_rank()), opt_shared);
  sgd_opt.train(datum_cfg, label_cfg, &mut train_data, Some(&mut valid_data), &mut worker);
}
