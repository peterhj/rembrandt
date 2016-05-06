extern crate array_cuda;
extern crate rembrandt;

#[macro_use]
extern crate log;
extern crate env_logger;
extern crate rand;
extern crate threadpool;

use array_cuda::device::context::{DeviceContext};
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
  SgdOptConfig, StepSizeSchedule, Momentum, SyncOrder, OptSharedData, SgdOpt,
};
use rembrandt::templates::resnet::{
  build_resnet18_warp256x256,
  build_resnet34_warp256x256,
};
use rembrandt::templates::vgg::{
  build_vgg_a,
};
use rembrandt::worker::allreduce_dist::{
  MpiDistSyncAllreduceCommWorker,
};
use rembrandt::worker::elasticserver_dist::{
  MpiDistElasticServerCommWorker,
};
use rembrandt::worker::gossip_dist::{
  MpiDistAsyncPushGossipCommWorker,
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

  let num_workers = 1;
  let batch_size = 16;
  let minibatch_size = 16;
  /*let batch_size = 32;
  let minibatch_size = 32;*/
  /*let batch_size = 64;
  let minibatch_size = 64;*/
  //info!("num workers: {} batch size: {}", num_workers, batch_size);
  info!("batch size: {}", batch_size);

  let sgd_opt_cfg = SgdOptConfig{
    init_t:         None,
    //init_t:         Some(120_000),
    minibatch_size: minibatch_size,
    //step_size:      StepSizeSchedule::Constant{step_size: 0.1},
    step_size:      StepSizeSchedule::Anneal2{
      step0: 0.05,
      step1: 0.005,  step1_iters: 80_000,
      //step2: 0.0005, step2_iters: 120_000,
      step2: 0.0005, step2_iters: 160_000,
    },
    //momentum:       MomentumStyle::Zero,
    momentum:       Momentum::UpdateNesterov{mu: 0.9},
    //momentum:       Momentum::GradientNesterov{mu: 0.9},
    l2_reg_coef:    1.0e-4,
    //sync_order:     SyncOrder::StepThenSyncParams,  // XXX: for gossip.
    sync_order:     SyncOrder::SyncParamsThenStep,  // XXX: for elastic averaging.
    //sync_order:     SyncOrder::SyncGradsThenStep,   // XXX: for allreduce.
    //comm_interval:  1,
    comm_interval:  10,
    /*display_iters:  25,
    save_iters:     625,
    valid_iters:    625,*/
    display_iters:  50,
    checkpoint_iters:   1250,
    //save_iters:     1250,
    //valid_iters:    1250,
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
    checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-elastic_x16_run4_tau10"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-sync_x16_run1"),
    //checkpoint_dir:     PathBuf::from("models/imagenet_warp256x256-test"),
  };
  info!("sgd: {:?}", sgd_opt_cfg);

  let datum_cfg = SampleDatumConfig::Bytes3d;
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 1000,
  };

  //let worker_cfg = build_vgg_a_arch();
  //let worker_cfg = build_vgg_a_avgpool_arch();

  //let worker_cfg = build_resnet10_arch();
  let worker_cfg = build_resnet18_warp256x256();
  //let worker_cfg = build_resnet20_warp256x256();

  info!("operator: {:?}", worker_cfg);

  // FIXME(20160331)
  //let comm_worker_builder = DeviceSyncGossipCommWorkerBuilder::new(num_workers, 1, worker_cfg.params_len());
  //let worker_builder = SequentialOperatorWorkerBuilder::new(num_workers, batch_size, worker_cfg, OpCapability::Backward);
  let worker_builder = MpiDistSequentialOperatorWorkerBuilder::new(batch_size, worker_cfg.clone(), OpCapability::Backward);
  let opt_shared = Arc::new(OptSharedData::new(num_workers));

  /*let pool = ThreadPool::new(num_workers);
  let join_barrier = Arc::new(Barrier::new(num_workers + 1));
  for tid in 0 .. num_workers {
    //let comm_worker_builder = comm_worker_builder.clone();
    let worker_builder = worker_builder.clone();
    let join_barrier = join_barrier.clone();
    let worker_cfg = worker_cfg.clone();
    let sgd_opt_cfg = sgd_opt_cfg.clone();
    let opt_shared = opt_shared.clone();
    pool.execute(move || {
      /*let context = Rc::new(DeviceContext::new(tid));
      let comm_worker = Rc::new(RefCell::new(comm_worker_builder.into_worker(tid, context.clone())));
      let mut worker = worker_builder.into_worker(tid, context, comm_worker);*/

      let context = Rc::new(DeviceContext::new(0));
      /*let gossip_cfg = GossipConfig{
        num_rounds: 1,
        buf_size:   worker_cfg.params_len(),
      };
      let comm_worker = Rc::new(RefCell::new(MpiDistAsyncPushGossipCommWorker::new(gossip_cfg, context.clone())));*/
      let paramserver_cfg = ParameterServerConfig{
        com_interval: 1,
        buf_size:     worker_cfg.params_len(),
      };
      //let comm_worker = Rc::new(RefCell::new(MpiDistElasticServerCommWorker::new(paramserver_cfg, context.clone())));
      let comm_worker = Rc::new(RefCell::new(MpiDistSyncAllreduceCommWorker::new(paramserver_cfg, context.clone())));
      let mut worker = worker_builder.into_worker(context, comm_worker);

      let dataset_cfg = DatasetConfig::open(&PathBuf::from("examples/imagenet.data"));
      let mut train_data =
          /*//RandomEpisodeIterator::new(
          SampleIterator::new(
              dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "train"),
          );*/
          dataset_cfg.build_partition_iterator("train", worker.worker_rank(), worker.num_workers());
      let mut valid_data =
          /*SampleIterator::new(
              Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "valid")))
          );*/
          //dataset_cfg.build_iterator("valid");
          dataset_cfg.build_partition_iterator("valid", worker.worker_rank(), worker.num_workers());

      let mut sgd_opt = SgdOpt::new(sgd_opt_cfg, Some(worker.worker_rank()), opt_shared);
      sgd_opt.train(datum_cfg, label_cfg, &mut *train_data, &mut *valid_data, &mut worker);
      join_barrier.wait();
    });
  }
  join_barrier.wait();*/

  let context = Rc::new(DeviceContext::new(0));
  let gossip_cfg = GossipConfig{
    num_rounds: 1,
    buf_size:   worker_cfg.params_len(),
  };
  let paramserver_cfg = ParameterServerConfig{
    com_interval: 1,
    buf_size:     worker_cfg.params_len(),
  };
  //let comm_worker = Rc::new(RefCell::new(MpiDistAsyncPushGossipCommWorker::new(gossip_cfg, context.clone())));
  let comm_worker = Rc::new(RefCell::new(MpiDistElasticServerCommWorker::new(paramserver_cfg, context.clone())));
  //let comm_worker = Rc::new(RefCell::new(MpiDistSyncAllreduceCommWorker::new(paramserver_cfg, context.clone())));
  let mut worker = worker_builder.into_worker(context, comm_worker);

  let dataset_cfg = DatasetConfig::open(&PathBuf::from("examples/imagenet.data"));
  let mut train_data =
      dataset_cfg.build_partition_iterator("train", worker.worker_rank(), worker.num_workers());
  let mut valid_data =
      dataset_cfg.build_partition_iterator("valid", worker.worker_rank(), worker.num_workers());

  let mut sgd_opt = SgdOpt::new(sgd_opt_cfg, Some(worker.worker_rank()), opt_shared);
  sgd_opt.train(datum_cfg, label_cfg, &mut *train_data, &mut *valid_data, &mut worker);
}
