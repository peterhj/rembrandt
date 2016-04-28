extern crate array_cuda;
extern crate rembrandt;

#[macro_use]
extern crate log;
extern crate env_logger;
extern crate rand;
extern crate threadpool;

use array_cuda::device::context::{DeviceContext};
use rembrandt::data::{
  DataIter,
  RandomSampleDataIter,
  CyclicSampleDataIter,
  AugmentDataIter,
  AsyncQueueDataIter,
  SourceDataIter,
  AsyncPoolDataIter,
};
use rembrandt::data::augment::{
  AugmentPreproc,
  RandomScalePreproc,
  RandomCropPreproc,
  CenterCropPreproc,
};
use rembrandt::data::codec::{
  DataCodec, PngDataCodec,
};
use rembrandt::data::varraydb_data::{
  VarrayDbShard,
};
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
  SgdOptConfig, StepSizeSchedule, Momentum, SyncOrder, OptSharedData,
};
use rembrandt::opt::sgd::async::{
  AsyncSgdOpt,
};
use rembrandt::templates::resnet::{
  build_resnet18_warp256x256,
  build_resnet34_warp256x256,
  build_resnet18_224x224,
};
use rembrandt::templates::vgg::{
  build_vgg_a,
};
use rembrandt::worker::gossip_dist::{
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
  let batch_size = 32;
  let minibatch_size = 32;
  info!("num workers: {} batch size: {}",
      num_workers, batch_size);

  let sgd_opt_cfg = SgdOptConfig{
    init_t:         None,
    minibatch_size: minibatch_size,
    //step_size:      StepSizeSchedule::Constant{step_size: 0.1},
    step_size:      StepSizeSchedule::Anneal2{
      /*step0: 0.05,
      step1: 0.005,  step1_iters: 80_000,
      step2: 0.0005, step2_iters: 120_000,*/
      step0: 0.1,
      step1: 0.01,  step1_iters: 150_000,
      step2: 0.001, step2_iters: 300_000,
    },
    //momentum:       MomentumStyle::Zero,
    momentum:       Momentum::UpdateNesterov{mu: 0.9},
    //momentum:       Momentum::GradientNesterov{mu: 0.9},
    l2_reg_coef:    1.0e-4,
    sync_order:     SyncOrder::StepThenSyncParams,
    /*display_iters:  25,
    save_iters:     625,
    valid_iters:    625,*/
    display_iters:  50,
    save_iters:     1250,
    valid_iters:    1250,
    //checkpoint_dir: PathBuf::from("models/imagenet_warp256x256-async_push_gossip_x8-run1-step2_120k"),
    //checkpoint_dir: PathBuf::from("models/imagenet_warp256x256-async_push_gossip_x8-run2"),
    //checkpoint_dir: PathBuf::from("models/imagenet_warp256x256-async_push_gossip_x8-run3"),
    //checkpoint_dir: PathBuf::from("models/imagenet_warp256x256-sync_x8-run0"),
    checkpoint_dir: PathBuf::from("models/imagenet_scale480-async_push_gossip_x8-test"),
  };
  info!("sgd: {:?}", sgd_opt_cfg);

  let datum_cfg = SampleDatumConfig::Bytes3d;
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 1000,
  };

  //let worker_cfg = build_vgg_a_arch();
  //let worker_cfg = build_vgg_a_avgpool_arch();

  //let worker_cfg = build_resnet18_warp256x256();
  let worker_cfg = build_resnet18_224x224();
  //let worker_cfg = build_resnet20_warp256x256();

  info!("operator: {:?}", worker_cfg);

  // FIXME(20160331)
  //let comm_worker_builder = DeviceSyncGossipCommWorkerBuilder::new(num_workers, 1, worker_cfg.params_len());
  //let worker_builder = SequentialOperatorWorkerBuilder::new(num_workers, batch_size, worker_cfg, OpCapability::Backward);
  let worker_builder = MpiDistSequentialOperatorWorkerBuilder::new(batch_size, worker_cfg, OpCapability::Backward);
  let pool = ThreadPool::new(num_workers);
  let join_barrier = Arc::new(Barrier::new(num_workers + 1));
  //let opt_shared = Arc::new(OptSharedData::new(num_workers));
  for tid in 0 .. num_workers {
    //let comm_worker_builder = comm_worker_builder.clone();
    let worker_builder = worker_builder.clone();
    let join_barrier = join_barrier.clone();
    let sgd_opt_cfg = sgd_opt_cfg.clone();
    //let opt_shared = opt_shared.clone();
    pool.execute(move || {
      /*let context = Rc::new(DeviceContext::new(tid));
      let comm_worker = Rc::new(RefCell::new(comm_worker_builder.into_worker(tid, context.clone())));
      let mut worker = worker_builder.into_worker(tid, context, comm_worker);*/

      let context = Rc::new(DeviceContext::new(0));
      let mut worker = worker_builder.into_worker(context);

      fn wrap_source(source: SourceDataIter) -> AugmentDataIter<RandomCropPreproc, AugmentDataIter<RandomScalePreproc, SourceDataIter>> {
        AugmentDataIter::new(RandomCropPreproc{
          crop_width:   224,
          crop_height:  224,
        },
        AugmentDataIter::new(RandomScalePreproc{
          min_side_lo:  256,
          min_side_hi:  480,
        },
        source,
        ))
      }

      let mut train_data =
          /*AsyncQueueDataIter::new(
          AugmentDataIter::new(RandomCropPreproc{
            crop_width:   224,
            crop_height:  224,
          },
          AugmentDataIter::new(RandomScalePreproc{
            min_side_lo:  256,
            min_side_hi:  480,
          },
          //CyclicSampleDataIter::new(
          RandomSampleDataIter::new(
          VarrayDbShard::open_partition(
              &PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480_shuf/ilsvrc2012_scale480_shuf_train_data.varraydb"),
              &PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480_shuf/ilsvrc2012_scale480_shuf_train_labels.varraydb"),
              PngDataCodec,
              worker.worker_rank(), worker.num_workers(),
          )))));*/
          AsyncPoolDataIter::new(8, wrap_source,
          //CyclicSampleDataIter::new(
          RandomSampleDataIter::new(
          VarrayDbShard::open_partition(
              &PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480_shuf/ilsvrc2012_scale480_shuf_train_data.varraydb"),
              &PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480_shuf/ilsvrc2012_scale480_shuf_train_labels.varraydb"),
              PngDataCodec,
              worker.worker_rank(), worker.num_workers(),
          )));

      /*let mut valid_data =
          AugmentDataIter::new(CenterCropPreproc{
            crop_width:   224,
            crop_height:  224,
          },
          AugmentDataIter::new(RandomScalePreproc{
            min_side_lo:  368,
            min_side_hi:  368,
          },
          CyclicSampleDataIter::new(
          VarrayDbShard::open_partition(
              &PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480_shuf/ilsvrc2012_scale480_shuf_valid_data.varraydb"),
              &PathBuf::from("/rscratch/phj/data/ilsvrc2012_scale480_shuf/ilsvrc2012_scale480_shuf_valid_labels.varraydb"),
              PngDataCodec,
              worker.worker_rank(), worker.num_workers(),
          ))));*/

      let mut sgd_opt = AsyncSgdOpt::new(sgd_opt_cfg, Some(worker.worker_rank()));
      sgd_opt.train(datum_cfg, label_cfg, &mut train_data, None, &mut worker);
      //sgd_opt.train(datum_cfg, label_cfg, &mut train_data, Some(&mut valid_data), &mut worker);
      join_barrier.wait();
    });
  }
  join_barrier.wait();
}
