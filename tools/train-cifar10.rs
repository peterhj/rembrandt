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
  AffineBackend, Conv2dFwdBackend, Conv2dBwdBackend, PoolOperation,
  Data3dOperatorConfig,
  AffineOperatorConfig,
  Conv2dOperatorConfig,
  Pool2dOperatorConfig,
};
use rembrandt::operator::loss::{
  CategoricalLossConfig,
};
use rembrandt::operator::comm::{
  CommWorkerBuilder,
  DeviceSyncGossipCommWorkerBuilder,
};
use rembrandt::operator::worker::{
  OperatorWorkerBuilder,
  PipelineOperatorWorkerConfig,
  PipelineOperatorWorkerBuilder,
};
use rembrandt::opt::sgd::{
  SgdOptConfig, StepSizeSchedule, MomentumStyle, OptSharedData, SgdOpt,
};
use threadpool::{ThreadPool};

//use rand::{thread_rng};
use std::cell::{RefCell};
use std::rc::{Rc};
use std::path::{PathBuf};
use std::sync::{Arc, Barrier};

fn main() {
  env_logger::init().unwrap();

  let num_workers = 8;
  let batch_size = 128;
  info!("num workers: {} batch size: {}",
      num_workers, batch_size);

  let sgd_opt_cfg = SgdOptConfig{
    init_t:         None,
    minibatch_size: num_workers * batch_size,
    step_size:      StepSizeSchedule::Constant{step_size: 0.01},
    momentum:       MomentumStyle::Nesterov{momentum: 0.9},
    l2_reg_coef:    1.0e-4,
    display_iters:  100,
    valid_iters:    500,
    save_iters:     5000,
  };
  info!("sgd: {:?}", sgd_opt_cfg);

  let datum_cfg = SampleDatumConfig::Bytes3d;
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 10,
  };

  let data_op_cfg = Data3dOperatorConfig{
    dims:           (32, 32, 3),
    normalize:      true,
    preprocs:       vec![],
  };
  let conv1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (32, 32, 3),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
    out_channels:   64,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInit::Uniform{half_range: 0.05},
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool1_op_cfg = Pool2dOperatorConfig{
    in_dims:        (32, 32, 64),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    //pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let conv2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (16, 16, 64),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
    out_channels:   128,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInit::Uniform{half_range: 0.05},
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool2_op_cfg = Pool2dOperatorConfig{
    in_dims:        (16, 16, 128),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    //pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let aff1_op_cfg = AffineOperatorConfig{
    in_channels:    8192,
    out_channels:   256,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.05},
    backend:        AffineBackend::CublasGemm,
  };
  let aff2_op_cfg = AffineOperatorConfig{
    in_channels:    256,
    out_channels:   10,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInit::Uniform{half_range: 0.05},
    backend:        AffineBackend::CublasGemm,
  };
  let loss_cfg = CategoricalLossConfig{
    num_categories: 10,
  };

  let mut worker_cfg = PipelineOperatorWorkerConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    .conv2d(conv1_op_cfg)
    .pool2d(pool1_op_cfg)
    .conv2d(conv2_op_cfg)
    .pool2d(pool2_op_cfg)
    .affine(aff1_op_cfg)
    .affine(aff2_op_cfg)
    .softmax_kl_loss(loss_cfg);

  // FIXME(20160331)
  let comm_worker_builder = DeviceSyncGossipCommWorkerBuilder::new(num_workers, 1, worker_cfg.params_len());
  let worker_builder = PipelineOperatorWorkerBuilder::new(num_workers, batch_size, worker_cfg, OpCapability::Backward);
  let pool = ThreadPool::new(num_workers);
  let join_barrier = Arc::new(Barrier::new(num_workers + 1));
  let opt_shared = Arc::new(OptSharedData::new(num_workers));
  for tid in 0 .. num_workers {
    let comm_worker_builder = comm_worker_builder.clone();
    let worker_builder = worker_builder.clone();
    let join_barrier = join_barrier.clone();
    let opt_shared = opt_shared.clone();
    pool.execute(move || {
      let context = Rc::new(DeviceContext::new(tid));
      let comm_worker = Rc::new(RefCell::new(comm_worker_builder.into_worker(tid)));
      let mut worker = worker_builder.into_worker(tid, context, comm_worker);

      let dataset_cfg = DatasetConfig::open(&PathBuf::from("examples/cifar10.data"));
      let mut train_data =
          RandomEpisodeIterator::new(
            dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "train"),
          );
      let mut valid_data =
          SampleIterator::new(
            Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "valid")))
          );

      let sgd_opt = SgdOpt::new(opt_shared);
      sgd_opt.train(sgd_opt_cfg, datum_cfg, label_cfg, &mut train_data, &mut valid_data, &mut worker);
      join_barrier.wait();
    });
  }
  join_barrier.wait();
}
