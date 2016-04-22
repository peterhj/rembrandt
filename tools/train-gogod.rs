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
  DropoutOperatorConfig,
};
use rembrandt::operator::loss::{
  CategoricalLossConfig,
};
use rembrandt::operator::comm::{
  CommWorkerBuilder,
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
  PipelineOperatorConfig,
  PipelineOperatorWorkerBuilder,
};
use rembrandt::opt::sgd::{
  SgdOptConfig, StepSizeSchedule, MomentumStyle, OptSharedData, SyncSgdOpt,
};
use threadpool::{ThreadPool};

//use rand::{thread_rng};
use std::cell::{RefCell};
use std::rc::{Rc};
use std::path::{PathBuf};
use std::sync::{Arc, Barrier};

fn build_vgg_a_arch() -> PipelineOperatorConfig {
  let data_op_cfg = Data3dOperatorConfig{
    in_dims:        (19, 19, 32),
    normalize:      true,
    preprocs:       vec![],
  };
  let conv1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (19, 19, 32),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   128,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let bnorm_conv1_op_cfg = BNormConv2dOperatorConfig{
    in_dims:        (19, 19, 32),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   128,
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: 0.01},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let hidden_conv_op_cfg = Conv2dOperatorConfig{
    in_dims:        (19, 19, 128),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   128,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let hidden_res_conv_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (19, 19, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: 0.01},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let final_conv_op_cfg = Conv2dOperatorConfig{
    in_dims:        (19, 19, 128),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   1,
    act_func:       ActivationFunction::Identity,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  /*let final_bnorm_conv_op_cfg = BNormConv2dOperatorConfig{
    in_dims:        (19, 19, 128),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   1,
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: 0.01},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Identity,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };*/
  let loss_cfg = CategoricalLossConfig{
    num_categories: 361,
  };

  let mut worker_cfg = PipelineOperatorConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    //.conv2d(conv1_op_cfg)
    .bnorm_conv2d(bnorm_conv1_op_cfg)
    /*.conv2d(hidden_conv_op_cfg)
    .conv2d(hidden_conv_op_cfg)
    .conv2d(hidden_conv_op_cfg)
    .conv2d(hidden_conv_op_cfg)
    .conv2d(hidden_conv_op_cfg)
    .conv2d(hidden_conv_op_cfg)
    .conv2d(hidden_conv_op_cfg)
    .conv2d(hidden_conv_op_cfg)
    .conv2d(hidden_conv_op_cfg)
    .conv2d(hidden_conv_op_cfg)*/
    .stack_res_conv2d(hidden_res_conv_op_cfg)
    .stack_res_conv2d(hidden_res_conv_op_cfg)
    .stack_res_conv2d(hidden_res_conv_op_cfg)
    .stack_res_conv2d(hidden_res_conv_op_cfg)
    .stack_res_conv2d(hidden_res_conv_op_cfg)
    .conv2d(final_conv_op_cfg)
    //.bnorm_conv2d(final_bnorm_conv_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}

fn main() {
  env_logger::init().unwrap();

  let num_workers = 1;
  //let num_workers = 4;
  let batch_size = 256;
  info!("num workers: {} batch size: {}",
      num_workers, batch_size);

  let sgd_opt_cfg = SgdOptConfig{
    init_t:         None,
    minibatch_size: batch_size,
    step_size:      StepSizeSchedule::Constant{step_size: 0.1},
    momentum:       MomentumStyle::Zero,
    //momentum:       MomentumStyle::Nesterov{momentum: 0.9},
    l2_reg_coef:    0.0,
    display_iters:  50,
    valid_iters:    1500,
    save_iters:     1500,
  };
  info!("sgd: {:?}", sgd_opt_cfg);

  //let datum_cfg = SampleDatumConfig::Bytes3d;
  let datum_cfg = SampleDatumConfig::BitsThenBytes3d{scale: 255};
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 361,
  };

  let worker_cfg = build_vgg_a_arch();

  info!("operator: {:?}", worker_cfg);

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

      //let dataset_cfg = DatasetConfig::open(&PathBuf::from("examples/imagenet.data"));
      let dataset_cfg = DatasetConfig::open(&PathBuf::from("examples/gogodb_w2015-preproc-alphav3m_19x19x32.data"));
      let mut train_data =
          RandomEpisodeIterator::new(
              dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "train"),
          );
      let mut valid_data =
          SampleIterator::new(
              Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "valid")))
          );

      let sgd_opt = SyncSgdOpt::new(opt_shared);
      sgd_opt.train(sgd_opt_cfg, datum_cfg, label_cfg, &mut train_data, &mut valid_data, &mut worker);
      join_barrier.wait();
    });
  }
  join_barrier.wait();
}
