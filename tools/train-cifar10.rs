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

fn build_krizh26_arch() -> PipelineOperatorConfig {
  let data_op_cfg = Data3dOperatorConfig{
    in_dims:        (32, 32, 3),
    normalize:      true,
    preprocs:       vec![],
  };
  let conv1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (32, 32, 3),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
    out_channels:   32,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingRectFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool1_op_cfg = Pool2dOperatorConfig{
    in_dims:        (32, 32, 32),
    pool_size:      3,
    pool_stride:    2,
    pool_pad:       1,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (16, 16, 32),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
    out_channels:   32,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingRectFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool2_op_cfg = Pool2dOperatorConfig{
    in_dims:        (16, 16, 32),
    pool_size:      3,
    pool_stride:    2,
    pool_pad:       1,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let conv3_op_cfg = Conv2dOperatorConfig{
    in_dims:        (8, 8, 32),
    conv_size:      5,
    conv_stride:    1,
    conv_pad:       2,
    out_channels:   64,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingRectFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool3_op_cfg = Pool2dOperatorConfig{
    in_dims:        (8, 8, 64),
    pool_size:      3,
    pool_stride:    2,
    pool_pad:       1,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let aff1_op_cfg = AffineOperatorConfig{
    in_channels:    1024,
    out_channels:   64,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.05},
    //init_weights:   ParamsInit::KaimingRectFwd,
    backend:        AffineBackend::CublasGemm,
  };
  let drop_op_cfg = DropoutOperatorConfig{
    channels:       64,
    drop_ratio:     0.5,
  };
  let aff2_op_cfg = AffineOperatorConfig{
    in_channels:    64,
    out_channels:   10,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInit::Uniform{half_range: 0.05},
    //init_weights:   ParamsInit::KaimingRectFwd,
    backend:        AffineBackend::CublasGemm,
  };
  let loss_cfg = CategoricalLossConfig{
    num_categories: 10,
  };

  let mut worker_cfg = PipelineOperatorConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    .conv2d(conv1_op_cfg)
    .pool2d(pool1_op_cfg)
    .conv2d(conv2_op_cfg)
    .pool2d(pool2_op_cfg)
    .conv2d(conv3_op_cfg)
    .pool2d(pool3_op_cfg)
    .affine(aff1_op_cfg)
    //.dropout(drop_op_cfg)
    .affine(aff2_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}

fn build_allconv_arch() -> PipelineOperatorConfig {
  let data_op_cfg = Data3dOperatorConfig{
    in_dims:        (32, 32, 3),
    normalize:      true,
    preprocs:       vec![],
  };
  let conv1_1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (32, 32, 3),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   96,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.10},
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv1_2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (32, 32, 32),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   96,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.10},
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool1_op_cfg = Pool2dOperatorConfig{
    in_dims:        (32, 32, 96),
    pool_size:      3,
    pool_stride:    2,
    pool_pad:       1,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv2_1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (16, 16, 96),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   192,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.10},
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv2_2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (16, 16, 192),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   192,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.10},
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool2_op_cfg = Pool2dOperatorConfig{
    in_dims:        (16, 16, 192),
    pool_size:      3,
    pool_stride:    2,
    pool_pad:       1,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv3_op_cfg = Conv2dOperatorConfig{
    in_dims:        (8, 8, 192),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       0,
    out_channels:   192,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.10},
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv4_op_cfg = Conv2dOperatorConfig{
    in_dims:        (6, 6, 192),
    conv_size:      1,
    conv_stride:    1,
    conv_pad:       0,
    out_channels:   192,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.10},
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv5_op_cfg = Conv2dOperatorConfig{
    in_dims:        (6, 6, 192),
    conv_size:      1,
    conv_stride:    1,
    conv_pad:       0,
    out_channels:   10,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.10},
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool5_op_cfg = Pool2dOperatorConfig{
    in_dims:        (6, 6, 10),
    pool_size:      6,
    pool_stride:    1,
    pool_pad:       0,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let loss_cfg = CategoricalLossConfig{
    num_categories: 10,
  };

  let mut worker_cfg = PipelineOperatorConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    .conv2d(conv1_1_op_cfg)
    .conv2d(conv1_2_op_cfg)
    .pool2d(pool1_op_cfg)
    .conv2d(conv2_1_op_cfg)
    .conv2d(conv2_2_op_cfg)
    .pool2d(pool2_op_cfg)
    .conv2d(conv3_op_cfg)
    .conv2d(conv4_op_cfg)
    .conv2d(conv5_op_cfg)
    .pool2d(pool5_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}

fn build_resnet20_arch() -> PipelineOperatorConfig {
  let data_op_cfg = Data3dOperatorConfig{
    in_dims:        (32, 32, 3),
    normalize:      true,
    preprocs:       vec![],
  };
  let conv1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (32, 32, 3),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   16,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingRectFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv1_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (32, 32, 16),
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingRectFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv2_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (32, 32, 16),
    out_dims:       (16, 16, 32),
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingRectFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv2_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (16, 16, 32),
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingRectFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv3_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (16, 16, 32),
    out_dims:       (8, 8, 64),
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingRectFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv3_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (8, 8, 64),
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingRectFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let global_pool_op_cfg = Pool2dOperatorConfig{
    in_dims:        (8, 8, 64),
    pool_size:      8,
    pool_stride:    1,
    pool_pad:       0,
    //pool_op:        PoolOperation::Max,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let affine_op_cfg = AffineOperatorConfig{
    in_channels:    64 * 64,
    //in_channels:    64,
    out_channels:   10,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInit::Uniform{half_range: 0.15},
    backend:        AffineBackend::CublasGemm,
  };
  let loss_cfg = CategoricalLossConfig{
    num_categories: 10,
  };

  let mut worker_cfg = PipelineOperatorConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    .conv2d(conv1_op_cfg)
    .stack_res_conv2d(res_conv1_op_cfg)
    .stack_res_conv2d(res_conv1_op_cfg)
    .stack_res_conv2d(res_conv1_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv2_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv3_op_cfg)
    .stack_res_conv2d(res_conv3_op_cfg)
    .stack_res_conv2d(res_conv3_op_cfg)
    //.pool2d(global_pool_op_cfg)
    .affine(affine_op_cfg)
    .softmax_kl_loss(loss_cfg);

  worker_cfg
}

fn main() {
  env_logger::init().unwrap();

  let num_workers = 1;
  let batch_size = 128;
  info!("num workers: {} batch size: {}",
      num_workers, batch_size);

  let sgd_opt_cfg = SgdOptConfig{
    init_t:         None,
    minibatch_size: batch_size,
    step_size:      StepSizeSchedule::Constant{step_size: 0.001},
    momentum:       MomentumStyle::Nesterov{momentum: 0.0},
    l2_reg_coef:    1.0e-4,
    display_iters:  100,
    valid_iters:    1000,
    save_iters:     5000,
  };
  info!("sgd: {:?}", sgd_opt_cfg);

  let datum_cfg = SampleDatumConfig::Bytes3d;
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 10,
  };

  // This works and gets 74% test accuracy as claimed.
  //let worker_cfg = build_krizh26_arch();
  // This doesn't really work.
  //let worker_cfg = build_allconv_arch();
  // ResNets.
  let worker_cfg = build_resnet20_arch();

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

      let sgd_opt = SyncSgdOpt::new(opt_shared);
      sgd_opt.train(sgd_opt_cfg, datum_cfg, label_cfg, &mut train_data, &mut valid_data, &mut worker);
      join_barrier.wait();
    });
  }
  join_barrier.wait();
}
