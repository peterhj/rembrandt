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
  SequentialOperatorConfig,
  SequentialOperatorWorkerBuilder,
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

const BNORM_EMA_FACTOR: f64 = 0.01;

fn build_vgg_a_arch() -> SequentialOperatorConfig {
  let data_op_cfg = Data3dOperatorConfig{
    in_dims:        (256, 256, 3),
    normalize:      true,
    preprocs:       vec![
      Data3dPreproc::SubtractElemwiseMean{
        mean_path:  PathBuf::from("imagenet_mean_256x256x3.ndarray"),
      },
      Data3dPreproc::FlipX,
      Data3dPreproc::Crop{
        crop_width:     224,
        crop_height:    224,
      },
    ],
  };
  let conv1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (224, 224, 3),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   64,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool1_op_cfg = Pool2dOperatorConfig{
    in_dims:        (224, 224, 64),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (112, 112, 64),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   128,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool2_op_cfg = Pool2dOperatorConfig{
    in_dims:        (112, 112, 128),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv3_1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (56, 56, 128),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   256,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv3_2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (56, 56, 256),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   256,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool3_op_cfg = Pool2dOperatorConfig{
    in_dims:        (56, 56, 256),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv4_1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (28, 28, 256),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   512,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv4_2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (28, 28, 512),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   512,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool4_op_cfg = Pool2dOperatorConfig{
    in_dims:        (28, 28, 512),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv5_1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (14, 14, 512),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   512,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv5_2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (14, 14, 512),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   512,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool5_op_cfg = Pool2dOperatorConfig{
    in_dims:        (14, 14, 512),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let aff1_op_cfg = AffineOperatorConfig{
    in_channels:    25088,
    out_channels:   4096,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.01},
    //init_weights:   ParamsInit::KaimingFwd,
    backend:        AffineBackend::CublasGemm,
  };
  let drop1_op_cfg = DropoutOperatorConfig{
    channels:       4096,
    drop_ratio:     0.5,
  };
  let aff2_op_cfg = AffineOperatorConfig{
    in_channels:    4096,
    out_channels:   4096,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::Uniform{half_range: 0.01},
    //init_weights:   ParamsInit::KaimingFwd,
    backend:        AffineBackend::CublasGemm,
  };
  let drop2_op_cfg = DropoutOperatorConfig{
    channels:       4096,
    drop_ratio:     0.5,
  };
  let aff3_op_cfg = AffineOperatorConfig{
    in_channels:    4096,
    out_channels:   1000,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInit::Uniform{half_range: 0.01},
    //init_weights:   ParamsInit::KaimingFwd,
    backend:        AffineBackend::CublasGemm,
  };
  let loss_cfg = CategoricalLossConfig{
    num_categories: 1000,
  };

  let mut worker_cfg = SequentialOperatorConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    .conv2d(conv1_op_cfg)
    .pool2d(pool1_op_cfg)
    .conv2d(conv2_op_cfg)
    .pool2d(pool2_op_cfg)
    .conv2d(conv3_1_op_cfg)
    .conv2d(conv3_2_op_cfg)
    .pool2d(pool3_op_cfg)
    .conv2d(conv4_1_op_cfg)
    .conv2d(conv4_2_op_cfg)
    .pool2d(pool4_op_cfg)
    .conv2d(conv5_1_op_cfg)
    .conv2d(conv5_2_op_cfg)
    .pool2d(pool5_op_cfg)
    .affine(aff1_op_cfg)
    .dropout(drop1_op_cfg)
    .affine(aff2_op_cfg)
    .dropout(drop2_op_cfg)
    .affine(aff3_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}

fn build_vgg_a_avgpool_arch() -> SequentialOperatorConfig {
  let data_op_cfg = Data3dOperatorConfig{
    in_dims:        (256, 256, 3),
    //in_dims:        (224, 224, 3),
    normalize:      true,
    preprocs:       vec![],
  };
  let conv1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (256, 256, 3),
    //in_dims:        (224, 224, 3),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   64,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool1_op_cfg = Pool2dOperatorConfig{
    in_dims:        (256, 256, 64),
    //in_dims:        (224, 224, 64),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (128, 128, 64),
    //in_dims:        (112, 112, 64),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   128,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool2_op_cfg = Pool2dOperatorConfig{
    in_dims:        (128, 128, 128),
    //in_dims:        (112, 112, 128),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv3_1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (64, 64, 128),
    //in_dims:        (56, 56, 128),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   256,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv3_2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (64, 64, 256),
    //in_dims:        (56, 56, 256),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   256,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool3_op_cfg = Pool2dOperatorConfig{
    in_dims:        (64, 64, 256),
    //in_dims:        (56, 56, 256),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv4_1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (32, 32, 256),
    //in_dims:        (28, 28, 256),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   512,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv4_2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (32, 32, 512),
    //in_dims:        (28, 28, 512),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   512,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool4_op_cfg = Pool2dOperatorConfig{
    in_dims:        (32, 32, 512),
    //in_dims:        (28, 28, 512),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let conv5_1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (16, 16, 512),
    //in_dims:        (14, 14, 512),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   512,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let conv5_2_op_cfg = Conv2dOperatorConfig{
    in_dims:        (16, 16, 512),
    //in_dims:        (14, 14, 512),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   512,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool5_op_cfg = Pool2dOperatorConfig{
    in_dims:        (16, 16, 512),
    //in_dims:        (14, 14, 512),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let global_pool_op_cfg = Pool2dOperatorConfig{
    in_dims:        (8, 8, 512),
    //in_dims:        (14, 14, 512),
    pool_size:      8,
    pool_stride:    1,
    pool_pad:       0,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let aff1_op_cfg = AffineOperatorConfig{
    in_channels:    512,
    out_channels:   1000,
    act_func:       ActivationFunction::Identity,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::Normal{std: 0.05},
    //init_weights:   ParamsInit::KaimingFwd,
    backend:        AffineBackend::CublasGemm,
  };
  let loss_cfg = CategoricalLossConfig{
    num_categories: 1000,
  };

  let mut worker_cfg = SequentialOperatorConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    .conv2d(conv1_op_cfg)
    .pool2d(pool1_op_cfg)
    .conv2d(conv2_op_cfg)
    .pool2d(pool2_op_cfg)
    .conv2d(conv3_1_op_cfg)
    .conv2d(conv3_2_op_cfg)
    .pool2d(pool3_op_cfg)
    .conv2d(conv4_1_op_cfg)
    .conv2d(conv4_2_op_cfg)
    .pool2d(pool4_op_cfg)
    .conv2d(conv5_1_op_cfg)
    .conv2d(conv5_2_op_cfg)
    .pool2d(pool5_op_cfg)
    .pool2d(global_pool_op_cfg)
    .affine(aff1_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}

fn build_resnet10_arch() -> SequentialOperatorConfig {
  let data_op_cfg = Data3dOperatorConfig{
    in_dims:        (256, 256, 3),
    normalize:      true,
    preprocs:       vec![
      Data3dPreproc::SubtractElemwiseMean{
        mean_path:  PathBuf::from("imagenet_mean_256x256x3.ndarray"),
      },
      Data3dPreproc::FlipX,
      Data3dPreproc::Crop{
        crop_width:     224,
        crop_height:    224,
      },
    ],
  };
  let conv1_op_cfg = Conv2dOperatorConfig{
    in_dims:        (224, 224, 3),
    conv_size:      3,
    conv_stride:    1,
    conv_pad:       1,
    out_channels:   64,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool1_op_cfg = Pool2dOperatorConfig{
    in_dims:        (224, 224, 64),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let proj_res_conv2_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (112, 112, 64),
    out_dims:       (56, 56, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv3_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (56, 56, 128),
    out_dims:       (28, 28, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv4_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (28, 28, 256),
    out_dims:       (14, 14, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv5_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (14, 14, 512),
    out_dims:       (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let global_pool_op_cfg = Pool2dOperatorConfig{
    in_dims:        (7, 7, 512),
    pool_size:      7,
    pool_stride:    1,
    pool_pad:       0,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let aff1_op_cfg = AffineOperatorConfig{
    in_channels:    512,
    out_channels:   1000,
    act_func:       ActivationFunction::Identity,
    init_weights:   ParamsInit::Uniform{half_range: 0.05},
    //init_weights:   ParamsInit::Normal{std: 0.05},
    //init_weights:   ParamsInit::KaimingFwd,
    backend:        AffineBackend::CublasGemm,
  };
  let loss_cfg = CategoricalLossConfig{
    num_categories: 1000,
  };

  let mut worker_cfg = SequentialOperatorConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    .conv2d(conv1_op_cfg)
    .pool2d(pool1_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv2_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv3_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv4_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv5_op_cfg)
    .pool2d(global_pool_op_cfg)
    .affine(aff1_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}

fn build_resnet18_arch() -> SequentialOperatorConfig {
  let data_op_cfg = Data3dOperatorConfig{
    in_dims:        (256, 256, 3),
    normalize:      true,
    preprocs:       vec![
      Data3dPreproc::SubtractElemwiseMean{
        mean_path:  PathBuf::from("imagenet_mean_256x256x3.ndarray"),
      },
      Data3dPreproc::FlipX,
      Data3dPreproc::Crop{
        crop_width:     224,
        crop_height:    224,
      },
      /*Data3dPreproc::AddGaussianNoise{
        std_dev:    0.01,
      },*/
    ],
  };
  let bnorm_conv1_op_cfg = BNormConv2dOperatorConfig{
    in_dims:        (224, 224, 3),
    conv_size:      7,
    conv_stride:    2,
    conv_pad:       3,
    out_channels:   64,
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool1_op_cfg = Pool2dOperatorConfig{
    in_dims:        (112, 112, 64),
    pool_size:      3,
    pool_stride:    2,
    pool_pad:       1,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let res_conv2_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (56, 56, 64),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv3_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (56, 56, 64),
    out_dims:       (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv3_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv4_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    out_dims:       (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv4_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv5_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    out_dims:       (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv5_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let global_pool_op_cfg = Pool2dOperatorConfig{
    in_dims:        (7, 7, 512),
    pool_size:      7,
    pool_stride:    1,
    pool_pad:       0,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let aff1_op_cfg = AffineOperatorConfig{
    in_channels:    512,
    out_channels:   1000,
    act_func:       ActivationFunction::Identity,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    //init_weights:   ParamsInit::Normal{std: 0.05},
    init_weights:   ParamsInit::Xavier,
    //init_weights:   ParamsInit::KaimingFwd,
    backend:        AffineBackend::CublasGemm,
  };
  let loss_cfg = CategoricalLossConfig{
    num_categories: 1000,
  };

  let mut worker_cfg = SequentialOperatorConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    .bnorm_conv2d(bnorm_conv1_op_cfg)
    .pool2d(pool1_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv3_op_cfg)
    .stack_res_conv2d(res_conv3_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv4_op_cfg)
    .stack_res_conv2d(res_conv4_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv5_op_cfg)
    .stack_res_conv2d(res_conv5_op_cfg)
    .pool2d(global_pool_op_cfg)
    .affine(aff1_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}

fn build_resnet20_arch() -> SequentialOperatorConfig {
  let data_op_cfg = Data3dOperatorConfig{
    in_dims:        (256, 256, 3),
    normalize:      true,
    preprocs:       vec![
      Data3dPreproc::SubtractElemwiseMean{
        mean_path:  PathBuf::from("imagenet_mean_256x256x3.ndarray"),
      },
      Data3dPreproc::FlipX,
      Data3dPreproc::Crop{
        crop_width:     224,
        crop_height:    224,
      },
      /*Data3dPreproc::AddGaussianNoise{
        std_dev:    0.01,
      },*/
    ],
  };
  let bnorm_conv1_op_cfg = BNormConv2dOperatorConfig{
    in_dims:        (224, 224, 3),
    conv_size:      7,
    conv_stride:    2,
    conv_pad:       3,
    out_channels:   64,
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool1_op_cfg = Pool2dOperatorConfig{
    in_dims:        (112, 112, 64),
    pool_size:      3,
    pool_stride:    2,
    pool_pad:       1,
    pool_op:        PoolOperation::Max,
    act_func:       ActivationFunction::Identity,
  };
  let res_conv2_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (56, 56, 64),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv3_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (56, 56, 64),
    out_dims:       (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv3_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv4_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    out_dims:       (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv4_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv5_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    out_dims:       (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv5_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  1.0e-4,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let global_pool_op_cfg = Pool2dOperatorConfig{
    in_dims:        (7, 7, 512),
    pool_size:      7,
    pool_stride:    1,
    pool_pad:       0,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let aff1_op_cfg = AffineOperatorConfig{
    in_channels:    512,
    out_channels:   1000,
    act_func:       ActivationFunction::Identity,
    //init_weights:   ParamsInit::Uniform{half_range: 0.05},
    //init_weights:   ParamsInit::Normal{std: 0.05},
    init_weights:   ParamsInit::Xavier,
    //init_weights:   ParamsInit::KaimingFwd,
    backend:        AffineBackend::CublasGemm,
  };
  let loss_cfg = CategoricalLossConfig{
    num_categories: 1000,
  };

  let mut worker_cfg = SequentialOperatorConfig::new();
  worker_cfg
    .data3d(data_op_cfg)
    .bnorm_conv2d(bnorm_conv1_op_cfg)
    .pool2d(pool1_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv3_op_cfg)
    .stack_res_conv2d(res_conv3_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv4_op_cfg)
    .stack_res_conv2d(res_conv4_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv5_op_cfg)
    .stack_res_conv2d(res_conv5_op_cfg)
    .stack_res_conv2d(res_conv5_op_cfg)
    .pool2d(global_pool_op_cfg)
    .affine(aff1_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}

fn build_resnet34_arch() -> SequentialOperatorConfig {
  // FIXME(20160421)
  unimplemented!();
}

fn main() {
  env_logger::init().unwrap();

  let num_workers = 1;
  //let num_workers = 4;
  let batch_size = 64;
  let minibatch_size = 256;
  info!("num workers: {} batch size: {}",
      num_workers, batch_size);

  let sgd_opt_cfg = SgdOptConfig{
    init_t:         None,
    minibatch_size: minibatch_size,
    step_size:      StepSizeSchedule::Constant{step_size: 0.1},
    //momentum:       MomentumStyle::Zero,
    //momentum:       MomentumStyle::Sgd{momentum: 0.9},
    momentum:       MomentumStyle::Nesterov{momentum: 0.9},
    l2_reg_coef:    1.0e-4,
    display_iters:  20,
    valid_iters:    5000,
    save_iters:     5000,
  };
  info!("sgd: {:?}", sgd_opt_cfg);

  let datum_cfg = SampleDatumConfig::Bytes3d;
  let label_cfg = SampleLabelConfig::Category{
    num_categories: 1000,
  };

  //let worker_cfg = build_vgg_a_arch();
  //let worker_cfg = build_vgg_a_avgpool_arch();

  //let worker_cfg = build_resnet10_arch();
  let worker_cfg = build_resnet18_arch();
  //let worker_cfg = build_resnet20_arch();

  info!("operator: {:?}", worker_cfg);

  // FIXME(20160331)
  let comm_worker_builder = DeviceSyncGossipCommWorkerBuilder::new(num_workers, 1, worker_cfg.params_len());
  let worker_builder = SequentialOperatorWorkerBuilder::new(num_workers, batch_size, worker_cfg, OpCapability::Backward);
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

      let dataset_cfg = DatasetConfig::open(&PathBuf::from("examples/imagenet.data"));
      let mut train_data =
          /*//RandomEpisodeIterator::new(
          SampleIterator::new(
              dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "train"),
          );*/
          dataset_cfg.build_iterator("train");
      let mut valid_data =
          /*SampleIterator::new(
              Box::new(PartitionDataSource::new(tid, num_workers, dataset_cfg.build_with_cfg(datum_cfg, label_cfg, "valid")))
          );*/
          dataset_cfg.build_iterator("valid");

      let sgd_opt = SyncSgdOpt::new(opt_shared);
      sgd_opt.train(sgd_opt_cfg, datum_cfg, label_cfg, &mut *train_data, &mut *valid_data, &mut worker);
      join_barrier.wait();
    });
  }
  join_barrier.wait();
}
