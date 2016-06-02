use data_new::{
  SampleDatumConfig, SampleLabelConfig,
  DatasetConfig,
  SampleIterator, RandomEpisodeIterator,
  PartitionDataSource,
};
use operator::{
  OpCapability,
  ActivationFunction,
  ParamsInit,
  PoolOperation,
  Pool2dOperatorConfig,
  DropoutOperatorConfig,
};
use operator::comm::{
  DeviceSyncGossipCommWorkerBuilder,
};
use operator::input::{
  Data3dPreproc,
  Data3dOperatorConfig,
  VarData3dPreprocConfig,
  VarData3dOperatorConfig,
};
use operator::loss::{
  CategoricalLossConfig,
};
use operator::affine::{
  AffineBackend,
  AffineOperatorConfig,
};
use operator::conv::{
  Conv2dFwdBackend,
  Conv2dBwdBackend,
  Conv2dOperatorConfig,
  BNormMovingAverage,
  BNormConv2dOperatorConfig,
  StackResConv2dOperatorConfig,
  ProjStackResConv2dOperatorConfig,
};
use operator::worker::{
  OperatorWorkerBuilder,
  OperatorWorker,
  //SequentialOperatorConfig,
  SequentialOperatorWorkerBuilder,
};
use operator::seq::{
  SequentialOperatorConfig,
};
use opt::sgd::{
  SgdOptConfig, StepSizeSchedule, Momentum, SyncOrder,
  //OptSharedData,
  //SgdOpt,
};
use worker::gossip_dist::{
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
const BNORM_EMA_FACTOR: f64 = 0.1;
const BNORM_EPSILON:    f64 = 1.0e-5;

pub fn build_resnet18_var224x224() -> SequentialOperatorConfig {
  let data_op_cfg = VarData3dOperatorConfig{
    in_stride:      16 * 480 * 480 * 3,
    out_dims:       (224, 224, 3),
    normalize:      true,
    preprocs:       vec![
      VarData3dPreprocConfig::ScaleBicubic{
        scale_lower:    256,
        scale_upper:    480,
      },
      VarData3dPreprocConfig::Crop{
        crop_width:     224,
        crop_height:    224,
      },
      VarData3dPreprocConfig::FlipX,
      VarData3dPreprocConfig::AddPixelwisePCALightingNoise{
        std_dev:    0.1,
        singular_vecs:  vec![
          vec![-0.5675, -0.5808, -0.5836],
          vec![ 0.7192, -0.0045, -0.6948],
          vec![ 0.4009, -0.8140,  0.4203],
        ],
        singular_vals:  vec![
          0.2175, 0.0188, 0.0045,
        ],
      },
    ],
  };
  let bnorm_conv1_op_cfg = BNormConv2dOperatorConfig{
    in_dims:        (224, 224, 3),
    conv_size:      7,
    conv_stride:    2,
    conv_pad:       3,
    out_channels:   64,
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    pre_act_func:   ActivationFunction::Identity,
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
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv3_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (56, 56, 64),
    out_dims:       (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv3_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv4_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    out_dims:       (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv4_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv5_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    out_dims:       (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv5_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
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
    .var_data3d(data_op_cfg)
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

pub fn build_resnet18pool_var224x224() -> SequentialOperatorConfig {
  let data_op_cfg = VarData3dOperatorConfig{
    in_stride:      16 * 480 * 480 * 3,
    out_dims:       (224, 224, 3),
    normalize:      true,
    preprocs:       vec![
      VarData3dPreprocConfig::ScaleBicubic{
        scale_lower:    256,
        scale_upper:    480,
      },
      VarData3dPreprocConfig::Crop{
        crop_width:     224,
        crop_height:    224,
      },
      VarData3dPreprocConfig::FlipX,
      VarData3dPreprocConfig::AddPixelwisePCALightingNoise{
        std_dev:    0.1,
        singular_vecs:  vec![
          vec![-0.5675, -0.5808, -0.5836],
          vec![ 0.7192, -0.0045, -0.6948],
          vec![ 0.4009, -0.8140,  0.4203],
        ],
        singular_vals:  vec![
          0.2175, 0.0188, 0.0045,
        ],
      },
    ],
  };
  let bnorm_conv1_op_cfg = BNormConv2dOperatorConfig{
    in_dims:        (224, 224, 3),
    conv_size:      7,
    conv_stride:    2,
    conv_pad:       3,
    out_channels:   64,
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    pre_act_func:   ActivationFunction::Identity,
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
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool2_op_cfg = Pool2dOperatorConfig{
    in_dims:        (56, 56, 64),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let proj_res_conv3_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (28, 28, 64),
    out_dims:       (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv3_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool3_op_cfg = Pool2dOperatorConfig{
    in_dims:        (28, 28, 128),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let proj_res_conv4_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (14, 14, 128),
    out_dims:       (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv4_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let pool4_op_cfg = Pool2dOperatorConfig{
    in_dims:        (14, 14, 256),
    pool_size:      2,
    pool_stride:    2,
    pool_pad:       0,
    pool_op:        PoolOperation::Average,
    act_func:       ActivationFunction::Identity,
  };
  let proj_res_conv5_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (7, 7, 256),
    out_dims:       (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv5_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
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
    .var_data3d(data_op_cfg)
    .bnorm_conv2d(bnorm_conv1_op_cfg)
    .pool2d(pool1_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .pool2d(pool2_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv3_op_cfg)
    .stack_res_conv2d(res_conv3_op_cfg)
    .pool2d(pool3_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv4_op_cfg)
    .stack_res_conv2d(res_conv4_op_cfg)
    .pool2d(pool4_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv5_op_cfg)
    .stack_res_conv2d(res_conv5_op_cfg)
    .pool2d(global_pool_op_cfg)
    .affine(aff1_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}

pub fn build_resnet34_var224x224() -> SequentialOperatorConfig {
  let data_op_cfg = VarData3dOperatorConfig{
    in_stride:      16 * 480 * 480 * 3,
    out_dims:       (224, 224, 3),
    normalize:      true,
    preprocs:       vec![
      VarData3dPreprocConfig::ScaleBicubic{
        scale_lower:    256,
        scale_upper:    480,
      },
      VarData3dPreprocConfig::Crop{
        crop_width:     224,
        crop_height:    224,
      },
      VarData3dPreprocConfig::FlipX,
      VarData3dPreprocConfig::AddPixelwisePCALightingNoise{
        std_dev:    0.1,
        singular_vecs:  vec![
          vec![-0.5675, -0.5808, -0.5836],
          vec![ 0.7192, -0.0045, -0.6948],
          vec![ 0.4009, -0.8140,  0.4203],
        ],
        singular_vals:  vec![
          0.2175, 0.0188, 0.0045,
        ],
      },
    ],
  };
  let bnorm_conv1_op_cfg = BNormConv2dOperatorConfig{
    in_dims:        (224, 224, 3),
    conv_size:      7,
    conv_stride:    2,
    conv_pad:       3,
    out_channels:   64,
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    pre_act_func:   ActivationFunction::Identity,
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
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv3_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (56, 56, 64),
    out_dims:       (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv3_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv4_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    out_dims:       (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv4_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let proj_res_conv5_op_cfg = ProjStackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    out_dims:       (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:    Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    bwd_backend:    Conv2dBwdBackend::CudnnFastest,
  };
  let res_conv5_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
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
    .var_data3d(data_op_cfg)
    .bnorm_conv2d(bnorm_conv1_op_cfg)
    .pool2d(pool1_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .stack_res_conv2d(res_conv2_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv3_op_cfg)
    .stack_res_conv2d(res_conv3_op_cfg)
    .stack_res_conv2d(res_conv3_op_cfg)
    .stack_res_conv2d(res_conv3_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv4_op_cfg)
    .stack_res_conv2d(res_conv4_op_cfg)
    .stack_res_conv2d(res_conv4_op_cfg)
    .stack_res_conv2d(res_conv4_op_cfg)
    .stack_res_conv2d(res_conv4_op_cfg)
    .stack_res_conv2d(res_conv4_op_cfg)
    .proj_stack_res_conv2d(proj_res_conv5_op_cfg)
    .stack_res_conv2d(res_conv5_op_cfg)
    .stack_res_conv2d(res_conv5_op_cfg)
    .pool2d(global_pool_op_cfg)
    .affine(aff1_op_cfg)
    .softmax_kl_loss(loss_cfg);
  worker_cfg
}
