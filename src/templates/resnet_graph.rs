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
use operator::graph::{GraphOperatorConfig};
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
  //Conv2dBwdBackend,
  Conv2dBwdFilterBackend,
  Conv2dBwdDataBackend,
  Conv2dOperatorConfig,
  BNormMovingAverage,
  BNormConv2dOperatorConfig,
  StackResConv2dOperatorConfig,
  ProjStackResConv2dOperatorConfig,
};

use std::path::{PathBuf};

// XXX(20160423): These are Torch (nn/THNN) defaults.
const BNORM_EMA_FACTOR: f64 = 0.1;
const BNORM_EPSILON:    f64 = 1.0e-5;

pub fn build_resnet18pool_var224x224() -> GraphOperatorConfig {
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
    fwd_backend:        Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    //bwd_backend:    Conv2dBwdBackend::CudnnNonDeterministic,
    bwd_filt_backend:   Conv2dBwdFilterBackend::CudnnDeterministic,
    bwd_data_backend:   Conv2dBwdDataBackend::CudnnNonDeterministic,
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
    fwd_backend:        Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    //bwd_backend:    Conv2dBwdBackend::CudnnNonDeterministic,
    bwd_filt_backend:   Conv2dBwdFilterBackend::CudnnNonDeterministic,
    bwd_data_backend:   Conv2dBwdDataBackend::CudnnDeterministic,
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
    fwd_backend:        Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    //bwd_backend:    Conv2dBwdBackend::CudnnNonDeterministic,
    bwd_filt_backend:   Conv2dBwdFilterBackend::CudnnNonDeterministic,
    bwd_data_backend:   Conv2dBwdDataBackend::CudnnDeterministic,
  };
  let res_conv3_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (28, 28, 128),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:        Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    //bwd_backend:    Conv2dBwdBackend::CudnnNonDeterministic,
    bwd_filt_backend:   Conv2dBwdFilterBackend::CudnnDeterministic,
    bwd_data_backend:   Conv2dBwdDataBackend::CudnnDeterministic,
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
    fwd_backend:        Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    //bwd_backend:    Conv2dBwdBackend::CudnnNonDeterministic,
    bwd_filt_backend:   Conv2dBwdFilterBackend::CudnnDeterministic,
    bwd_data_backend:   Conv2dBwdDataBackend::CudnnDeterministic,
  };
  let res_conv4_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (14, 14, 256),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:        Conv2dFwdBackend::CudnnFft,
    //bwd_backend:    Conv2dBwdBackend::CudnnNonDeterministic,
    bwd_filt_backend:   Conv2dBwdFilterBackend::CudnnFft,
    bwd_data_backend:   Conv2dBwdDataBackend::CudnnFft,
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
    fwd_backend:        Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    //bwd_backend:    Conv2dBwdBackend::CudnnNonDeterministic,
    bwd_filt_backend:   Conv2dBwdFilterBackend::CudnnNonDeterministic,
    bwd_data_backend:   Conv2dBwdDataBackend::CudnnDeterministic,
  };
  let res_conv5_op_cfg = StackResConv2dOperatorConfig{
    in_dims:        (7, 7, 512),
    bnorm_mov_avg:  BNormMovingAverage::Exponential{ema_factor: BNORM_EMA_FACTOR},
    bnorm_epsilon:  BNORM_EPSILON,
    act_func:       ActivationFunction::Rect,
    init_weights:   ParamsInit::KaimingFwd,
    fwd_backend:        Conv2dFwdBackend::CudnnImplicitPrecompGemm,
    //bwd_backend:    Conv2dBwdBackend::CudnnNonDeterministic,
    bwd_filt_backend:   Conv2dBwdFilterBackend::CudnnNonDeterministic,
    bwd_data_backend:   Conv2dBwdDataBackend::CudnnDeterministic,
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

  let mut graph_cfg = GraphOperatorConfig::new();
  graph_cfg
    .var_data3d(            "data",                 data_op_cfg)
    .bnorm_conv2d(          "conv1",    "data",     bnorm_conv1_op_cfg)
    .pool2d(                "pool1",    "conv1",    pool1_op_cfg)
    .stack_res_conv2d(      "conv2_1",  "pool1",    res_conv2_op_cfg)
    .stack_res_conv2d(      "conv2_2",  "conv2_1",  res_conv2_op_cfg)
    .pool2d(                "pool2",    "conv2_2",  pool2_op_cfg)
    .proj_stack_res_conv2d( "conv3_1",  "pool2",    proj_res_conv3_op_cfg)
    .stack_res_conv2d(      "conv3_2",  "conv3_1",  res_conv3_op_cfg)
    .pool2d(                "pool3",    "conv3_2",  pool3_op_cfg)
    .proj_stack_res_conv2d( "conv4_1",  "pool3",    proj_res_conv4_op_cfg)
    .stack_res_conv2d(      "conv4_2",  "conv4_1",  res_conv4_op_cfg)
    .pool2d(                "pool4",    "conv4_2",  pool4_op_cfg)
    .proj_stack_res_conv2d( "conv5_1",  "pool4",    proj_res_conv5_op_cfg)
    .stack_res_conv2d(      "conv5_2",  "conv5_1",  res_conv5_op_cfg)
    .pool2d(                "pool5",    "conv5_2",  global_pool_op_cfg)
    .affine(                "affine",   "pool5",    aff1_op_cfg)
    .softmax_kl_loss(       "loss",     "affine",   loss_cfg);
  graph_cfg
}
