use data_new::{
  SampleDatumConfig, SampleLabelConfig,
  DatasetConfig,
  SampleIterator, RandomEpisodeIterator,
  PartitionDataSource,
};
use operator::{
  OpCapability,
  ActivationFunction, ParamsInit,
  Data3dPreproc, AffineBackend, Conv2dFwdBackend, Conv2dBwdBackend, PoolOperation,
  Data3dOperatorConfig,
  AffineOperatorConfig,
  Conv2dOperatorConfig,
  Pool2dOperatorConfig,
  DropoutOperatorConfig,
};
use operator::loss::{
  CategoricalLossConfig,
};
use operator::comm::{
  DeviceSyncGossipCommWorkerBuilder,
};
use operator::conv::{
  BNormMovingAverage,
  BNormConv2dOperatorConfig,
  StackResConv2dOperatorConfig,
  ProjStackResConv2dOperatorConfig,
};
use operator::worker::{
  OperatorWorkerBuilder,
  OperatorWorker,
  SequentialOperatorConfig,
  SequentialOperatorWorkerBuilder,
};
use opt::sgd::{
  SgdOptConfig, StepSizeSchedule, Momentum, SyncOrder, OptSharedData, //SgdOpt,
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

pub fn build_vgg_a() -> SequentialOperatorConfig {
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
