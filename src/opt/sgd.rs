use data_new::{
  DataIterator,
  SampleDatum, SampleDatumConfig, SampleLabel, SampleLabelConfig, 
};
use operator::{Operator};

//use array_cuda::device::context::{DeviceCtxRef};

#[derive(Clone, Copy, Debug)]
pub struct SgdOptConfig {
  pub init_t:         Option<usize>,
  pub minibatch_size: usize,
  pub step_size:      StepSizeSchedule,
  pub momentum:       MomentumSchedule,
  pub l2_reg_coef:    f32,

  pub display_iters:  usize,
  pub valid_iters:    usize,
  pub save_iters:     usize,
}

#[derive(Clone, Copy, Debug)]
pub enum StepSizeSchedule {
  Constant{step_size: f32},
  DecayOnce{
    step0:          f32,
    step0_iters:    usize,
    final_step:     f32,
  },
  Decay{
    init_step:      f32,
    decay_rate:     f32,
    decay_iters:    usize,
  },
}

#[derive(Clone, Copy, Debug)]
pub enum MomentumSchedule {
  Constant{momentum: f32},
}

pub struct SgdOpt;

impl SgdOpt {
  pub fn train(&self, mut sgd_opt_cfg: SgdOptConfig, train_data: &mut DataIterator, valid_data: &mut DataIterator, operator: &mut Operator) {
    // FIXME(20160330)
    unimplemented!();
  }
}
