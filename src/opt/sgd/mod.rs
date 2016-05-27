use data::{DataShard, DataIter};
use data_new::{
  DataIterator,
  SampleDatum, SampleDatumConfig, SampleLabel, SampleLabelConfig, 
};
use operator::{Operator, CompleteOperator, OpPhase, Regularization};
use operator::worker::{OperatorWorker};

use array::{Shape};
//use array_cuda::device::context::{DeviceCtxRef};

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{Write, BufWriter};
use std::path::{PathBuf};
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use time::{get_time};

pub mod new;

pub mod parallel;
//pub mod seq;

#[derive(Clone, RustcDecodable, RustcEncodable, Debug)]
pub struct SgdOptConfig {
  pub init:           InitBehavior,
  pub minibatch_size: usize,
  pub step_size:      StepSizeSchedule,
  pub step_ref:       StepSizeReference,
  pub momentum:       Momentum,
  pub l2_reg_coef:    f32,
  pub sync_order:     SyncOrder,
  pub comm_interval:  usize,
  pub checkpoint:     CheckpointBehavior,

  pub display_iters:  usize,
  pub checkpoint_iters:   usize,
  pub checkpoint_dir:     PathBuf,
  pub save_iters:     usize,
  pub valid_iters:    usize,
}

#[derive(Clone, Copy, RustcDecodable, RustcEncodable, Debug)]
pub enum InitBehavior {
  InitOrResume,
  ResumeFrom{t: usize},
}

#[derive(Clone, Copy, RustcDecodable, RustcEncodable, Debug)]
pub enum StepSizeSchedule {
  Constant{step_size: f32},
  Anneal1{
    step0:          f32,
    step1:          f32,
    step1_iters:    usize,
  },
  Anneal2{
    step0:          f32,
    step1:          f32,
    step1_iters:    usize,
    step2:          f32,
    step2_iters:    usize,
  },
  Anneal4{
    step0:          f32,
    step1:          f32,
    step1_iters:    usize,
    step2:          f32,
    step2_iters:    usize,
    step3:          f32,
    step3_iters:    usize,
    step4:          f32,
    step4_iters:    usize,
  },
  Decay{
    init_step:      f32,
    decay_rate:     f32,
    decay_iters:    usize,
  },
  Inverse{
    init_step_size: f32,
    lambda:         f32,
  },
}

impl StepSizeSchedule {
  pub fn at_iter(&self, t: usize) -> f32 {
    match self {
      &StepSizeSchedule::Constant{step_size} => {
        step_size
      }
      &StepSizeSchedule::Anneal1{step0, step1_iters, step1} => {
        if t < step1_iters {
          step0
        } else {
          step1
        }
      }
      &StepSizeSchedule::Anneal2{step0, step1_iters, step1, step2_iters, step2} => {
        if t < step1_iters {
          step0
        } else if t < step2_iters {
          step1
        } else {
          step2
        }
      }
      &StepSizeSchedule::Anneal4{
        step0,
        step1_iters, step1,
        step2_iters, step2,
        step3_iters, step3,
        step4_iters, step4} =>
      {
        if t < step1_iters {
          step0
        } else if t < step2_iters {
          step1
        } else if t < step3_iters {
          step2
        } else if t < step4_iters {
          step3
        } else {
          step4
        }
      }
      &StepSizeSchedule::Decay{..} => {
        // FIXME(20160330)
        unimplemented!();
      }
      &StepSizeSchedule::Inverse{init_step_size, lambda} => {
        init_step_size / (1.0 + init_step_size * lambda * (t as f32))
      }
    }
  }
}

#[derive(Clone, Copy, RustcDecodable, RustcEncodable, Debug)]
pub enum StepSizeReference {
  Local,
  Checkpoint,
}

#[derive(Clone, Copy, RustcDecodable, RustcEncodable, Debug)]
pub enum Momentum {
  Zero,
  Update{mu: f32},
  UpdateNesterov{mu: f32},
  Gradient{mu: f32},
  GradientNesterov{mu: f32},
}

#[derive(Clone, Copy, RustcDecodable, RustcEncodable, Debug)]
pub enum SyncOrder {
  StepThenSyncParams,
  SyncParamsThenStep,
  SyncGradsThenStep,
  SyncParamsAndGradsThenStep,
}

#[derive(Clone, Copy, RustcDecodable, RustcEncodable, Debug)]
pub enum CheckpointBehavior {
  Discard,
  Keep,
}

/*pub struct OptSharedData {
  pub acc_correct_count:    AtomicUsize,
  pub acc_total_count:      AtomicUsize,
  barrier:  Barrier,
}

impl OptSharedData {
  pub fn new(num_workers: usize) -> OptSharedData {
    OptSharedData{
      acc_correct_count:    AtomicUsize::new(0),
      acc_total_count:      AtomicUsize::new(0),
      barrier:  Barrier::new(num_workers),
    }
  }

  pub fn sync(&self) {
    self.barrier.wait();
    fence(Ordering::AcqRel);
  }
}*/

pub struct SerialSgdOptConfig {
  pub init:           InitBehavior,
  pub minibatch_size: usize,
  pub step_size:      StepSizeSchedule,
  pub momentum:       Momentum,
  pub l2_reg_coef:    f32,

  pub display_iters:  usize,
  pub checkpoint_iters:   usize,
  pub checkpoint_dir:     PathBuf,
  pub save_iters:     usize,
  pub valid_iters:    usize,
}

pub struct SerialSgdOpt {
  config:   SerialSgdOptConfig,
}

impl SerialSgdOpt {
  pub fn new(config: SerialSgdOptConfig) -> SerialSgdOpt {
    unimplemented!();
  }

  pub fn train(&mut self,
      mut train_data: &mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>,
      mut valid_data: Option<&mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>>,
      operator: &mut CompleteOperator)
  {
    unimplemented!();
  }

  pub fn validate(&mut self,
      mut valid_data: Option<&mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>>,
      operator: &mut CompleteOperator)
  {
    unimplemented!();
  }
}
