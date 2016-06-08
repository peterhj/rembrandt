use data::{DataShard, DataIter};
use data_new::{
  DataIterator,
  SampleDatum, SampleDatumConfig, SampleLabel, SampleLabelConfig, 
};
use operator::{Operator, CompleteOperator, OpPhase, Regularization};
use operator::worker::{OperatorWorker};

use array::{Shape};
//use array_cuda::device::context::{DeviceCtxRef};

use rand::{Rng, thread_rng};
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

#[derive(Clone, Debug)]
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
    SerialSgdOpt{config: config}
  }

  pub fn train(&mut self,
      mut train_data: &mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>,
      mut valid_data: Option<&mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>>,
      operator: &mut CompleteOperator)
  {
    let batch_size = operator.batch_size();
    let minibatch_size = self.config.minibatch_size;
    let minibatch_weight = 1.0 / minibatch_size as f32;
    let epoch_size = train_data.num_shard_samples();

    let mut init_t = 0;
    /*match self.config.init {
      InitBehavior::InitOrResume => {
        if operator.can_rollback(&self.config.checkpoint_dir).is_some() {
          operator.rollback_state(&self.config.checkpoint_dir);
          init_t = operator.can_rollback(&self.config.checkpoint_dir).unwrap();
        } else {
          let shared_seed = operator.shared_seed();
          operator.init_param(shared_seed);
        }
      }
      InitBehavior::ResumeFrom{t} => {
        operator.rollback_param(Some(t), &self.config.checkpoint_dir);
        init_t = t;
      }
    }*/
    //let shared_seed = operator.shared_seed();
    let seed = [thread_rng().next_u64(), thread_rng().next_u64()];
    operator.init_param(seed);

    // If we are using the standard Nesterov update, apply some extra
    // momentum before the next iteration begins.
    match self.config.momentum {
      Momentum::UpdateNesterov{mu} => {
        if init_t > 0 {
          operator.update_param(mu);
        }
      }
      _ => {}
    }

    let mut start_time = get_time();
    let mut minibatch_start_time = start_time;
    let mut minibatch_offset_ms: i64 = 0;

    let mut minibatch_acc_correct_count = 0;
    let mut minibatch_acc_total_count = 0;
    let mut minibatch_acc_loss = 0.0;

    let mut display_acc_correct_count = 0;
    let mut display_acc_total_count = 0;
    let mut display_acc_loss = 0.0;

    let mut epoch_counter = 0;
    let mut iter_counter = init_t;
    let mut local_counter = 0;
    let mut batch_counter = 0;

    loop {
      train_data.reset();
      for (datum, maybe_label) in &mut train_data {
        /*match (label_cfg, maybe_label.as_ref()) {
          (SampleLabelConfig::Category{num_categories}, Some(&SampleLabel::Category{category})) => {
            assert!(category >= 0);
            assert!(category < num_categories);
          }
          _ => panic!("SgdOpt: unsupported label"),
        }*/
        match datum {
          SampleDatum::WHCBytes(ref frame_bytes) => {
            //println!("DEBUG: frame: {:?}", frame_bytes.as_slice());
            let frame_len = frame_bytes.bound().len();
            operator
              .stage_shape(batch_counter, frame_bytes.bound());
            operator
              .expose_host_frame_buf(batch_counter)[ .. frame_len]
              .copy_from_slice(frame_bytes.as_slice());
            operator
              .preload_frame(batch_counter);
          }
          _ => unimplemented!(),
        }
        operator.stage_label(batch_counter, &maybe_label.unwrap());
        operator.stage_weight(batch_counter, minibatch_weight);
        local_counter += 1;
        epoch_counter = local_counter / epoch_size;
        let epoch_offset = local_counter % epoch_size;
        batch_counter += 1;

        // With a full batch of data, compute a full forward/backward pass.
        assert!(batch_counter <= batch_size);
        if batch_counter == batch_size {
          //operator.next();
          //operator.input_operator().load_frames(batch_size);
          operator.wait_preload_frames(batch_size);
          operator.load_labels(batch_size);
          operator.load_weights(batch_size);
          operator.forward(batch_size, OpPhase::Training{t: iter_counter});
          operator.backward(batch_size);
          operator.store_output_categories(batch_size);
          let local_loss = operator.store_loss(batch_size);
          let local_correct_count = operator.accuracy_count(batch_size);
          display_acc_correct_count += local_correct_count;
          display_acc_total_count += batch_size;
          display_acc_loss += local_loss;
          minibatch_acc_correct_count += local_correct_count;
          minibatch_acc_total_count += batch_size;
          minibatch_acc_loss += local_loss;
          batch_counter = 0;
        }

        if local_counter % minibatch_size == 0 {
          let l2_reg_coef = self.config.l2_reg_coef;
          let step_size = self.config.step_size.at_iter(iter_counter);

          // Apply regularization to the current gradient.
          operator.regularize(Regularization::L2{l2_reg_coef: l2_reg_coef});

          // If we are using the standard Nesterov update, unapply the extra
          // momentum.
          match self.config.momentum {
            Momentum::UpdateNesterov{mu} => {
              if iter_counter > 0 {
                operator.update_param(-mu);
              }
            }
            _ => {}
          }

          // Increase the iteration counter _before_ updates and communication.
          iter_counter += 1;

          // Compute the update, possibly with momentum.
          match self.config.momentum {
            Momentum::Zero                  => operator.accumulate_grad(-step_size, 0.0),
            Momentum::Update{mu} |
            Momentum::UpdateNesterov{mu}    => operator.accumulate_grad(-step_size, mu),
            // XXX(20160422): These are the Torch `optim.sgd`-style update;
            // see: <https://github.com/torch/optim/blob/master/sgd.lua>.
            Momentum::Gradient{mu}          => operator.accumulate_grad(1.0, mu),
            Momentum::GradientNesterov{mu}  => operator.accumulate_grad(1.0, mu),
          }

          // Apply the update, possibly with momentum.
          match self.config.momentum {
            Momentum::Zero                  => operator.update_param(1.0),
            Momentum::Update{mu} |
            Momentum::UpdateNesterov{mu}    => operator.update_param(1.0),
            // XXX(20160422): These are the Torch `optim.sgd`-style update;
            // see: <https://github.com/torch/optim/blob/master/sgd.lua>.
            Momentum::Gradient{mu}          => operator.update_param(-step_size),
            Momentum::GradientNesterov{mu}  => operator.update_param2(-step_size, -step_size * mu),
          }

          operator.reset();

          let minibatch_lap_time = get_time();
          let minibatch_elapsed_ms = (minibatch_lap_time - minibatch_start_time).num_milliseconds() + minibatch_offset_ms;
          let minibatch_accuracy = minibatch_acc_correct_count as f32 / minibatch_acc_total_count as f32;
          /*writeln!(&mut self.local_log_file, "{},step,{:.6},{:.4},{:.3}",
              iter_counter, minibatch_acc_loss, 1.0 - minibatch_accuracy, minibatch_elapsed_ms as f32 * 0.001).unwrap();*/
          minibatch_start_time = get_time();
          minibatch_offset_ms = 0;
          minibatch_acc_correct_count = 0;
          minibatch_acc_total_count = 0;
          minibatch_acc_loss = 0.0;

          if iter_counter % self.config.display_iters == 0 {
            let lap_time = minibatch_lap_time;
            let elapsed_ms = (lap_time - start_time).num_milliseconds();
            let acc_correct_count = display_acc_correct_count;
            let acc_total_count = display_acc_total_count;
            let accuracy = acc_correct_count as f32 / acc_total_count as f32;
            let avg_loss = display_acc_loss / self.config.display_iters as f32;
            info!("SgdOpt: train: iter: {} epoch: {} sample: {}/{} step: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
                iter_counter, epoch_counter,
                epoch_offset, epoch_size,
                step_size,
                avg_loss,
                accuracy,
                elapsed_ms as f32 * 0.001,
            );
            display_acc_correct_count = 0;
            display_acc_total_count = 0;
            display_acc_loss = 0.0;
            //self.local_log_file.flush().unwrap();
            start_time = get_time();
            minibatch_start_time = start_time;
          }

          if iter_counter % self.config.valid_iters == 0 {
            if let Some(ref mut valid_data) = valid_data {
              self.validate(*valid_data, operator);
            }
            start_time = get_time();
            minibatch_start_time = start_time;
          }

          // If we are using the standard Nesterov update, apply some extra
          // momentum before the next iteration begins.
          // XXX(20160406): Interestingly, we should use the local update rather
          // than the communicated update with momentum.
          //operator.set_grads_with_params_diff();
          match self.config.momentum {
            Momentum::UpdateNesterov{mu} => {
              operator.update_param(mu);
            }
            _ => {}
          }
        }
      }
    }
  }

  pub fn validate(&mut self,
      mut valid_data: &mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>,
      operator: &mut CompleteOperator)
  {
    let batch_size = operator.batch_size();
    let epoch_size = valid_data.num_shard_samples();
    let weight = 1.0 / epoch_size as f32;

    let mut start_time = get_time();

    let mut display_acc_correct_count = 0;
    let mut display_acc_total_count = 0;
    let mut display_acc_loss = 0.0;

    let mut local_counter = 0;
    let mut batch_counter = 0;

    valid_data.reset();
    for (datum, maybe_label) in valid_data.take(epoch_size) {
      match datum {
        SampleDatum::WHCBytes(ref frame_bytes) => {
          //println!("DEBUG: frame: {:?}", frame_bytes.as_slice());
          let frame_len = frame_bytes.bound().len();
          operator
            .stage_shape(batch_counter, frame_bytes.bound());
          operator
            .expose_host_frame_buf(batch_counter)[ .. frame_len]
            .copy_from_slice(frame_bytes.as_slice());
          operator
            .preload_frame(batch_counter);
        }
        _ => unimplemented!(),
      }
      operator.stage_label(batch_counter, &maybe_label.unwrap());
      operator.stage_weight(batch_counter, weight);
      local_counter += 1;
      batch_counter += 1;

      // With a full batch of data, compute a full forward/backward pass.
      assert!(batch_counter <= batch_size);
      if batch_counter == batch_size {
        //operator.next();
        //operator.input_operator().load_frames(batch_size);
        operator.wait_preload_frames(batch_size);
        operator.load_labels(batch_size);
        operator.load_weights(batch_size);
        operator.forward(batch_size, OpPhase::Inference);
        operator.store_output_categories(batch_size);
        let local_loss = operator.store_loss(batch_size);
        let local_correct_count = operator.accuracy_count(batch_size);
        display_acc_correct_count += local_correct_count;
        display_acc_total_count += batch_size;
        display_acc_loss += local_loss;
        batch_counter = 0;
      }
    }
    if batch_counter > 0 {
      let batch_size = batch_counter;
      operator.wait_preload_frames(batch_size);
      operator.load_labels(batch_size);
      operator.load_weights(batch_size);
      operator.forward(batch_size, OpPhase::Inference);
      operator.store_output_categories(batch_size);
      let local_loss = operator.store_loss(batch_size);
      let local_correct_count = operator.accuracy_count(batch_size);
      display_acc_correct_count += local_correct_count;
      display_acc_total_count += batch_size;
      display_acc_loss += local_loss;
      batch_counter = 0;
    }
    assert_eq!(local_counter, epoch_size);

    let lap_time = get_time();
    let elapsed_ms = (lap_time - start_time).num_milliseconds();
    let acc_correct_count = display_acc_correct_count;
    let acc_total_count = display_acc_total_count;
    let accuracy = acc_correct_count as f32 / acc_total_count as f32;
    let avg_loss = display_acc_loss / self.config.display_iters as f32;
    info!("SgdOpt: valid: samples: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
        epoch_size,
        avg_loss,
        accuracy,
        elapsed_ms as f32 * 0.001,
    );
  }
}
