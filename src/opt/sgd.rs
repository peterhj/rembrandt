use data_new::{
  DataIterator,
  SampleDatum, SampleDatumConfig, SampleLabel, SampleLabelConfig, 
};
use operator::{Operator, OpPhase, Regularization};
use operator::worker::{OperatorWorker};

//use array_cuda::device::context::{DeviceCtxRef};

use std::slice::bytes::{copy_memory};
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use time::{get_time};

#[derive(Clone, Copy, Debug)]
pub struct SgdOptConfig {
  pub init_t:         Option<usize>,
  pub minibatch_size: usize,
  pub step_size:      StepSizeSchedule,
  pub momentum:       MomentumStyle,
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

impl StepSizeSchedule {
  pub fn at_iter(&self, t: usize) -> f32 {
    match self {
      &StepSizeSchedule::Constant{step_size} => {
        step_size
      }
      &StepSizeSchedule::DecayOnce{..} => {
        // FIXME(20160330)
        unimplemented!();
      }
      &StepSizeSchedule::Decay{..} => {
        // FIXME(20160330)
        unimplemented!();
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum MomentumStyle {
  Zero,
  Sgd{momentum: f32},
  Nesterov{momentum: f32},
}

impl MomentumStyle {
  pub fn at_iter(&self, _t: usize) -> f32 {
    match self {
      &MomentumStyle::Zero => {
        0.0
      }
      &MomentumStyle::Sgd{momentum} => {
        momentum
      }
      &MomentumStyle::Nesterov{momentum} => {
        momentum
      }
    }
  }

  pub fn is_nesterov(&self) -> bool {
    match self {
      &MomentumStyle::Nesterov{..} => {
        true
      }
      _ => {
        false
      }
    }
  }
}

pub struct OptSharedData {
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
    fence(Ordering::AcqRel);
    self.barrier.wait();
  }
}

pub struct Validation;

impl Validation {
  pub fn validate(&self, shared: Arc<OptSharedData>, sgd_opt_cfg: SgdOptConfig, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, valid_data: &mut DataIterator, operator: &mut OperatorWorker) {
    let batch_size = operator.batch_size();
    let num_workers = operator.num_workers();
    let tid = operator.worker_rank();

    let mut start_time = get_time();
    let mut batch_counter = 0;
    valid_data.each_sample(datum_cfg, label_cfg, &mut |epoch_idx, datum, maybe_label| {
      match (label_cfg, maybe_label) {
        (SampleLabelConfig::Category{num_categories}, Some(&SampleLabel::Category{category})) => {
          assert!(category >= 0);
          assert!(category < num_categories);
        }
        _ => panic!("SgdOpt: unsupported label"),
      }
      match datum {
        &SampleDatum::WHCBytes(ref frame_bytes) => {
          //println!("DEBUG: frame: {:?}", frame_bytes.as_slice());
          copy_memory(
              frame_bytes.as_slice(),
              operator.input_operator().expose_host_frame_buf(batch_counter),
          );
        }
      }
      operator.loss_operator(0).stage_label(batch_counter, maybe_label.unwrap());
      batch_counter += 1;

      if batch_counter == batch_size {
        operator.input_operator().load_frames(batch_size);
        operator.loss_operator(0).load_labels(batch_size);
        operator.forward(batch_size, OpPhase::Inference);
        operator.loss_operator(0).store_output_categories(batch_size);
        let local_correct_count = operator.loss_operator(0).accuracy_count(batch_size);
        shared.acc_correct_count.fetch_add(local_correct_count, Ordering::AcqRel);
        shared.acc_total_count.fetch_add(batch_size, Ordering::AcqRel);
        batch_counter = 0;
      }
    });
    if batch_counter < batch_size {
      let batch_size = batch_counter;
      operator.input_operator().load_frames(batch_size);
      operator.loss_operator(0).load_labels(batch_size);
      operator.forward(batch_size, OpPhase::Inference);
      operator.loss_operator(0).store_output_categories(batch_size);
      let local_correct_count = operator.loss_operator(0).accuracy_count(batch_size);
      shared.acc_correct_count.fetch_add(local_correct_count, Ordering::AcqRel);
      shared.acc_total_count.fetch_add(batch_size, Ordering::AcqRel);
      batch_counter = 0;
    }

    shared.sync();
    if tid == 0 {
      let lap_time = get_time();
      let elapsed_ms = (lap_time - start_time).num_milliseconds();
      start_time = lap_time;
      let acc_correct_count = shared.acc_correct_count.load(Ordering::Acquire);
      let acc_total_count = shared.acc_total_count.load(Ordering::Acquire);
      let accuracy = acc_correct_count as f32 / acc_total_count as f32;
      info!("SgdOpt: sample count: {} valid accuracy: {:.03} elapsed: {:.03} s",
          acc_total_count,
          accuracy,
          elapsed_ms as f32 * 0.001,
      );
      shared.acc_correct_count.store(0, Ordering::Release);
      shared.acc_total_count.store(0, Ordering::Release);
    }
    shared.sync();
  }
}

pub struct SgdOpt {
  shared:   Arc<OptSharedData>,
}

impl SgdOpt {
  pub fn new(shared: Arc<OptSharedData>) -> SgdOpt {
    SgdOpt{
      shared:   shared,
    }
  }

  pub fn train(&self, sgd_opt_cfg: SgdOptConfig, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, train_data: &mut DataIterator, valid_data: &mut DataIterator, operator: &mut OperatorWorker) {
    let batch_size = operator.batch_size();
    let num_workers = operator.num_workers();
    let tid = operator.worker_rank();
    let minibatch_size = sgd_opt_cfg.minibatch_size;
    let local_minibatch_size = minibatch_size / num_workers;
    let local_minibatch_weight = 1.0 / local_minibatch_size as f32;
    let epoch_size = (train_data.max_num_samples() / local_minibatch_size) * local_minibatch_size;

    let shared_seed = operator.shared_seed();
    operator.init_params(shared_seed);

    let mut start_time = get_time();
    let mut epoch_counter = 0;
    let mut iter_counter = 0;
    let mut local_counter = 0;
    loop {
      let mut batch_counter = 0;
      train_data.each_sample(datum_cfg, label_cfg, &mut |epoch_idx, datum, maybe_label| {
        if epoch_idx >= epoch_size {
          return;
        }
        match (label_cfg, maybe_label) {
          (SampleLabelConfig::Category{num_categories}, Some(&SampleLabel::Category{category})) => {
            assert!(category >= 0);
            assert!(category < num_categories);
          }
          _ => panic!("SgdOpt: unsupported label"),
        }
        match datum {
          &SampleDatum::WHCBytes(ref frame_bytes) => {
            //println!("DEBUG: frame: {:?}", frame_bytes.as_slice());
            copy_memory(
                frame_bytes.as_slice(),
                operator.input_operator().expose_host_frame_buf(batch_counter),
            );
          }
        }
        operator.loss_operator(0).stage_label(batch_counter, maybe_label.unwrap());
        operator.loss_operator(0).stage_weight(batch_counter, local_minibatch_weight);
        local_counter += 1;
        epoch_counter = local_counter * num_workers / epoch_size;
        let epoch_offset = local_counter * num_workers % epoch_size;
        batch_counter += 1;

        if batch_counter == batch_size {
          operator.input_operator().load_frames(batch_size);
          operator.loss_operator(0).load_labels(batch_size);
          operator.loss_operator(0).load_weights(batch_size);
          operator.forward(batch_size, OpPhase::Training);
          operator.loss_operator(0).store_output_categories(batch_size);
          operator.backward(batch_size);
          let local_correct_count = operator.loss_operator(0).accuracy_count(batch_size);
          self.shared.acc_correct_count.fetch_add(local_correct_count, Ordering::AcqRel);
          self.shared.acc_total_count.fetch_add(batch_size, Ordering::AcqRel);
          batch_counter = 0;
        }

        if local_counter % local_minibatch_size == 0 {
          let l2_reg_coef = sgd_opt_cfg.l2_reg_coef;
          let step_size = sgd_opt_cfg.step_size.at_iter(iter_counter);
          let momentum = sgd_opt_cfg.momentum.at_iter(iter_counter);
          let nesterov = sgd_opt_cfg.momentum.is_nesterov();
          operator.regularize(Regularization::L2{l2_reg_coef: l2_reg_coef});
          if nesterov {
            operator.update_params(-momentum);
          }
          operator.accumulate_grads(-step_size, momentum);
          //operator.save_params();
          operator.update_params(1.0);
          if num_workers > 1 {
            operator.stage_params();
            operator.sync_params();
          }
          // XXX(20160406): Interestingly, we should use the local update rather
          // than the communicated update with momentum.
          //operator.set_grads_with_params_diff();
          if nesterov {
            operator.update_params(momentum);
          }
          operator.reset();
          iter_counter += 1;

          if iter_counter % sgd_opt_cfg.display_iters == 0 {
            self.shared.sync();
            if tid == 0 {
              let lap_time = get_time();
              let elapsed_ms = (lap_time - start_time).num_milliseconds();
              start_time = lap_time;
              let acc_correct_count = self.shared.acc_correct_count.load(Ordering::Acquire);
              let acc_total_count = self.shared.acc_total_count.load(Ordering::Acquire);
              let accuracy = acc_correct_count as f32 / acc_total_count as f32;
              info!("SgdOpt: iter: {} epoch: {} sample {}/{} step: {} momentum: {} loss: {:.06} train accuracy: {:.03} elapsed: {:.03} s",
                  iter_counter, epoch_counter,
                  epoch_offset, epoch_size,
                  step_size, momentum,
                  0.0, //avg_loss,
                  accuracy,
                  elapsed_ms as f32 * 0.001,
              );
              self.shared.acc_correct_count.store(0, Ordering::Release);
              self.shared.acc_total_count.store(0, Ordering::Release);
            }
            self.shared.sync();
          }

          if iter_counter % sgd_opt_cfg.save_iters == 0 {
          }

          if iter_counter % sgd_opt_cfg.valid_iters == 0 {
            let validation = Validation;
            validation.validate(self.shared.clone(), sgd_opt_cfg, datum_cfg, label_cfg, valid_data, operator);
          }
        }
      });

      //epoch_counter += 1;
    }
  }
}
