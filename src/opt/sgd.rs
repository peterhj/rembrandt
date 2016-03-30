use data_new::{
  DataIterator,
  SampleDatum, SampleDatumConfig, SampleLabel, SampleLabelConfig, 
};
use operator::{Operator, OpPhase};
use operator::arch::{OperatorWorker};

//use array_cuda::device::context::{DeviceCtxRef};

use std::slice::bytes::{copy_memory};
use time::{get_time};

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

impl StepSizeSchedule {
  pub fn at_iter(&self, t: usize) -> f32 {
    unimplemented!();
  }
}

#[derive(Clone, Copy, Debug)]
pub enum MomentumSchedule {
  Constant{momentum: f32},
}

impl MomentumSchedule {
  pub fn at_iter(&self, t: usize) -> f32 {
    unimplemented!();
  }
}

pub struct SgdOpt;

impl SgdOpt {
  pub fn train(&self, sgd_opt_cfg: SgdOptConfig, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, train_data: &mut DataIterator, valid_data: &mut DataIterator, operator: &mut OperatorWorker) {
    let batch_size = operator.batch_size();
    let num_workers = operator.num_workers();
    let tid = operator.tid();
    let minibatch_size = sgd_opt_cfg.minibatch_size;
    let minibatch_weight = 1.0 / minibatch_size as f32;
    let local_minibatch_size = minibatch_size / num_workers;
    let epoch_size = (train_data.max_num_samples() / local_minibatch_size) * local_minibatch_size;

    let shared_seed = operator.shared_seed();
    operator.init_params(shared_seed);

    let mut start_time = get_time();
    let mut epoch_counter = 0;
    let mut iter_counter = 0;
    loop {
      let mut local_idx = 0;
      let mut batch_idx = 0;
      let mut acc_correct_count = 0;
      let mut acc_total_count = 0;
      train_data.each_sample(datum_cfg, label_cfg, &mut |epoch_idx, datum, maybe_label| {
        match (label_cfg, maybe_label) {
          (SampleLabelConfig::Category{num_categories}, Some(&SampleLabel::Category{category})) => {
            assert!(category >= 0);
            assert!(category < num_categories);
          }
          _ => {}
        }
        match datum {
          &SampleDatum::WHCBytes(ref arr) => {
            copy_memory(
                arr.as_slice(),
                operator.input_operator().expose_host_frame_buf(batch_idx),
            );
          }
        }
        operator.loss_operator(0).stage_label(batch_idx, maybe_label.unwrap());
        operator.loss_operator(0).stage_weight(batch_idx, minibatch_weight);
        local_idx += 1;
        batch_idx += 1;

        if batch_idx == batch_size {
          operator.input_operator().load_frames(batch_size);
          operator.loss_operator(0).load_labels(batch_size);
          operator.loss_operator(0).load_weights(batch_size);
          operator.forward(batch_size, OpPhase::Training);
          operator.backward(batch_size);
          operator.loss_operator(0).store_output_categories(batch_size);
          acc_correct_count += operator.loss_operator(0).accuracy_count(batch_size);
          acc_total_count += batch_size;
          batch_idx = 0;
        }

        if local_idx % local_minibatch_size == 0 {
          let step_size = sgd_opt_cfg.step_size.at_iter(iter_counter);
          let momentum = sgd_opt_cfg.momentum.at_iter(iter_counter);
          operator.update_params(step_size, sgd_opt_cfg.l2_reg_coef);
          operator.sync_params();
          operator.reset_grads(momentum);
          iter_counter += 1;

          if iter_counter % sgd_opt_cfg.display_iters == 0 {
            if tid == 0 {
              let lap_time = get_time();
              let elapsed_ms = (lap_time - start_time).num_milliseconds();
              start_time = lap_time;
              let accuracy = acc_correct_count as f32 / acc_total_count as f32;
              info!("SgdOpt: epoch: {} iter: {} sample {}/{} step: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
                  epoch_counter, iter_counter,
                  local_idx * num_workers, epoch_size,
                  step_size,
                  0.0, //avg_loss,
                  accuracy,
                  elapsed_ms as f32 * 0.001,
              );
            }
          }

          if iter_counter % sgd_opt_cfg.save_iters == 0 {
          }

          if iter_counter % sgd_opt_cfg.valid_iters == 0 {
          }
        }
      });
    }
  }
}
