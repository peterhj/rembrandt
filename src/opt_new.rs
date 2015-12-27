use arch_new::{AtomicData, Worker, ArchWorker};
use data_new::{SampleDatum, SampleLabel, SampleLabelConfig, DataIterator};
use layer_new::{LossLayer, Phase};

use array_cuda::device::{DeviceCtxRef};

use std::sync::atomic::{AtomicUsize, Ordering};
use time::{get_time};

#[derive(Clone, Copy)]
pub struct SgdOptConfig {
  pub minibatch_size: usize,
  pub learning_rate:  LearningRateSchedule,
  pub momentum:       f32,
  pub l2_reg_coef:    f32,

  pub display_iters:  usize,
  pub valid_iters:    usize,
  pub save_iters:     usize,
}

#[derive(Clone, Copy)]
pub enum LearningRateSchedule {
  Fixed{lr: f32},
  StepOnce{
    lr0: f32, lr0_iters: usize,
    lr_final: f32,
  },
  StepTwice{
    lr0: f32, lr0_iters: usize,
    lr1: f32, lr1_iters: usize,
    lr_final: f32,
  },
}

impl LearningRateSchedule {
  pub fn at_iter(&self, t: usize) -> f32 {
    match *self {
      LearningRateSchedule::Fixed{lr} => lr,
      LearningRateSchedule::StepOnce{lr0, lr0_iters, lr_final} => {
        if t < lr0_iters {
          lr0
        } else {
          lr_final
        }
      }
      LearningRateSchedule::StepTwice{lr0, lr0_iters, lr1, lr1_iters, lr_final} => {
        if t < lr0_iters {
          lr0
        } else if t < lr1_iters {
          lr1
        } else {
          lr_final
        }
      }
    }
  }
}

pub struct SupervisedData {
  pub sample_size:  AtomicUsize,
  pub num_correct:  AtomicUsize,
}

impl SupervisedData {
  pub fn new() -> SupervisedData {
    SupervisedData{
      sample_size:  AtomicUsize::new(0),
      num_correct:  AtomicUsize::new(0),
    }
  }

  pub fn read_accuracy(&self) -> f32 {
    let sample_size = self.sample_size.load(Ordering::SeqCst);
    let num_correct = self.num_correct.load(Ordering::SeqCst);
    num_correct as f32 / sample_size as f32
  }
}

impl AtomicData for SupervisedData {
  fn reset(&self) {
    self.sample_size.store(0, Ordering::SeqCst);
    self.num_correct.store(0, Ordering::SeqCst);
  }

  fn update(&self, batch_size: usize, loss_layer: &mut LossLayer, ctx: &DeviceCtxRef) {
    // TODO(20151220)
    loss_layer.store_labels(batch_size, ctx);
    let batch_correct = loss_layer.count_accuracy(batch_size);
    self.sample_size.fetch_add(batch_size, Ordering::SeqCst);
    self.num_correct.fetch_add(batch_correct, Ordering::SeqCst);
  }
}

pub struct SupervisedResults {
  pub sample_size:  usize,
  pub num_correct:  usize,
  pub accuracy:     f32,
}

pub struct Validation;

impl Validation {
  pub fn validate(&self, minibatch_size: usize, label: SampleLabelConfig, arch: &mut ArchWorker<SupervisedData>, data: &mut DataIterator, ctx: &DeviceCtxRef) -> SupervisedResults {
    let epoch_size = (data.max_num_samples() / minibatch_size) * minibatch_size;
    let batch_size = arch.batch_size();
    assert!(minibatch_size >= batch_size);
    assert_eq!(0, minibatch_size % batch_size);
    let tid = arch.tid();

    let mut batch_idx = 0;
    arch.reset_atomic_data();
    data.each_sample(label, &mut |epoch_idx, datum, maybe_label| {
      if epoch_idx >= epoch_size {
        return;
      }
      if epoch_idx % tid == 0 {
        arch.input_layer().preload_frame(batch_idx, datum, ctx);
        arch.loss_layer().preload_label(batch_idx, maybe_label.unwrap(), ctx);
        batch_idx += 1;
      }
      if batch_idx == batch_size {
        arch.input_layer().load_frames(batch_size, ctx);
        arch.loss_layer().load_labels(batch_size, ctx);
        arch.forward(batch_size, Phase::Training, ctx);
        arch.loss_layer().store_labels(batch_size, ctx);
        arch.update_atomic_data(batch_size, ctx);
        arch.backward(batch_size, 1.0, ctx);
        batch_idx = 0;
      }
    });

    let supervised_data = arch.get_atomic_data();
    arch.sync_workers();
    let sample_size = supervised_data.sample_size.load(Ordering::SeqCst);
    let num_correct = supervised_data.num_correct.load(Ordering::SeqCst);
    let accuracy = (num_correct as f32) / (sample_size as f32);
    SupervisedResults{
      sample_size:  sample_size,
      num_correct:  num_correct,
      accuracy:     accuracy,
    }
  }
}

pub struct SgdOptimization;

impl SgdOptimization {
  pub fn train(&self, sgd_opt_cfg: SgdOptConfig, label_cfg: SampleLabelConfig, arch: &mut ArchWorker<SupervisedData>, train_data: &mut DataIterator, valid_data: &mut DataIterator, ctx: &DeviceCtxRef) {
    let epoch_size = (train_data.max_num_samples() / sgd_opt_cfg.minibatch_size) * sgd_opt_cfg.minibatch_size;
    let batch_size = arch.batch_size();
    assert!(sgd_opt_cfg.minibatch_size >= batch_size);
    assert_eq!(0, sgd_opt_cfg.minibatch_size % batch_size);
    let tid = arch.tid();
    let num_workers = arch.num_workers();

    let mut start_time = get_time();

    arch.initialize_layer_params(ctx);
    arch.reset_gradients(0.0, ctx);
    arch.reset_atomic_data();

    let mut idx = 0;
    let mut batch_idx = 0;
    let mut t = 0;
    let mut epoch = 0;
    loop {
      train_data.each_sample(label_cfg, &mut |epoch_idx, datum, maybe_label| {
        if epoch_idx >= epoch_size {
          return;
        }
        arch.input_layer().preload_frame(batch_idx, datum, ctx);
        arch.loss_layer().preload_label(batch_idx, maybe_label.unwrap(), ctx);
        batch_idx += 1;
        idx += 1;
        if batch_idx == batch_size {
          arch.input_layer().load_frames(batch_size, ctx);
          arch.loss_layer().load_labels(batch_size, ctx);
          arch.forward(batch_size, Phase::Training, ctx);
          arch.backward(batch_size, 1.0, ctx);
          arch.loss_layer().store_labels(batch_size, ctx);
          arch.update_atomic_data(batch_size, ctx);
          batch_idx = 0;
        }
        if idx % sgd_opt_cfg.minibatch_size == 0 {
          // TODO(20151220): reduce gradients.
          arch.allreduce_gradients(ctx);
          let descent_scale = sgd_opt_cfg.learning_rate.at_iter(t) / (sgd_opt_cfg.minibatch_size as f32);
          //println!("DEBUG: descend: scale {:.6}", descent_scale);
          arch.descend(descent_scale, sgd_opt_cfg.l2_reg_coef, ctx);
          arch.reset_gradients(sgd_opt_cfg.momentum, ctx);
          t += 1;
          if t % sgd_opt_cfg.display_iters == 0 {
            if tid == 0 {
              let lap_time = get_time();
              let elapsed_ms = (lap_time - start_time).num_milliseconds();
              start_time = lap_time;
              let supervised_data = arch.get_atomic_data();
              let accuracy = supervised_data.read_accuracy();
              println!("DEBUG: iter: {} accuracy: {:.03} elapsed: {:.3} s", t, accuracy, elapsed_ms as f32 * 0.001);
              arch.reset_atomic_data();
            }
          }
          if t % sgd_opt_cfg.save_iters == 0 {
            arch.save_layer_params(t, ctx);
          }
          if t % sgd_opt_cfg.valid_iters == 0 {
            // TODO(20151220)
            /*let validation = Validation;
            let results = validation.validate(sgd_opt_cfg.minibatch_size, label_cfg, arch, valid_data, ctx);*/
          }
        }
      });
      epoch += 1;
    }
  }
}
