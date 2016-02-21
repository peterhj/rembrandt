use arch_new::{AtomicData, Worker, ArchWorker};
use data_new::{SampleDatum, SampleDatumConfig, SampleLabel, SampleLabelConfig, DataIterator};
use layer_new::{LossLayer, Phase};

use array_cuda::device::{DeviceCtxRef};

use std::mem::{transmute};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use time::{get_time};

#[derive(Clone, Copy, Debug)]
pub struct SgdOptConfig {
  pub init_t:         usize,
  pub minibatch_size: usize,
  pub step_size:      StepSizeSchedule,
  pub momentum:       f32,
  pub l2_reg_coef:    f32,

  pub display_iters:  usize,
  pub valid_iters:    usize,
  pub save_iters:     usize,
}

pub type LearningRateSchedule = StepSizeSchedule;

#[derive(Clone, Copy, Debug)]
pub struct NewBobStepSizeState {
  curr_step:    f32,
  curr_loss:    f32,
  prev_loss:    f32,
  prev_epoch:   usize,
}

impl NewBobStepSizeState {
  pub fn new() -> NewBobStepSizeState {
    // FIXME(20160207)
    //NewBobStepSizeState{
    //}
    unimplemented!();
  }
}

#[derive(Clone, Copy, Debug)]
pub enum StepSizeSchedule {
  // XXX(20160207): old variants.
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

  // XXX(20160207): new variants.
  Constant{step: f32},
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
  NewBob{
    init_step:      f32,
    epoch_iters:    usize,
    state:          NewBobStepSizeState,
  },
}

impl StepSizeSchedule {
  pub fn at_iter(&mut self, t: usize) -> f32 {
    match *self {
      StepSizeSchedule::Fixed{lr} => lr,
      StepSizeSchedule::StepOnce{lr0, lr0_iters, lr_final} => {
        if t < lr0_iters {
          lr0
        } else {
          lr_final
        }
      }
      StepSizeSchedule::StepTwice{lr0, lr0_iters, lr1, lr1_iters, lr_final} => {
        if t < lr0_iters {
          lr0
        } else if t < lr1_iters {
          lr1
        } else {
          lr_final
        }
      }
      StepSizeSchedule::Constant{step} => step,
      StepSizeSchedule::DecayOnce{step0, step0_iters, final_step} => {
        if t < step0_iters {
          step0
        } else {
          final_step
        }
      }
      StepSizeSchedule::Decay{init_step, decay_rate, decay_iters} => {
        let p = (t / decay_iters) as i32;
        init_step * decay_rate.powi(p)
      }
      StepSizeSchedule::NewBob{..} => {
        // FIXME(20160207)
        unimplemented!();
      }
    }
  }
}

pub struct SupervisedData {
  pub sample_size:  AtomicUsize,
  pub num_correct:  AtomicUsize,
  pub loss:         AtomicUsize,
}

impl SupervisedData {
  pub fn new() -> SupervisedData {
    SupervisedData{
      sample_size:  AtomicUsize::new(0),
      num_correct:  AtomicUsize::new(0),
      loss:         AtomicUsize::new(0),
    }
  }

  pub fn read_accuracy(&self) -> (f32, usize, usize) {
    let sample_size = self.sample_size.load(Ordering::SeqCst);
    let num_correct = self.num_correct.load(Ordering::SeqCst);
    (num_correct as f32 / sample_size as f32, num_correct, sample_size)
  }

  pub fn reset_loss(&self) {
    self.loss.store(0, Ordering::Release);
  }

  pub fn add_loss(&self, local_loss: f32) {
    loop {
      let prev_x = self.loss.load(Ordering::Acquire);
      let prev_loss: f64 = unsafe { transmute(prev_x) };
      let next_x: usize = unsafe { transmute(prev_loss + local_loss as f64) };
      let curr_x = self.loss.compare_and_swap(prev_x, next_x, Ordering::AcqRel);
      if prev_x == curr_x {
        return;
      }
    }
  }

  pub fn read_loss(&self) -> f32 {
    let uint_loss = self.loss.load(Ordering::Acquire);
    let float_loss: f64 = unsafe { transmute(uint_loss) };
    float_loss as f32
  }
}

impl AtomicData for SupervisedData {
  fn reset(&self) {
    self.sample_size.store(0, Ordering::SeqCst);
    self.num_correct.store(0, Ordering::SeqCst);
  }

  fn update(&self, batch_size: usize, loss_layer: &mut LossLayer, phase: Phase, ctx: &DeviceCtxRef) {
    // TODO(20151220)
    loss_layer.store_labels(batch_size, phase, ctx);
    let (batch_correct, batch_total) = loss_layer.count_accuracy(batch_size, phase);
    self.sample_size.fetch_add(batch_total, Ordering::SeqCst);
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
  pub fn validate(&self, minibatch_size: usize, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, arch: &mut ArchWorker<SupervisedData>, data: &mut DataIterator, ctx: &DeviceCtxRef) -> SupervisedResults {
    let epoch_size = (data.max_num_samples() / minibatch_size) * minibatch_size;
    let batch_size = arch.batch_size();
    assert!(minibatch_size >= batch_size);
    assert_eq!(0, minibatch_size % batch_size);
    let tid = arch.tid();

    let start_time = get_time();

    let mut batch_idx = 0;
    arch.sync_workers();
    arch.reset_atomic_data();
    data.each_sample(datum_cfg, label_cfg, &mut |epoch_idx, datum, maybe_label| {
      if epoch_idx >= epoch_size {
        return;
      }
      arch.input_layer().preload_frame(batch_idx, datum, ctx);
      arch.loss_layer().preload_label(batch_idx, maybe_label.unwrap(), Phase::Inference);
      batch_idx += 1;
      if batch_idx == batch_size {
        arch.input_layer().load_frames(batch_size, ctx);
        arch.loss_layer().load_labels(batch_size, ctx);
        arch.forward(batch_size, Phase::Inference, ctx);
        arch.loss_layer().accumulate_loss(batch_size, ctx);
        arch.loss_layer().store_labels(batch_size, Phase::Inference, ctx);
        arch.update_atomic_data(batch_size, Phase::Inference, ctx);
        batch_idx = 0;
      }
    });

    arch.loss_layer().store_loss(ctx);
    let local_loss = arch.loss_layer().get_loss();
    arch.loss_layer().reset_loss(batch_size, ctx);
    arch.get_atomic_data().add_loss(local_loss);

    arch.sync_workers();
    fence(Ordering::SeqCst);

    let lap_time = get_time();
    let elapsed_ms = (lap_time - start_time).num_milliseconds();

    let supervised_data = arch.get_atomic_data();
    let (accuracy, num_correct, sample_size) = supervised_data.read_accuracy();
    if tid == 0 {
      // FIXME(20160209): assuming that validation epoch size is for a partition.
      let num_workers = arch.num_workers();
      let total_loss = supervised_data.read_loss();
      let avg_loss = total_loss / (epoch_size * num_workers) as f32;
      info!("sgd: validation: samples: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
          num_workers * epoch_size, avg_loss, accuracy, elapsed_ms as f32 * 0.001);
      supervised_data.reset_loss();
      arch.reset_atomic_data();
    }
    arch.sync_workers();
    fence(Ordering::SeqCst);

    SupervisedResults{
      sample_size:  sample_size,
      num_correct:  num_correct,
      accuracy:     accuracy,
    }
  }
}

pub struct SgdOptimization;

impl SgdOptimization {
  pub fn train(&self, mut sgd_opt_cfg: SgdOptConfig, datum_cfg: SampleDatumConfig, train_label_cfg: SampleLabelConfig, valid_label_cfg: SampleLabelConfig, arch: &mut ArchWorker<SupervisedData>, train_data: &mut DataIterator, valid_data: &mut DataIterator, ctx: &DeviceCtxRef) {
    let tid = arch.tid();
    let num_workers = arch.num_workers();
    let minibatch_size = sgd_opt_cfg.minibatch_size;
    assert_eq!(0, minibatch_size % num_workers);
    let local_minibatch_size = minibatch_size / num_workers;
    let epoch_size = (train_data.max_num_samples() / minibatch_size) * minibatch_size;
    let local_epoch_size = epoch_size / num_workers;
    let batch_size = arch.batch_size();
    assert!(minibatch_size >= batch_size);
    assert_eq!(0, minibatch_size % batch_size);
    assert_eq!(batch_size * num_workers, minibatch_size);

    info!("sgd: tid: {}/{} batch size: {} epoch size: {}",
        tid, num_workers, batch_size, epoch_size);

    let mut start_time = get_time();

    if sgd_opt_cfg.init_t == 0 {
      arch.initialize_layer_params(ctx);
      if tid == 0 {
        arch.save_layer_params(0, ctx);
      }
    } else {
      // FIXME(20160127): load specific iteration of params.
      arch.load_layer_params(None, ctx);
    }
    arch.reset_gradients(0.0, ctx);
    arch.loss_layer().reset_loss(batch_size, ctx);
    arch.reset_atomic_data();

    let mut local_idx = sgd_opt_cfg.init_t * local_minibatch_size;
    let mut batch_idx = 0;
    let mut t = sgd_opt_cfg.init_t;
    //let mut t = 0;
    let mut epoch = 0;
    loop {
      //for (epoch_idx, (datum, maybe_label)) in train_data.iter(datum_cfg, label_cfg).take(local_epoch_size).enumerate() {
      train_data.each_sample(datum_cfg, train_label_cfg, &mut |epoch_idx, datum, maybe_label| {
        //if epoch_idx >= local_epoch_size {
        if epoch_idx >= epoch_size {
          //break;
          return;
        }
        match (train_label_cfg, maybe_label) {
          (SampleLabelConfig::Category{num_categories}, Some(&SampleLabel::Category{category})) => {
            assert!(category >= 0);
            assert!(category < num_categories);
          }
          _ => {}
        }
        arch.input_layer().preload_frame(batch_idx, datum, ctx);
        arch.loss_layer().preload_label(batch_idx, maybe_label.unwrap(), Phase::Training);
        batch_idx += 1;
        local_idx += 1;

        if batch_idx == batch_size {
          arch.input_layer().load_frames(batch_size, ctx);
          arch.loss_layer().load_labels(batch_size, ctx);
          arch.forward(batch_size, Phase::Training, ctx);
          arch.backward(batch_size, 1.0, ctx);
          arch.loss_layer().accumulate_loss(batch_size, ctx);
          arch.loss_layer().store_labels(batch_size, Phase::Training, ctx);
          arch.update_atomic_data(batch_size, Phase::Training, ctx);
          batch_idx = 0;
        }

        //if idx % sgd_opt_cfg.minibatch_size == 0 {
        if local_idx % local_minibatch_size == 0 {
          arch.dev_allreduce_sum_gradients(ctx);
          let descent_scale = sgd_opt_cfg.step_size.at_iter(t) / (sgd_opt_cfg.minibatch_size as f32);
          arch.descend(descent_scale, sgd_opt_cfg.l2_reg_coef, ctx);
          arch.reset_gradients(sgd_opt_cfg.momentum, ctx);
          let step_size = sgd_opt_cfg.step_size.at_iter(t);
          t += 1;
          epoch = local_idx / local_epoch_size;
          if t % sgd_opt_cfg.display_iters == 0 {
            // FIXME(20160220: should reduce local losses; AtomicFloat would be useful here.
            arch.loss_layer().store_loss(ctx);
            let local_loss = arch.loss_layer().get_loss();
            arch.loss_layer().reset_loss(batch_size, ctx);
            arch.get_atomic_data().add_loss(local_loss);

            arch.sync_workers();
            fence(Ordering::SeqCst);

            if tid == 0 {
              let lap_time = get_time();
              let elapsed_ms = (lap_time - start_time).num_milliseconds();
              start_time = lap_time;

              let total_loss = {
                let supervised_data = arch.get_atomic_data();
                supervised_data.read_loss()
              };
              let avg_loss = total_loss / (minibatch_size * sgd_opt_cfg.display_iters) as f32;

              let (accuracy, _, _) = {
                let supervised_data = arch.get_atomic_data();
                supervised_data.read_accuracy()
              };

              info!("sgd: epoch: {} iter: {} sample {}/{} step: {} loss: {:.06} accuracy: {:.03} elapsed: {:.3} s",
                  epoch, t, local_idx * num_workers, epoch_size, step_size, avg_loss, accuracy, elapsed_ms as f32 * 0.001);

              arch.get_atomic_data().reset_loss();
              arch.reset_atomic_data();
            }
            arch.sync_workers();
            fence(Ordering::SeqCst);
          }

          if t % sgd_opt_cfg.save_iters == 0 {
            // FIXME(20160207): Check integrity of layer params (all workers
            // should agree on their layer param values).
            //arch.sync_workers();
            if tid == 0 {
              arch.save_layer_params(t, ctx);
            }
            arch.sync_workers();
          }

          if t % sgd_opt_cfg.valid_iters == 0 {
            let validation = Validation;
            let _ = validation.validate(sgd_opt_cfg.minibatch_size, datum_cfg, valid_label_cfg, arch, valid_data, ctx);
          }
        }
      });
      //epoch += 1;
    }
  }
}
