use data_new::{
  DataIterator,
  SampleDatum, SampleDatumConfig, SampleLabel, SampleLabelConfig, 
};
use operator::{Operator, OpPhase, Regularization};
use operator::worker::{OperatorWorker};

//use array_cuda::device::context::{DeviceCtxRef};

use std::path::{PathBuf};
use std::slice::bytes::{copy_memory};
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use time::{get_time};

#[derive(Clone, RustcDecodable, RustcEncodable, Debug)]
pub struct SgdOptConfig {
  pub init_t:         Option<usize>,
  pub minibatch_size: usize,
  pub step_size:      StepSizeSchedule,
  pub momentum:       Momentum,
  pub l2_reg_coef:    f32,
  pub sync_order:     SyncOrder,

  pub display_iters:  usize,
  pub save_iters:     usize,
  pub valid_iters:    usize,
  pub checkpoint_dir: PathBuf,
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
      &StepSizeSchedule::Decay{..} => {
        // FIXME(20160330)
        unimplemented!();
      }
    }
  }
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
  SyncUpdatesThenStep,
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
    self.barrier.wait();
    fence(Ordering::AcqRel);
  }
}

pub struct SgdOpt {
  shared:   Arc<OptSharedData>,
  local_accum_loss: f32,
}

impl SgdOpt {
  pub fn new(shared: Arc<OptSharedData>) -> SgdOpt {
    SgdOpt{
      shared:   shared,
      local_accum_loss: 0.0,
    }
  }

  pub fn train(&mut self, sgd_opt_cfg: &SgdOptConfig, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, train_data: &mut DataIterator, valid_data: &mut DataIterator, operator: &mut OperatorWorker) {
    assert_eq!(0, sgd_opt_cfg.save_iters % sgd_opt_cfg.display_iters);
    assert_eq!(0, sgd_opt_cfg.valid_iters % sgd_opt_cfg.save_iters);

    let batch_size = operator.batch_size();
    let num_workers = operator.num_workers();
    let rank = operator.worker_rank();
    let minibatch_size = sgd_opt_cfg.minibatch_size;
    let minibatch_weight = 1.0 / minibatch_size as f32;
    let epoch_size = (train_data.max_num_samples() / minibatch_size) * minibatch_size;
    //let local_minibatch_size = minibatch_size / num_workers;
    //let local_minibatch_weight = 1.0 / local_minibatch_size as f32;
    //let epoch_size = (train_data.max_num_samples() / local_minibatch_size) * local_minibatch_size;

    let shared_seed = operator.shared_seed();
    operator.init_params(shared_seed);
    operator.reset();

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
        operator.loss_operator(0).stage_weight(batch_counter, minibatch_weight);
        local_counter += 1;
        //epoch_counter = local_counter * num_workers / epoch_size;
        //let epoch_offset = local_counter * num_workers % epoch_size;
        epoch_counter = local_counter / epoch_size;
        let epoch_offset = local_counter % epoch_size;
        batch_counter += 1;

        // With a full batch of data, compute a full forward/backward pass.
        assert!(batch_counter <= batch_size);
        if batch_counter == batch_size {
          operator.next();
          operator.input_operator().load_frames(batch_size);
          operator.loss_operator(0).load_labels(batch_size);
          operator.loss_operator(0).load_weights(batch_size);
          operator.forward(batch_size, OpPhase::Training{t: iter_counter});
          operator.loss_operator(0).store_output_categories(batch_size);
          let local_loss = operator.loss_operator(0).store_loss(batch_size);
          operator.backward(batch_size);
          let local_correct_count = operator.loss_operator(0).accuracy_count(batch_size);
          self.shared.acc_correct_count.fetch_add(local_correct_count, Ordering::AcqRel);
          self.shared.acc_total_count.fetch_add(batch_size, Ordering::AcqRel);
          self.local_accum_loss += local_loss;
          batch_counter = 0;
        }

        if local_counter % minibatch_size == 0 {
          let l2_reg_coef = sgd_opt_cfg.l2_reg_coef;
          let step_size = sgd_opt_cfg.step_size.at_iter(iter_counter);

          // Apply regularization to the current gradient.
          operator.regularize(Regularization::L2{l2_reg_coef: l2_reg_coef});

          // If we are using the standard Nesterov update, unapply the extra
          // momentum.
          match sgd_opt_cfg.momentum {
            Momentum::UpdateNesterov{mu} => {
              if iter_counter > 0 {
                operator.update_params(-mu);
              }
            }
            _ => {}
          }

          // Depending on the order of the gradient step, apply different
          // update rules.
          match sgd_opt_cfg.sync_order {
            SyncOrder::StepThenSyncParams => {
              // Compute the update, possibly with momentum.
              match sgd_opt_cfg.momentum {
                Momentum::Zero => {
                  operator.accumulate_grads(1.0, 0.0);
                  operator.update_params(-step_size);
                }
                Momentum::Update{mu} |
                Momentum::UpdateNesterov{mu} => {
                  operator.accumulate_grads(-step_size, mu);
                  operator.update_params(1.0);
                }
                // XXX(20160422): These are the Torch `optim.sgd`-style update;
                // see: <https://github.com/torch/optim/blob/master/sgd.lua>.
                Momentum::Gradient{mu} => {
                  operator.accumulate_grads(1.0, mu);
                  operator.update_params(-step_size);
                }
                Momentum::GradientNesterov{mu} => {
                  operator.accumulate_grads(1.0, mu);
                  operator.update_params2(-step_size, -step_size * mu);
                }
              }

              // Communicate the parameters.
              operator.sync_params_v2();
            }

            SyncOrder::SyncUpdatesThenStep => {
              // FIXME(20160424): necessary for sync all-reduce.
              unimplemented!();
            }
          }

          // If we are using the standard Nesterov update, apply some extra
          // momentum.
          // XXX(20160406): Interestingly, we should use the local update rather
          // than the communicated update with momentum.
          //operator.set_grads_with_params_diff();
          match sgd_opt_cfg.momentum {
            Momentum::UpdateNesterov{mu} => {
              operator.update_params(mu);
            }
            _ => {}
          }

          operator.reset();
          iter_counter += 1;
          //info!("DEBUG: rank: {} post iter: {}", rank, iter_counter);

          if iter_counter % sgd_opt_cfg.display_iters == 0 {
            self.shared.sync();
            let lap_time = get_time();
            if rank == 0 {
              let elapsed_ms = (lap_time - start_time).num_milliseconds();
              let acc_correct_count = self.shared.acc_correct_count.load(Ordering::Acquire);
              let acc_total_count = self.shared.acc_total_count.load(Ordering::Acquire);
              let accuracy = acc_correct_count as f32 / acc_total_count as f32;
              let avg_loss = self.local_accum_loss / sgd_opt_cfg.display_iters as f32;
              info!("SgdOpt: train: iter: {} epoch: {} sample: {}/{} step: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
                  iter_counter, epoch_counter,
                  epoch_offset, epoch_size,
                  step_size,
                  avg_loss,
                  accuracy,
                  elapsed_ms as f32 * 0.001,
              );
              self.shared.acc_correct_count.store(0, Ordering::Release);
              self.shared.acc_total_count.store(0, Ordering::Release);
              self.local_accum_loss = 0.0;
            }
            self.shared.sync();
            start_time = get_time();
          }

          if iter_counter % sgd_opt_cfg.save_iters == 0 {
            if rank == 0 {
              //info!("DEBUG: signal checkpoint...");
              operator.signal_checkpoint();
            }
          }

          /*if iter_counter % sgd_opt_cfg.valid_iters == 0 {
            info!("DEBUG: rank: {} validate, iter: {}", rank, iter_counter);
            self.validate(&sgd_opt_cfg, datum_cfg, label_cfg, valid_data, operator);
            start_time = get_time();
          }*/

          if operator.wait_checkpoint() {
            //info!("DEBUG: rank: {} signaled checkpoint! iter: {}", rank, iter_counter);
            operator.save_params();

            //info!("DEBUG: rank: {} exact sync params, iter: {}", rank, iter_counter);
            operator.exact_sync_params();

            if rank == 0 {
              //info!("DEBUG: rank: {} checkpoint params, iter: {}", rank, iter_counter);
              operator.checkpoint_params(iter_counter, &sgd_opt_cfg.checkpoint_dir);
            }

            //info!("DEBUG: rank: {} validate, iter: {}", rank, iter_counter);
            self.validate(&sgd_opt_cfg, datum_cfg, label_cfg, valid_data, operator);

            //info!("DEBUG: rank: {} restore params, iter: {}", rank, iter_counter);
            operator.restore_params();
            start_time = get_time();
            //info!("DEBUG: rank: {} done checkpoint, iter: {}", rank, iter_counter);
          }
        }
      });

      //epoch_counter += 1;
    }
  }

  pub fn validate(&mut self, sgd_opt_cfg: &SgdOptConfig, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, valid_data: &mut DataIterator, operator: &mut OperatorWorker) {
    let batch_size = operator.batch_size();
    let num_workers = operator.num_workers();
    let rank = operator.worker_rank();

    let num_samples = valid_data.max_num_samples();

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
      operator.loss_operator(0).stage_weight(batch_counter, 1.0 / num_samples as f32);
      batch_counter += 1;

      if batch_counter == batch_size {
        operator.input_operator().load_frames(batch_size);
        operator.loss_operator(0).load_labels(batch_size);
        operator.loss_operator(0).load_weights(batch_size);
        operator.forward(batch_size, OpPhase::Inference);
        operator.loss_operator(0).store_output_categories(batch_size);
        let local_correct_count = operator.loss_operator(0).accuracy_count(batch_size);
        let local_loss = operator.loss_operator(0).store_loss(batch_size);
        self.shared.acc_correct_count.fetch_add(local_correct_count, Ordering::AcqRel);
        self.shared.acc_total_count.fetch_add(batch_size, Ordering::AcqRel);
        self.local_accum_loss += local_loss;
        batch_counter = 0;
      }
    });
    if batch_counter > 0 && batch_counter < batch_size {
      let batch_size = batch_counter;
      operator.input_operator().load_frames(batch_size);
      operator.loss_operator(0).load_labels(batch_size);
      operator.loss_operator(0).load_weights(batch_size);
      operator.forward(batch_size, OpPhase::Inference);
      operator.loss_operator(0).store_output_categories(batch_size);
      let local_correct_count = operator.loss_operator(0).accuracy_count(batch_size);
      let local_loss = operator.loss_operator(0).store_loss(batch_size);
      self.shared.acc_correct_count.fetch_add(local_correct_count, Ordering::AcqRel);
      self.shared.acc_total_count.fetch_add(batch_size, Ordering::AcqRel);
      self.local_accum_loss += local_loss;
      batch_counter = 0;
    }

    self.shared.sync();
    if rank == 0 {
      let lap_time = get_time();
      let elapsed_ms = (lap_time - start_time).num_milliseconds();
      start_time = lap_time;
      let acc_correct_count = self.shared.acc_correct_count.load(Ordering::Acquire);
      let acc_total_count = self.shared.acc_total_count.load(Ordering::Acquire);
      let accuracy = acc_correct_count as f32 / acc_total_count as f32;
      let avg_loss = self.local_accum_loss;
      info!("SgdOpt: valid: sample count: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
          acc_total_count,
          avg_loss,
          accuracy,
          elapsed_ms as f32 * 0.001,
      );
      self.shared.acc_correct_count.store(0, Ordering::Release);
      self.shared.acc_total_count.store(0, Ordering::Release);
      self.local_accum_loss = 0.0;
    }
    self.shared.sync();
  }
}
