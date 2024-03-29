use data_new::{
  DataIterator,
  SampleDatum, SampleDatumConfig, SampleLabel, SampleLabelConfig, 
};
use operator::{Operator, OpPhase, Regularization};
use operator::worker::{OperatorWorker};

//use array_cuda::device::context::{DeviceCtxRef};

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{Write, BufWriter};
use std::path::{PathBuf};
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use time::{get_time};

//pub mod async;
pub mod new;

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

/*pub struct SgdOpt {
  config:   SgdOptConfig,
  rank:     Option<usize>,
  shared:   Arc<OptSharedData>,

  elapsed_checkpoint_iters: usize,
  local_step_size:  f32,
  local_accum_loss: f32,
  local_log_file:   BufWriter<File>,
}

impl SgdOpt {
  pub fn new(config: SgdOptConfig, rank: Option<usize>, shared: Arc<OptSharedData>) -> SgdOpt {
    create_dir_all(&config.checkpoint_dir).ok();
    let mut local_log_path = config.checkpoint_dir.clone();
    match rank {
      None => {
        local_log_path.push("trace_sgd.log");
      }
      Some(rank) => {
        local_log_path.push(&format!("trace_sgd.{}.log", rank));
      }
    }
    let local_log_file = match OpenOptions::new()
      .read(true).write(true).create(true).truncate(true)
      .open(&local_log_path)
    {
      Err(e) => panic!("SgdOpt: failed to open log file: {:?}", e),
      Ok(file) => file,
    };
    let init_step_size = config.step_size.at_iter(0);
    let mut writer = BufWriter::new(local_log_file);
    writeln!(&mut writer, "t,event,loss,error,elapsed").unwrap();
    writer.flush().unwrap();
    SgdOpt{
      config:   config,
      rank:     rank,
      shared:   shared,
      elapsed_checkpoint_iters: 0,
      local_step_size:  init_step_size,
      local_accum_loss: 0.0,
      local_log_file:   writer,
    }
  }

  pub fn train(&mut self, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, train_data: &mut DataIterator, valid_data: &mut DataIterator, operator: &mut OperatorWorker) {
    //assert_eq!(0, self.config.save_iters % self.config.display_iters);
    //assert_eq!(0, self.config.valid_iters % self.config.save_iters);
    assert_eq!(0, self.config.checkpoint_iters % self.config.display_iters);

    let batch_size = operator.batch_size();
    let num_workers = operator.num_workers();
    let rank = operator.worker_rank();
    let minibatch_size = self.config.minibatch_size;
    let minibatch_weight = 1.0 / minibatch_size as f32;
    let epoch_size = (train_data.max_num_samples() / minibatch_size) * minibatch_size;
    //let local_minibatch_size = minibatch_size / num_workers;
    //let local_minibatch_weight = 1.0 / local_minibatch_size as f32;
    //let epoch_size = (train_data.max_num_samples() / local_minibatch_size) * local_minibatch_size;

    match self.config.init_t {
      None => {
        let shared_seed = operator.shared_seed();
        operator.init_params(shared_seed);
      }
      Some(t) => {
        operator.rollback_params(Some(t), &self.config.checkpoint_dir);
      }
    }
    operator.reset();

    // Do an initial (one-way) sync (necessary for parameter servers).
    operator.first_one_way_sync_params();

    let mut start_time = get_time();
    let mut minibatch_start_time = start_time;

    let mut minibatch_acc_correct_count = 0;
    let mut minibatch_acc_total_count = 0;
    let mut minibatch_acc_loss = 0.0;

    let mut epoch_counter = 0;
    let mut iter_counter = self.config.init_t.unwrap_or(0);
    let mut local_counter = 0;
    let mut batch_counter = 0;

    loop {
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
            operator.input_operator().expose_host_frame_buf(batch_counter)
              .copy_from_slice(frame_bytes.as_slice());
          }
          _ => unimplemented!(),
        }
        operator.loss_operator(0).stage_label(batch_counter, maybe_label.unwrap());
        operator.loss_operator(0).stage_weight(batch_counter, minibatch_weight);
        local_counter += 1;
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
          minibatch_acc_correct_count += local_correct_count;
          minibatch_acc_total_count += batch_size;
          minibatch_acc_loss += local_loss;
          batch_counter = 0;
        }

        if local_counter % minibatch_size == 0 {
          let l2_reg_coef = self.config.l2_reg_coef;
          //let step_size = self.config.step_size.at_iter(iter_counter);
          let step_size = self.local_step_size;

          // Apply regularization to the current gradient.
          operator.regularize(Regularization::L2{l2_reg_coef: l2_reg_coef});

          // If we are using the standard Nesterov update, unapply the extra
          // momentum.
          match self.config.momentum {
            Momentum::UpdateNesterov{mu} => {
              if iter_counter > 0 {
                operator.update_params(-mu);
              }
            }
            _ => {}
          }

          // Depending on the order of the gradient step, apply different
          // update rules.
          match self.config.sync_order {
            SyncOrder::StepThenSyncParams => {
              // Compute the update, possibly with momentum.
              match self.config.momentum {
                Momentum::Zero                  => operator.accumulate_grads(-step_size, 0.0),
                Momentum::Update{mu} |
                Momentum::UpdateNesterov{mu}    => operator.accumulate_grads(-step_size, mu),
                // XXX(20160422): These are the Torch `optim.sgd`-style update;
                // see: <https://github.com/torch/optim/blob/master/sgd.lua>.
                Momentum::Gradient{mu}          => operator.accumulate_grads(1.0, mu),
                Momentum::GradientNesterov{mu}  => operator.accumulate_grads(1.0, mu),
              }

              // Apply the update, possibly with momentum.
              match self.config.momentum {
                Momentum::Zero                  => operator.update_params(1.0),
                Momentum::Update{mu} |
                Momentum::UpdateNesterov{mu}    => operator.update_params(1.0),
                // XXX(20160422): These are the Torch `optim.sgd`-style update;
                // see: <https://github.com/torch/optim/blob/master/sgd.lua>.
                Momentum::Gradient{mu}          => operator.update_params(-step_size),
                Momentum::GradientNesterov{mu}  => operator.update_params2(-step_size, -step_size * mu),
              }

              // Communicate the parameters.
              if iter_counter > 0 && iter_counter % self.config.comm_interval == 0 {
                operator.sync_params_v2();
              }
            }

            SyncOrder::SyncParamsThenStep => {
              // Compute the update, possibly with momentum.
              match self.config.momentum {
                Momentum::Zero                  => operator.accumulate_grads(-step_size, 0.0),
                Momentum::Update{mu} |
                Momentum::UpdateNesterov{mu}    => operator.accumulate_grads(-step_size, mu),
                // XXX(20160422): These are the Torch `optim.sgd`-style update;
                // see: <https://github.com/torch/optim/blob/master/sgd.lua>.
                Momentum::Gradient{mu}          => operator.accumulate_grads(1.0, mu),
                Momentum::GradientNesterov{mu}  => operator.accumulate_grads(1.0, mu),
              }

              // Communicate the parameters.
              if iter_counter > 0 && iter_counter % self.config.comm_interval == 0 {
                operator.sync_params_v2();
              }

              // Apply the update, possibly with momentum.
              match self.config.momentum {
                Momentum::Zero                  => operator.update_params(1.0),
                Momentum::Update{mu} |
                Momentum::UpdateNesterov{mu}    => operator.update_params(1.0),
                // XXX(20160422): These are the Torch `optim.sgd`-style update;
                // see: <https://github.com/torch/optim/blob/master/sgd.lua>.
                Momentum::Gradient{mu}          => operator.update_params(-step_size),
                Momentum::GradientNesterov{mu}  => operator.update_params2(-step_size, -step_size * mu),
              }
            }

            SyncOrder::SyncGradsThenStep => {
              // Communicate the gradients.
              if iter_counter > 0 && iter_counter % self.config.comm_interval == 0 {
                operator.sync_grads_v2(false);
              }

              // Compute the update, possibly with momentum.
              match self.config.momentum {
                Momentum::Zero                  => operator.accumulate_grads(-step_size, 0.0),
                Momentum::Update{mu} |
                Momentum::UpdateNesterov{mu}    => operator.accumulate_grads(-step_size, mu),
                // XXX(20160422): These are the Torch `optim.sgd`-style update;
                // see: <https://github.com/torch/optim/blob/master/sgd.lua>.
                Momentum::Gradient{mu}          => operator.accumulate_grads(1.0, mu),
                Momentum::GradientNesterov{mu}  => operator.accumulate_grads(1.0, mu),
              }

              // Apply the update, possibly with momentum.
              match self.config.momentum {
                Momentum::Zero                  => operator.update_params(1.0),
                Momentum::Update{mu} |
                Momentum::UpdateNesterov{mu}    => operator.update_params(1.0),
                // XXX(20160422): These are the Torch `optim.sgd`-style update;
                // see: <https://github.com/torch/optim/blob/master/sgd.lua>.
                Momentum::Gradient{mu}          => operator.update_params(-step_size),
                Momentum::GradientNesterov{mu}  => operator.update_params2(-step_size, -step_size * mu),
              }
            }

            SyncOrder::SyncParamsAndGradsThenStep => {
              // FIXME(20160513)
              unimplemented!();
            }
          }

          operator.reset();

          iter_counter += 1;
          //info!("DEBUG: rank: {} post iter: {}", rank, iter_counter);

          // FIXME(20160425): write iteration stats to log file.
          let minibatch_lap_time = get_time();
          let minibatch_elapsed_ms = (minibatch_lap_time - minibatch_start_time).num_milliseconds();
          let minibatch_accuracy = minibatch_acc_correct_count as f32 / minibatch_acc_total_count as f32;
          writeln!(&mut self.local_log_file, "{},step,{:.6},{:.4},{:.3}",
              iter_counter, minibatch_acc_loss, 1.0 - minibatch_accuracy, minibatch_elapsed_ms as f32 * 0.001).unwrap();
          minibatch_start_time = get_time();
          minibatch_acc_correct_count = 0;
          minibatch_acc_total_count = 0;
          minibatch_acc_loss = 0.0;

          if iter_counter % self.config.display_iters == 0 {
            self.shared.sync();
            let lap_time = minibatch_lap_time;
            if rank == 0 {
              let elapsed_ms = (lap_time - start_time).num_milliseconds();
              let acc_correct_count = self.shared.acc_correct_count.load(Ordering::Acquire);
              let acc_total_count = self.shared.acc_total_count.load(Ordering::Acquire);
              let accuracy = acc_correct_count as f32 / acc_total_count as f32;
              let avg_loss = self.local_accum_loss / self.config.display_iters as f32;
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
            self.local_log_file.flush().unwrap();
            self.shared.sync();
            start_time = get_time();
            minibatch_start_time = start_time;
          }

          //if iter_counter % self.config.save_iters == 0 {
          if iter_counter % self.config.checkpoint_iters == 0 {
            //println!("DEBUG: sgd: rank: {} signal checkpoint", rank);
            operator.signal_checkpoint();
          }

          if operator.wait_checkpoint() {
            //println!("DEBUG: sgd: rank: {} checkpoint...", rank);
            match self.local_log_file.flush() {
              Ok(_) => {}
              Err(e) => panic!("train: failed to flush local log file: {:?}", e),
            }
            operator.save_params();
            //println!("DEBUG: sgd: rank: {} exact sync...", rank);
            operator.exact_sync_params();
            //println!("DEBUG: sgd: rank: {} done exact sync", rank);
            if rank == 0 {
              operator.checkpoint_params(iter_counter, &self.config.checkpoint_dir);
            }
            self.validate(iter_counter, datum_cfg, label_cfg, valid_data, operator);
            operator.restore_params();
            self.elapsed_checkpoint_iters += self.config.checkpoint_iters;
            self.local_step_size = self.config.step_size.at_iter(self.elapsed_checkpoint_iters);
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
              operator.update_params(mu);
            }
            _ => {}
          }
        }
      });

      //epoch_counter += 1;
    }
  }

  pub fn validate(&mut self, iter_counter: usize, datum_cfg: SampleDatumConfig, label_cfg: SampleLabelConfig, valid_data: &mut DataIterator, operator: &mut OperatorWorker) {
    let batch_size = operator.batch_size();
    let num_workers = operator.num_workers();
    let rank = operator.worker_rank();

    let num_samples = valid_data.max_num_samples();

    //info!("DEBUG: validate: num samples: {}", num_samples);
    /*self.shared.acc_correct_count.store(0, Ordering::Release);
    self.shared.acc_total_count.store(0, Ordering::Release);
    self.local_accum_loss = 0.0;*/
    let mut local_acc_correct_count = 0;
    let mut local_acc_total_count = 0;
    let mut local_acc_loss = 0.0;

    let mut start_time = get_time();
    let mut local_counter = 0;
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
          operator.input_operator().expose_host_frame_buf(batch_counter)
            .copy_from_slice(frame_bytes.as_slice());
        }
        _ => unimplemented!(),
      }
      operator.loss_operator(0).stage_label(batch_counter, maybe_label.unwrap());
      operator.loss_operator(0).stage_weight(batch_counter, 1.0 / num_samples as f32);
      local_counter += 1;
      batch_counter += 1;

      if batch_counter == batch_size {
        operator.input_operator().load_frames(batch_size);
        operator.loss_operator(0).load_labels(batch_size);
        operator.loss_operator(0).load_weights(batch_size);
        operator.forward(batch_size, OpPhase::Inference);
        operator.loss_operator(0).store_output_categories(batch_size);
        let local_correct_count = operator.loss_operator(0).accuracy_count(batch_size);
        let local_loss = operator.loss_operator(0).store_loss(batch_size);
        /*self.shared.acc_correct_count.fetch_add(local_correct_count, Ordering::AcqRel);
        self.shared.acc_total_count.fetch_add(batch_size, Ordering::AcqRel);
        self.local_accum_loss += local_loss;*/
        local_acc_correct_count += local_correct_count;
        local_acc_total_count += batch_size;
        local_acc_loss += local_loss;
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
      /*self.shared.acc_correct_count.fetch_add(local_correct_count, Ordering::AcqRel);
      self.shared.acc_total_count.fetch_add(batch_size, Ordering::AcqRel);
      self.local_accum_loss += local_loss;*/
      local_acc_correct_count += local_correct_count;
      local_acc_total_count += batch_size;
      local_acc_loss += local_loss;
      batch_counter = 0;
    }

    assert_eq!(local_counter, num_samples);

    self.shared.sync();
    /*let acc_correct_count = self.shared.acc_correct_count.load(Ordering::Acquire);
    let acc_total_count = self.shared.acc_total_count.load(Ordering::Acquire);
    //let accuracy = acc_correct_count as f32 / acc_total_count as f32;
    let local_loss = self.local_accum_loss;*/
    let local_stats = vec![1.0, local_acc_correct_count as f32, local_acc_total_count as f32, local_acc_loss];
    let mut total_stats = vec![0.0, 0.0, 0.0, 0.0];
    operator.allreduce(&local_stats, &mut total_stats);

    let accuracy = total_stats[1] / total_stats[2];
    let loss = total_stats[3] / total_stats[0];

    let lap_time = get_time();
    let elapsed_ms = (lap_time - start_time).num_milliseconds();
    start_time = lap_time;

    writeln!(&mut self.local_log_file, "{},valid,{:.6},{:.4},{:.3}",
        iter_counter, loss, 1.0 - accuracy, elapsed_ms as f32 * 0.001).unwrap();

    if rank == 0 {
      info!("SgdOpt: valid: sample count: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
          local_acc_total_count,
          loss,
          accuracy,
          elapsed_ms as f32 * 0.001,
      );
    }
    self.local_log_file.flush().unwrap();
    /*self.shared.acc_correct_count.store(0, Ordering::Release);
    self.shared.acc_total_count.store(0, Ordering::Release);
    self.local_accum_loss = 0.0;*/
    self.shared.sync();
  }
}*/
