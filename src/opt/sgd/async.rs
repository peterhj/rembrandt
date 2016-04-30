use data::{DataShard, DataIter};
use data_new::{
  //DataIterator,
  SampleDatum, SampleDatumConfig, SampleLabel, SampleLabelConfig, 
};
use operator::{Operator, OpPhase, Regularization};
use operator::worker::{OperatorWorker};
use opt::sgd::{SgdOptConfig, StepSizeSchedule, Momentum, SyncOrder};

//use array_cuda::device::context::{DeviceCtxRef};

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{Write, BufWriter};
use std::path::{PathBuf};
use std::sync::{Arc, Barrier};
use std::sync::atomic::{AtomicUsize, Ordering, fence};
use time::{get_time};

pub struct AsyncSgdOpt {
  config:   SgdOptConfig,
  rank:     Option<usize>,
  //shared:   Arc<OptSharedData>,
  local_accum_loss: f32,
  local_log_file:   BufWriter<File>,
}

impl AsyncSgdOpt {
  pub fn new(config: SgdOptConfig, rank: Option<usize>/*, shared: Arc<OptSharedData>*/) -> AsyncSgdOpt {
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
      Err(e) => panic!("AsyncSgdOpt: failed to open log file: {:?}", e),
      Ok(file) => file,
    };
    let mut writer = BufWriter::new(local_log_file);
    writeln!(&mut writer, "t,event,loss,error,elapsed").unwrap();
    writer.flush().unwrap();
    AsyncSgdOpt{
      config:   config,
      rank:     rank,
      //shared:   shared,
      local_accum_loss: 0.0,
      local_log_file:   writer,
    }
  }

  pub fn train(&mut self,
      datum_cfg: SampleDatumConfig,
      label_cfg: SampleLabelConfig,
      mut train_data: &mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>,
      mut valid_data: Option<&mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>>,
      operator: &mut OperatorWorker)
  {
    assert_eq!(0, self.config.save_iters % self.config.display_iters);
    assert_eq!(0, self.config.valid_iters % self.config.save_iters);

    let batch_size = operator.batch_size();
    let num_workers = operator.num_workers();
    let rank = operator.worker_rank();
    let minibatch_size = self.config.minibatch_size;
    let minibatch_weight = 1.0 / minibatch_size as f32;
    //let epoch_size = (train_data.max_num_samples() / minibatch_size) * minibatch_size;
    let epoch_size = train_data.num_shard_samples();

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

    let mut start_time = get_time();
    let mut minibatch_start_time = start_time;

    let mut minibatch_acc_correct_count = 0;
    let mut minibatch_acc_total_count = 0;
    let mut minibatch_acc_loss = 0.0;

    let mut local_acc_correct_count = 0;
    let mut local_acc_total_count = 0;
    let mut local_acc_loss = 0.0;

    let mut epoch_counter = 0;
    let mut iter_counter = self.config.init_t.unwrap_or(0);
    let mut local_counter = 0;
    let mut batch_counter = 0;

    loop {
      //println!("DEBUG: rank: {} reset train data", rank);
      train_data.reset();
      //println!("DEBUG: rank: {} begin for loop", rank);
      for (datum, maybe_label) in &mut train_data {
        //println!("DEBUG: rank: {} get sample", rank);
        match (label_cfg, maybe_label.as_ref()) {
          (SampleLabelConfig::Category{num_categories}, Some(&SampleLabel::Category{category})) => {
            assert!(category >= 0);
            assert!(category < num_categories);
          }
          _ => panic!("AsyncSgdOpt: unsupported label"),
        }
        match datum {
          SampleDatum::WHCBytes(ref frame_bytes) => {
            //println!("DEBUG: frame: {:?}", frame_bytes.as_slice());
            operator.input_operator().expose_host_frame_buf(batch_counter)
              .copy_from_slice(frame_bytes.as_slice());
          }
          _ => unimplemented!(),
        }
        operator.loss_operator(0).stage_label(batch_counter, &maybe_label.unwrap());
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
          let batch_loss = operator.loss_operator(0).store_loss(batch_size);
          operator.backward(batch_size);
          let batch_correct_count = operator.loss_operator(0).accuracy_count(batch_size);
          minibatch_acc_correct_count += batch_correct_count;
          minibatch_acc_total_count += batch_size;
          minibatch_acc_loss += batch_loss;
          local_acc_correct_count += batch_correct_count;
          local_acc_total_count += batch_size;
          local_acc_loss += batch_loss;
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

            SyncOrder::SyncParamsThenStep => {
              // FIXME(20160428): necessary for elastic averagin and pull gossip.
              unimplemented!();
            }

            SyncOrder::SyncGradsThenStep => {
              // FIXME(20160424): necessary for sync all-reduce.
              unimplemented!();
            }
          }

          // If we are using the standard Nesterov update, apply some extra
          // momentum.
          // XXX(20160406): Interestingly, we should use the local update rather
          // than the communicated update with momentum.
          //operator.set_grads_with_params_diff();
          match self.config.momentum {
            Momentum::UpdateNesterov{mu} => {
              operator.update_params(mu);
            }
            _ => {}
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
            //self.shared.sync();
            let lap_time = minibatch_lap_time;
            if rank == 0 {
              let elapsed_ms = (lap_time - start_time).num_milliseconds();
              let acc_correct_count = local_acc_correct_count;
              let acc_total_count = local_acc_total_count;
              let accuracy = acc_correct_count as f32 / acc_total_count as f32;
              let avg_loss = local_acc_loss / self.config.display_iters as f32;
              info!("AsyncSgdOpt: train: iter: {} epoch: {} sample: {}/{} step: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
                  iter_counter, epoch_counter,
                  epoch_offset, epoch_size,
                  step_size,
                  avg_loss,
                  accuracy,
                  elapsed_ms as f32 * 0.001,
              );
              local_acc_correct_count = 0;
              local_acc_total_count = 0;
              local_acc_loss = 0.0;
            }
            self.local_log_file.flush().unwrap();
            //self.shared.sync();
            start_time = get_time();
            minibatch_start_time = start_time;
          }

          if iter_counter % self.config.save_iters == 0 {
            if rank == 0 {
              operator.signal_checkpoint();
            }
          }

          if operator.wait_checkpoint() {
            match self.local_log_file.flush() {
              Ok(_) => {}
              Err(e) => panic!("train: failed to flush local log file: {:?}", e),
            }
            operator.save_params();
            operator.exact_sync_params();
            if rank == 0 {
              operator.checkpoint_params(iter_counter, &self.config.checkpoint_dir);
            }
            if let Some(ref mut valid_data) = valid_data {
              self.validate(iter_counter, datum_cfg, label_cfg, *valid_data, operator);
            }
            operator.restore_params();
            start_time = get_time();
            minibatch_start_time = start_time;
          }
        }
      }
    }
  }

  pub fn validate(&mut self,
      iter_counter: usize,
      datum_cfg: SampleDatumConfig,
      label_cfg: SampleLabelConfig,
      mut valid_data: &mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>,
      operator: &mut OperatorWorker)
  {
    let batch_size = operator.batch_size();
    let num_workers = operator.num_workers();
    let rank = operator.worker_rank();

    //let num_samples = valid_data.max_num_samples();
    let num_shard_samples = valid_data.num_shard_samples();

    //info!("DEBUG: validate: num samples: {}", num_samples);
    let mut local_acc_correct_count = 0;
    let mut local_acc_total_count = 0;
    let mut local_acc_loss = 0.0;

    let mut start_time = get_time();
    let mut local_counter = 0;
    let mut batch_counter = 0;
    //valid_data.each_sample(datum_cfg, label_cfg, &mut |epoch_idx, datum, maybe_label| {
    for (datum, maybe_label) in (&mut valid_data).take(num_shard_samples) {
      match (label_cfg, maybe_label.as_ref()) {
        (SampleLabelConfig::Category{num_categories}, Some(&SampleLabel::Category{category})) => {
          assert!(category >= 0);
          assert!(category < num_categories);
        }
        _ => panic!("AsyncSgdOpt: unsupported label"),
      }
      match datum {
        SampleDatum::WHCBytes(ref frame_bytes) => {
          //println!("DEBUG: frame: {:?}", frame_bytes.as_slice());
          operator.input_operator().expose_host_frame_buf(batch_counter)
            .copy_from_slice(frame_bytes.as_slice());
        }
        _ => unimplemented!(),
      }
      operator.loss_operator(0).stage_label(batch_counter, &maybe_label.unwrap());
      operator.loss_operator(0).stage_weight(batch_counter, 1.0 / num_shard_samples as f32);
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
        local_acc_correct_count += local_correct_count;
        local_acc_total_count += batch_size;
        local_acc_loss += local_loss;
        batch_counter = 0;
      }
    }
    if batch_counter > 0 && batch_counter < batch_size {
      let batch_size = batch_counter;
      operator.input_operator().load_frames(batch_size);
      operator.loss_operator(0).load_labels(batch_size);
      operator.loss_operator(0).load_weights(batch_size);
      operator.forward(batch_size, OpPhase::Inference);
      operator.loss_operator(0).store_output_categories(batch_size);
      let local_correct_count = operator.loss_operator(0).accuracy_count(batch_size);
      let local_loss = operator.loss_operator(0).store_loss(batch_size);
      local_acc_correct_count += local_correct_count;
      local_acc_total_count += batch_size;
      local_acc_loss += local_loss;
      batch_counter = 0;
    }

    assert_eq!(local_counter, num_shard_samples);

    //self.shared.sync();
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
      info!("AsyncSgdOpt: valid: sample count: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
          local_acc_total_count,
          loss,
          accuracy,
          elapsed_ms as f32 * 0.001,
      );
    }
    self.local_log_file.flush().unwrap();
    //self.shared.sync();
  }
}
