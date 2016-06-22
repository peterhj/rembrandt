use data::{SampleDatum, SampleLabel, DataShard, DataIter};
use operator::{Operator, CompleteOperator, OpPhase, Regularization};
use opt::sgd::{
  InitBehavior,
  Momentum,
};
use opt::sgd::parallel::{ParallelSgdOptWorker};
use opt::second::parallel::solver::{ParallelSolver};

use array::{Shape};
use array_cuda::device::{DeviceBufferRef, DeviceBufferRefMut};

use std::ops::{Deref, DerefMut};
use std::path::{PathBuf};
use time::{get_time};

pub mod solver;

pub trait ParallelSecondOptWorker: ParallelSgdOptWorker {
  fn stage<'ctx>(&'ctx mut self, src: &DeviceBufferRef<'ctx, f32>);
  fn sync(&mut self);
  fn merge<'ctx>(&'ctx mut self, dst: &mut DeviceBufferRefMut<'ctx, f32>);
  fn read_step<'ctx>(&'ctx mut self, dst: &mut DeviceBufferRefMut<'ctx, f32>);
}

#[derive(Clone, Debug)]
pub struct ParallelSecondOptConfig {
  pub init:           InitBehavior,
  pub minibatch_size: usize,
  pub l2_reg_coef:    f32,

  pub display_iters:  usize,
  pub checkpoint_iters:   usize,
  pub checkpoint_dir:     PathBuf,
  pub save_iters:     usize,
  pub valid_iters:    usize,
}

pub struct ParallelSecondOpt<Solver> where Solver: ParallelSolver {
  config:   ParallelSecondOptConfig,
  solver:   Solver,
}

impl<Solver> ParallelSecondOpt<Solver> where Solver: ParallelSolver {
  pub fn new(config: ParallelSecondOptConfig, solver: Solver) -> ParallelSecondOpt<Solver> {
    ParallelSecondOpt{
      config:   config,
      solver:   solver,
    }
  }

  pub fn train(&mut self,
      mut train_data: &mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>,
      mut valid_data: Option<&mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>>,
      worker: &mut ParallelSecondOptWorker)
  {
    let batch_size = worker.operator().batch_size();
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
    let shared_seed = worker.shared_seed();
    worker.operator().init_param(shared_seed);
    worker.save_param();
    worker.operator().reset_grad();
    //worker.operator().reset_stats();

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
    let mut acc_batch_size = 0;

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
            worker.operator()
              .stage_shape(batch_counter, frame_bytes.bound());
            worker.operator()
              .expose_host_frame_buf(batch_counter)[ .. frame_len]
              .copy_from_slice(frame_bytes.as_slice());
            worker.operator()
              .preload_frame(batch_counter);
          }
          _ => unimplemented!(),
        }
        worker.operator().stage_label(batch_counter, &maybe_label.unwrap());
        worker.operator().stage_weight(batch_counter, minibatch_weight);
        batch_counter += 1;
        local_counter += 1;
        epoch_counter = local_counter / epoch_size;
        let epoch_offset = local_counter % epoch_size;

        // With a full batch of data, compute a full forward/backward pass.
        assert!(batch_counter <= batch_size);
        if batch_counter == batch_size {
          worker.operator().wait_preload_frames(batch_size);
          worker.operator().load_labels(batch_size);
          worker.operator().load_weights(batch_size);
          worker.operator().forward(batch_size, OpPhase::Training{t: iter_counter});
          worker.operator().backward(batch_size);
          worker.operator().store_output_categories(batch_size);
          let local_loss = worker.operator().store_loss(batch_size);
          let local_correct_count = worker.operator().accuracy_count(batch_size);
          display_acc_correct_count += local_correct_count;
          display_acc_total_count += batch_size;
          display_acc_loss += local_loss;
          minibatch_acc_correct_count += local_correct_count;
          minibatch_acc_total_count += batch_size;
          minibatch_acc_loss += local_loss;
          batch_counter = 0;
          acc_batch_size += batch_size;
        }

        if local_counter % minibatch_size == 0 {
          let l2_reg_coef = self.config.l2_reg_coef;

          assert_eq!(acc_batch_size, minibatch_size);
          acc_batch_size = 0;

          // Apply regularization to the current gradient.
          worker.operator().regularize(Regularization::L2{l2_reg_coef: l2_reg_coef});

          // Increase the iteration counter _before_ updates and communication.
          iter_counter += 1;

          // Communicate to synchronize the gradient.
          worker.sync_grad();

          self.solver.solve(batch_size, worker);
          worker.accumulate_grad(1.0, 0.0);
          worker.step(0.1);

          worker.save_param();

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
            if worker.worker_rank() == 0 {
              info!("SecondOpt: train: iter: {} epoch: {} sample: {}/{} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
                  iter_counter, epoch_counter,
                  epoch_offset, epoch_size,
                  avg_loss,
                  accuracy,
                  elapsed_ms as f32 * 0.001,
              );
            }
            display_acc_correct_count = 0;
            display_acc_total_count = 0;
            display_acc_loss = 0.0;
            //self.local_log_file.flush().unwrap();
            start_time = get_time();
            minibatch_start_time = start_time;
          }

          if iter_counter % self.config.valid_iters == 0 {
            if let Some(ref mut valid_data) = valid_data {
              self.validate(*valid_data, worker);
            }
            start_time = get_time();
            minibatch_start_time = start_time;
          }

          worker.operator().reset_grad();
        }
      }
    }
  }

  pub fn validate(&mut self,
      mut valid_data: &mut DataIter<Item=(SampleDatum, Option<SampleLabel>)>,
      worker: &mut ParallelSecondOptWorker)
  {
    let batch_size = worker.operator().batch_size();
    let epoch_size = valid_data.num_shard_samples();
    let total_size = valid_data.num_total_samples();
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
          worker.operator()
            .stage_shape(batch_counter, frame_bytes.bound());
          worker.operator()
            .expose_host_frame_buf(batch_counter)[ .. frame_len]
            .copy_from_slice(frame_bytes.as_slice());
          worker.operator()
            .preload_frame(batch_counter);
        }
        _ => unimplemented!(),
      }
      worker.operator().stage_label(batch_counter, &maybe_label.unwrap());
      worker.operator().stage_weight(batch_counter, weight);
      local_counter += 1;
      batch_counter += 1;

      // With a full batch of data, compute a full forward/backward pass.
      assert!(batch_counter <= batch_size);
      if batch_counter == batch_size {
        worker.operator().wait_preload_frames(batch_size);
        worker.operator().load_labels(batch_size);
        worker.operator().load_weights(batch_size);
        worker.operator().forward(batch_size, OpPhase::Inference);
        worker.operator().store_output_categories(batch_size);
        let local_loss = worker.operator().store_loss(batch_size);
        let local_correct_count = worker.operator().accuracy_count(batch_size);
        display_acc_correct_count += local_correct_count;
        display_acc_total_count += batch_size;
        display_acc_loss += local_loss;
        batch_counter = 0;
      }
    }
    if batch_counter > 0 {
      let batch_size = batch_counter;
      worker.operator().wait_preload_frames(batch_size);
      worker.operator().load_labels(batch_size);
      worker.operator().load_weights(batch_size);
      worker.operator().forward(batch_size, OpPhase::Inference);
      worker.operator().store_output_categories(batch_size);
      let local_loss = worker.operator().store_loss(batch_size);
      let local_correct_count = worker.operator().accuracy_count(batch_size);
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
    // FIXME(20160609): need to all-reduce the loss and accuracy.
    let avg_loss = display_acc_loss;
    //if worker.worker_rank() == 0 {
      info!("SecondOpt: valid: samples: {} loss: {:.06} accuracy: {:.03} elapsed: {:.03} s",
          total_size,
          avg_loss,
          accuracy,
          elapsed_ms as f32 * 0.001,
      );
    //}
  }
}
