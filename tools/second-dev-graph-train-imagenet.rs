extern crate array_cuda;
extern crate rembrandt;

#[macro_use]
extern crate log;
extern crate crossbeam;
extern crate env_logger;
extern crate rand;

use array_cuda::device::context::{DeviceContext};
use crossbeam::{scope};
use rembrandt::data::{
  AsyncQueueDataIter,
  CyclicSampleDataIter,
  RandomSampleDataIter,
  GenerateRandomArray3dDataIter,
};
use rembrandt::data::codec::{
  Array3dDataCodec,
  TurboJpegDataCodec,
};
use rembrandt::data::varraydb_data::{VarrayDbShard};
use rembrandt::operator::{
  Operator,
  OpCapability,
};
use rembrandt::operator::graph::{GraphOperator};
use rembrandt::opt::sgd::{
  InitBehavior,
};
use rembrandt::opt::second::parallel::{
  ParallelSecondOptConfig,
  ParallelSecondOpt,
};
use rembrandt::opt::second::parallel::solver::{
  //ParallelSolver,
  FisherIteration,
  CgDeviceParallelSolver,
};
use rembrandt::opt::parallel::dev_allreduce::{
  DeviceAllreduceOptWorkerBuilder,
};
use rembrandt::templates::resnet_graph::{
  build_resnet18pool_var224x224,
};

//use rand::{thread_rng};
use std::rc::{Rc};
use std::path::{PathBuf};

fn main() {
  env_logger::init().unwrap();

  let num_local_workers = 1;
  let batch_size = 32;
  let minibatch_size = 32;
  //info!("batch size: {}", batch_size);
  info!("num workers: {} batch size: {}", num_local_workers, batch_size);

  let opt_cfg = ParallelSecondOptConfig{
    init:           InitBehavior::InitOrResume,
    minibatch_size: minibatch_size,
    l2_reg_coef:    1.0e-4,
    display_iters:      1,
    checkpoint_iters:   625,
    save_iters:         625,
    valid_iters:        625,

    checkpoint_dir:     PathBuf::from("models/imagenet_maxscale480-resnet18pool_dev_x4_test"),
  };
  info!("opt: {:?}", opt_cfg);

  scope(|scope| {
    let builder = DeviceAllreduceOptWorkerBuilder::new(num_local_workers);
    let mut guards = vec![];
    for tid in 0 .. num_local_workers {
      let opt_cfg = opt_cfg.clone();
      let builder = builder.clone();
      let guard = scope.spawn(move || {
        let context = Rc::new(DeviceContext::new(tid));
        let operator_cfg = build_resnet18pool_var224x224();
        if tid == 0 {
          info!("operator: {:?}", operator_cfg);
        }
        let operator = Box::new(GraphOperator::new(operator_cfg, batch_size, OpCapability::RForward, context.clone()));
        let params_len = operator.params_len();
        let mut opt = ParallelSecondOpt::new(opt_cfg, CgDeviceParallelSolver::new(params_len, 20, FisherIteration::new(), context.clone()));
        let mut worker = builder.into_worker(tid, context, operator);

        let mut train_data =
            /*AsyncQueueDataIter::new(
            GenerateRandomArray3dDataIter::new(
                (480, 480, 3), 1000,
            ));*/
            AsyncQueueDataIter::new(
            //CyclicSampleDataIter::new(
            RandomSampleDataIter::new(
            VarrayDbShard::open_partition(
                &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_shuf/ilsvrc2012_maxscale480_shuf_train_data.varraydb"),
                &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_shuf/ilsvrc2012_maxscale480_shuf_train_labels.varraydb"),
                TurboJpegDataCodec::new(),
                //worker.worker_rank(), worker.num_workers(),
                //tid, num_local_workers,
                0, 1,
            )));

        let mut valid_data =
            /*GenerateRandomArray3dDataIter::new(
                (256, 256, 3), 1000,
            );*/
            //AsyncQueueDataIter::new(
            CyclicSampleDataIter::new(
            VarrayDbShard::open_partition(
                &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_orig/ilsvrc2012_scale256_orig_valid_data.varraydb"),
                &PathBuf::from("/rscratch/phj/data/ilsvrc2012_multiv2_orig/ilsvrc2012_scale256_orig_valid_labels.varraydb"),
                Array3dDataCodec::new(),
                //worker.worker_rank(), worker.num_workers(),
                tid, num_local_workers,
            ));

        opt.train(&mut train_data, Some(&mut valid_data), &mut worker);
      });
      guards.push(guard);
    }
    for guard in guards {
      guard.join();
    }
  });
}
