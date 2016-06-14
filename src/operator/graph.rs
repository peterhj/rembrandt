use data_new::{SampleLabel};
use operator::{
  Operator, InputOperator, LossOperator, CompleteOperator,
  OpRead, OpWrite,
  OperatorVariant,
  OperatorConfig,
  OpCapability,
  OpPhase,
  ActivationFunction,
  Regularization,
  SplitOperatorConfig,
  JoinOperatorConfig,
  Pool2dOperatorConfig,
};
use operator::affine::{
  AffineOperatorConfig,
};
use operator::conv::{
  Conv2dOperatorConfig,
  BNormConv2dOperatorConfig,
  StackResConv2dOperatorConfig,
  ProjStackResConv2dOperatorConfig,
};
use operator::input::{
  VarData3dOperatorConfig,
};
use operator::loss::{
  CategoricalLossConfig,
};
use operator::pool::{
};

use array::{Array2d};
use array_cuda::device::context::{DeviceContext};
use rng::xorshift::{Xorshiftplus128Rng};

use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, BTreeSet};
use std::collections::btree_map::{Entry};
use std::rc::{Rc};

#[derive(Clone, Debug)]
pub struct GraphOperatorConfig {
  nodes:    BTreeMap<String, ConfigNode>,
  key_ids:  BTreeMap<String, usize>,
  id_keys:  Vec<String>,
  //input_keys:   Vec<String>,
  id_counter:   usize,
}

#[derive(Clone, Debug)]
struct ConfigNode {
  in_keys:  Vec<String>,
  config:   OperatorConfig,
}

impl GraphOperatorConfig {
  pub fn new() -> GraphOperatorConfig {
    GraphOperatorConfig{
      nodes:    BTreeMap::new(),
      key_ids:  BTreeMap::new(),
      id_keys:  vec![],
      //input_keys:   vec![],
      id_counter:   0,
    }
  }

  pub fn var_data3d(&mut self, key: &str, cfg: VarData3dOperatorConfig) -> &mut Self {
    match self.nodes.entry(key.to_owned()) {
      Entry::Occupied(_) => panic!(),
      Entry::Vacant(e) => {
        e.insert(ConfigNode{
          in_keys:  vec![],
          config:   OperatorConfig::VarData3d(cfg),
        });
      }
    }
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    //self.input_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn copy_split(&mut self, key: &str, in_key: &str, cfg: SplitOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::CopySplit(cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn add_join(&mut self, key: &str, in_keys: Vec<&str>, cfg: JoinOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  in_keys.iter().map(|&key| key.to_owned()).collect(),
      config:   OperatorConfig::AddJoin(cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn affine(&mut self, key: &str, in_key: &str, cfg: AffineOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::Affine(cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn conv2d(&mut self, key: &str, in_key: &str, cfg: Conv2dOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::Conv2d(cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn bnorm_conv2d(&mut self, key: &str, in_key: &str, cfg: BNormConv2dOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::BNormConv2d(cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn old_stack_res_conv2d(&mut self, key: &str, in_key: &str, cfg: StackResConv2dOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::StackResConv2d(cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn old_proj_stack_res_conv2d(&mut self, key: &str, in_key: &str, cfg: ProjStackResConv2dOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::ProjStackResConv2d(cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn stack_res_conv2d(&mut self, prefix: &str, in_key: &str, cfg: StackResConv2dOperatorConfig) -> &mut Self {
    let split_cfg = SplitOperatorConfig{
      in_dims:      cfg.in_dims,
      num_out_arms: 2,
    };
    let conv1_cfg = BNormConv2dOperatorConfig{
      in_dims:          cfg.in_dims,
      conv_size:        3,
      conv_stride:      1,
      conv_pad:         1,
      out_channels:     cfg.in_dims.2,
      bnorm_mov_avg:    cfg.bnorm_mov_avg,
      bnorm_epsilon:    cfg.bnorm_epsilon,
      pre_act_func:     ActivationFunction::Identity,
      act_func:         ActivationFunction::Rect,
      init_weights:     cfg.init_weights,
      fwd_backend:      cfg.fwd_backend,
      bwd_backend:      cfg.bwd_backend,
    };
    let conv2_cfg = BNormConv2dOperatorConfig{
      in_dims:          cfg.in_dims,
      conv_size:        3,
      conv_stride:      1,
      conv_pad:         1,
      out_channels:     cfg.in_dims.2,
      bnorm_mov_avg:    cfg.bnorm_mov_avg,
      bnorm_epsilon:    cfg.bnorm_epsilon,
      pre_act_func:     ActivationFunction::Identity,
      act_func:         ActivationFunction::Identity,
      init_weights:     cfg.init_weights,
      fwd_backend:      cfg.fwd_backend,
      bwd_backend:      cfg.bwd_backend,
    };
    let join_cfg = JoinOperatorConfig{
      num_in_arms:  2,
      out_dims:     cfg.in_dims,
      act_func:     ActivationFunction::Rect,
    };
    self.nodes.insert(format!("{}__split", prefix), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::CopySplit(split_cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(format!("{}__split", prefix), id);
    self.id_keys.push(format!("{}__split", prefix));
    self.id_counter += 1;
    self.bnorm_conv2d(&format!("{}__conv1", prefix), &format!("{}__split", prefix), conv1_cfg);
    self.bnorm_conv2d(&format!("{}__conv2", prefix), &format!("{}__conv1", prefix), conv2_cfg);
    self.nodes.insert(prefix.to_owned(), ConfigNode{
      in_keys:  vec![format!("{}__split", prefix), format!("{}__conv2", prefix)],
      config:   OperatorConfig::AddJoin(join_cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(prefix.to_owned(), id);
    self.id_keys.push(prefix.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn proj_stack_res_conv2d(&mut self, prefix: &str, in_key: &str, cfg: ProjStackResConv2dOperatorConfig) -> &mut Self {
    // FIXME(20160529): should constrain so that projection only changes the
    // number of output channels.
    assert_eq!(cfg.in_dims.0, cfg.out_dims.0);
    assert_eq!(cfg.in_dims.1, cfg.out_dims.1);
    let split_cfg = SplitOperatorConfig{
      in_dims:      cfg.in_dims,
      num_out_arms: 2,
    };
    let proj_conv_cfg = BNormConv2dOperatorConfig{
      in_dims:          cfg.in_dims,
      conv_size:        1,
      conv_stride:      1,
      conv_pad:         0,
      out_channels:     cfg.out_dims.2,
      bnorm_mov_avg:    cfg.bnorm_mov_avg,
      bnorm_epsilon:    cfg.bnorm_epsilon,
      pre_act_func:     ActivationFunction::Identity,
      act_func:         ActivationFunction::Identity,
      init_weights:     cfg.init_weights,
      fwd_backend:      cfg.fwd_backend,
      bwd_backend:      cfg.bwd_backend,
    };
    let conv1_cfg = BNormConv2dOperatorConfig{
      in_dims:          cfg.in_dims,
      conv_size:        3,
      conv_stride:      1,
      conv_pad:         1,
      out_channels:     cfg.out_dims.2,
      bnorm_mov_avg:    cfg.bnorm_mov_avg,
      bnorm_epsilon:    cfg.bnorm_epsilon,
      pre_act_func:     ActivationFunction::Identity,
      act_func:         ActivationFunction::Rect,
      init_weights:     cfg.init_weights,
      fwd_backend:      cfg.fwd_backend,
      bwd_backend:      cfg.bwd_backend,
    };
    let conv2_cfg = BNormConv2dOperatorConfig{
      in_dims:          cfg.out_dims,
      conv_size:        3,
      conv_stride:      1,
      conv_pad:         1,
      out_channels:     cfg.out_dims.2,
      bnorm_mov_avg:    cfg.bnorm_mov_avg,
      bnorm_epsilon:    cfg.bnorm_epsilon,
      pre_act_func:     ActivationFunction::Identity,
      act_func:         ActivationFunction::Identity,
      init_weights:     cfg.init_weights,
      fwd_backend:      cfg.fwd_backend,
      bwd_backend:      cfg.bwd_backend,
    };
    let join_cfg = JoinOperatorConfig{
      num_in_arms:  2,
      out_dims:     cfg.out_dims,
      act_func:     ActivationFunction::Rect,
    };
    self.nodes.insert(format!("{}__split", prefix), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::CopySplit(split_cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(format!("{}__split", prefix), id);
    self.id_keys.push(format!("{}__split", prefix));
    self.id_counter += 1;
    self.bnorm_conv2d(&format!("{}__conv0", prefix), &format!("{}__split", prefix), proj_conv_cfg);
    self.bnorm_conv2d(&format!("{}__conv1", prefix), &format!("{}__split", prefix), conv1_cfg);
    self.bnorm_conv2d(&format!("{}__conv2", prefix), &format!("{}__conv1", prefix), conv2_cfg);
    self.nodes.insert(prefix.to_owned(), ConfigNode{
      in_keys:  vec![format!("{}__conv0", prefix), format!("{}__conv2", prefix)],
      config:   OperatorConfig::AddJoin(join_cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(prefix.to_owned(), id);
    self.id_keys.push(prefix.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn pool2d(&mut self, key: &str, in_key: &str, cfg: Pool2dOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::Pool2d(cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }

  pub fn softmax_kl_loss(&mut self, key: &str, in_key: &str, cfg: CategoricalLossConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::SoftmaxKLLoss(cfg),
    });
    let id = self.id_counter;
    self.key_ids.insert(key.to_owned(), id);
    self.id_keys.push(key.to_owned());
    self.id_counter += 1;
    self
  }
}

pub struct GraphOperator {
  config:       GraphOperatorConfig,
  context:      Rc<DeviceContext>,
  operators:    Vec<Box<Operator>>,
  input_ops:    Vec<usize>,
  loss_ops:     Vec<usize>,
  fwd_toporder: Vec<usize>,
  bwd_toporder: Vec<usize>,
}

impl GraphOperator {
  pub fn new(mut config: GraphOperatorConfig, batch_size: usize, capability: OpCapability, context: Rc<DeviceContext>) -> GraphOperator {
    /*let mut id_counter = 0;
    let mut key_ids = BTreeMap::new();
    let mut id_keys = Vec::with_capacity(config.nodes.len());
    for key in config.input_keys.iter() {
      key_ids.insert(key.clone(), id_counter);
      id_keys.push(key.clone());
      id_counter += 1;
    }
    for key in config.nodes.keys() {
      key_ids.entry(key.clone()).or_insert_with(|| {
        id_keys.push(key.clone());
        let id = id_counter;
        id_counter += 1;
        id
      });
    }
    assert_eq!(id_counter, config.nodes.len());*/

    let mut fwd_in_edges = BTreeMap::new();
    let mut bwd_in_edges = BTreeMap::new();
    for id in 0 .. config.nodes.len() {
      fwd_in_edges.entry(id).or_insert(vec![]);
      bwd_in_edges.entry(id).or_insert(vec![]);
      let id_key = &config.id_keys[id];
      for left_key in config.nodes[id_key].in_keys.iter() {
        let left_id = config.key_ids[left_key];
        fwd_in_edges.entry(id).or_insert(vec![])
          .push(left_id);
        bwd_in_edges.entry(left_id).or_insert(vec![])
          .push(id);
      }
    }
    for id in 0 .. config.nodes.len() {
      fwd_in_edges.get_mut(&id).unwrap().sort();
      bwd_in_edges.get_mut(&id).unwrap().sort();
    }

    fn topsort(out_edges: &BTreeMap<usize, Vec<usize>>) -> Vec<usize> {
      let mut toporder = Vec::with_capacity(out_edges.len());
      let mut unmarkset = BTreeSet::new();
      let mut tmpmarkset = BTreeSet::new();
      for id in 0 .. out_edges.len() {
        unmarkset.insert(id);
      }
      while !unmarkset.is_empty() {
        let id = *unmarkset.iter().next().unwrap();
        topsort_visit(id, &out_edges, &mut unmarkset, &mut tmpmarkset, &mut toporder);
      }
      toporder.reverse();
      toporder
    }

    fn topsort_visit(id: usize, out_edges: &BTreeMap<usize, Vec<usize>>, unmarkset: &mut BTreeSet<usize>, tmpmarkset: &mut BTreeSet<usize>, toporder: &mut Vec<usize>) {
      if tmpmarkset.contains(&id) {
        panic!("GraphOperator: not a dag");
      }
      if unmarkset.contains(&id) {
        tmpmarkset.insert(id);
        for &next_id in out_edges[&id].iter() {
          topsort_visit(next_id, out_edges, unmarkset, tmpmarkset, toporder);
        }
        unmarkset.remove(&id);
        tmpmarkset.remove(&id);
        toporder.push(id);
      }
    }

    let fwd_toporder = topsort(&bwd_in_edges);
    let bwd_toporder = topsort(&fwd_in_edges);

    /*println!("DEBUG: key ids:   {:?}", config.key_ids);
    println!("DEBUG: fwd edges: {:?}", fwd_in_edges);
    println!("DEBUG: bwd edges: {:?}", bwd_in_edges);
    println!("DEBUG: fwd order: {:?}", fwd_toporder);
    println!("DEBUG: bwd order: {:?}", bwd_toporder);*/

    let mut operators_map: BTreeMap<usize, Box<Operator>> = BTreeMap::new();
    let mut input_ops = vec![];
    let mut loss_ops = vec![];
    for &id in fwd_toporder.iter() {
      let key = &config.id_keys[id];
      let op_var = {
        let mut prev_ops = vec![];
        for &prev_id in fwd_in_edges[&id].iter() {
          assert!(prev_id < id);
          let mut prev_arm: Option<usize> = None;
          for (arm, &prev_next_id) in bwd_in_edges[&prev_id].iter().enumerate() {
            if id == prev_next_id {
              prev_arm = Some(arm);
              break;
            }
          }
          assert!(prev_arm.is_some());
          let prev_arm = prev_arm.unwrap();
          prev_ops.push((prev_arm, &*operators_map[&prev_id]));
        }
        config.nodes[key].config.build_variant(batch_size, capability, prev_ops, context.clone())
      };
      match op_var {
        OperatorVariant::Hidden(op) => {
          operators_map.insert(id, op);
        }
        OperatorVariant::Input(op) => {
          operators_map.insert(id, op);
          input_ops.push(id);
        }
        OperatorVariant::Loss(op) => {
          operators_map.insert(id, op);
          loss_ops.push(id);
        }
      }
    }
    let mut operators = Vec::with_capacity(config.nodes.len());
    for id in 0 .. config.nodes.len() {
      operators.push(operators_map.remove(&id).unwrap());
    }
    assert!(operators_map.is_empty());

    GraphOperator{
      config:       config,
      context:      context,
      operators:    operators,
      input_ops:    input_ops,
      loss_ops:     loss_ops,
      fwd_toporder: fwd_toporder,
      bwd_toporder: bwd_toporder,
    }
  }
}

impl Operator for GraphOperator {
  fn batch_size(&self) -> usize {
    self.operators[0].batch_size()
  }

  fn params_len(&self) -> usize {
    let mut params_len = 0;
    for &id in self.fwd_toporder.iter() {
      params_len += self.operators[id].params_len();
    }
    params_len
  }

  fn init_param(&mut self, shared_seed: [u64; 2]) {
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    for &id in self.fwd_toporder.iter() {
      let op_seed = [rng.next_u64(), rng.next_u64()];
      self.operators[id].init_param(op_seed);
    }
  }

  fn forward(&mut self, batch_size: usize, phase: OpPhase) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].forward(batch_size, phase);
    }
  }

  fn backward(&mut self, batch_size: usize) {
    for &id in self.bwd_toporder.iter() {
      self.operators[id].backward(batch_size);
    }
  }

  fn reset(&mut self) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].reset();
    }
  }

  fn reset_grad(&mut self) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].reset_grad();
    }
  }

  fn read_grad(&mut self, init_offset: usize, reader: &mut OpRead) -> usize {
    let mut offset = init_offset;
    for &id in self.fwd_toporder.iter() {
      offset += self.operators[id].read_grad(offset, reader);
    }
    offset - init_offset
  }

  fn write_grad(&mut self, init_offset: usize, writer: &mut OpWrite) -> usize {
    let mut offset = init_offset;
    for &id in self.fwd_toporder.iter() {
      offset += self.operators[id].write_grad(offset, writer);
    }
    offset - init_offset
  }

  fn accumulate_grad_(&mut self, init_offset: usize, alpha: f32, mu: f32, writer: &mut OpWrite) -> usize {
    let mut offset = init_offset;
    for &id in self.fwd_toporder.iter() {
      offset += self.operators[id].accumulate_grad_(offset, alpha, mu, writer);
    }
    offset - init_offset
  }

  fn step(&mut self, init_offset: usize, step_size: f32, reader: &mut OpRead) -> usize {
    let mut offset = init_offset;
    for &id in self.fwd_toporder.iter() {
      offset += self.operators[id].step(offset, step_size, reader);
    }
    offset - init_offset
  }

  fn regularize(&mut self, reg: Regularization) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].regularize(reg);
    }
  }

  fn reset_stats(&mut self) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].reset_stats();
    }
  }

  fn estimate_stats(&mut self, acc_sample_size: usize, batch_size: usize) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].estimate_stats(acc_sample_size, batch_size);
    }
  }

  fn finalize_stats(&mut self, minibatch_size: usize) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].finalize_stats(minibatch_size);
    }
  }

  fn accumulate_grad(&mut self, scale: f32, momentum: f32) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].accumulate_grad(scale, momentum);
    }
  }

  fn update_param(&mut self, scale: f32) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].update_param(scale);
    }
  }

  fn update_param2(&mut self, grad_scale: f32, update_scale: f32) {
    for &id in self.fwd_toporder.iter() {
      self.operators[id].update_param2(grad_scale, update_scale);
    }
  }
}

impl InputOperator for GraphOperator {
  fn stage_shape(&mut self, batch_idx: usize, shape: (usize, usize, usize)) {
    for &id in self.input_ops.iter() {
      self.operators[id].upcast_input().stage_shape(batch_idx, shape);
    }
  }

  fn expose_host_frame_buf(&mut self, batch_idx: usize) -> &mut [u8] {
    // FIXME(20160601): potentially multiple inputs.
    let id0 = *self.input_ops.iter().next().unwrap();
    self.operators[id0].upcast_input().expose_host_frame_buf(batch_idx)
  }

  fn load_frames(&mut self, batch_size: usize) {
    for &id in self.input_ops.iter() {
      self.operators[id].upcast_input().load_frames(batch_size);
    }
  }

  fn preload_frame(&mut self, batch_idx: usize) {
    for &id in self.input_ops.iter() {
      self.operators[id].upcast_input().preload_frame(batch_idx);
    }
  }

  fn wait_preload_frames(&mut self, batch_size: usize) {
    for &id in self.input_ops.iter() {
      self.operators[id].upcast_input().wait_preload_frames(batch_size);
    }
  }
}

impl LossOperator for GraphOperator {
  fn stage_label(&mut self, batch_idx: usize, label: &SampleLabel) {
    // FIXME(20160601): potentially multiple inputs.
    let id0 = *self.loss_ops.iter().next().unwrap();
    self.operators[id0].upcast_loss().stage_label(batch_idx, label);
  }

  fn load_labels(&mut self, batch_size: usize) {
    // FIXME(20160601): potentially multiple inputs.
    let id0 = *self.loss_ops.iter().next().unwrap();
    self.operators[id0].upcast_loss().load_labels(batch_size);
  }

  fn stage_weight(&mut self, batch_idx: usize, weight: f32) {
    // FIXME(20160601): potentially multiple inputs.
    let id0 = *self.loss_ops.iter().next().unwrap();
    self.operators[id0].upcast_loss().stage_weight(batch_idx, weight);
  }

  fn load_weights(&mut self, batch_size: usize) {
    // FIXME(20160601): potentially multiple inputs.
    let id0 = *self.loss_ops.iter().next().unwrap();
    self.operators[id0].upcast_loss().load_weights(batch_size);
  }

  fn store_loss(&mut self, batch_size: usize) -> f32 {
    // FIXME(20160601): potentially multiple inputs.
    let id0 = *self.loss_ops.iter().next().unwrap();
    self.operators[id0].upcast_loss().store_loss(batch_size)
  }

  fn store_output_values(&mut self, batch_size: usize) { unimplemented!(); }
  fn get_output_values(&self, batch_size: usize) -> &Array2d<f32> { unimplemented!(); }

  fn store_output_categories(&mut self, batch_size: usize) {
    // FIXME(20160601): potentially multiple inputs.
    let id0 = *self.loss_ops.iter().next().unwrap();
    self.operators[id0].upcast_loss().store_output_categories(batch_size);
  }

  fn get_output_categories(&mut self, batch_size: usize) -> &[i32] {
    // FIXME(20160601): potentially multiple inputs.
    let id0 = *self.loss_ops.iter().next().unwrap();
    self.operators[id0].upcast_loss().get_output_categories(batch_size)
  }

  fn accuracy_count(&mut self, batch_size: usize) -> usize {
    // FIXME(20160601): potentially multiple inputs.
    let id0 = *self.loss_ops.iter().next().unwrap();
    self.operators[id0].upcast_loss().accuracy_count(batch_size)
  }
}

impl CompleteOperator for GraphOperator {
}
