use operator::{
  Operator,
  OpCapability,
  OperatorVariant,
  OperatorConfig,
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

use array_cuda::device::context::{DeviceContext};

use std::collections::{BTreeMap, BTreeSet};
use std::collections::btree_map::{Entry};
use std::rc::{Rc};

pub struct GraphOperatorConfig {
  nodes:    BTreeMap<String, ConfigNode>,
  input_keys:   Vec<String>,
}

struct ConfigNode {
  in_keys:  Vec<String>,
  config:   OperatorConfig,
}

impl GraphOperatorConfig {
  pub fn new() -> GraphOperatorConfig {
    GraphOperatorConfig{
      nodes:    BTreeMap::new(),
      input_keys:   vec![],
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
    self.input_keys.push(key.to_owned());
    self
  }

  pub fn copy_split(&mut self, key: &str, in_key: &str, cfg: SplitOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::CopySplit(cfg),
    });
    self
  }

  pub fn add_join(&mut self, key: &str, in_keys: Vec<&str>, cfg: JoinOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  in_keys.iter().map(|&key| key.to_owned()).collect(),
      config:   OperatorConfig::AddJoin(cfg),
    });
    self
  }

  pub fn affine(&mut self, key: &str, in_key: &str, cfg: AffineOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::Affine(cfg),
    });
    self
  }

  pub fn conv2d(&mut self, key: &str, in_key: &str, cfg: Conv2dOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::Conv2d(cfg),
    });
    self
  }

  pub fn bnorm_conv2d(&mut self, key: &str, in_key: &str, cfg: BNormConv2dOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::BNormConv2d(cfg),
    });
    self
  }

  pub fn stack_res_conv2d(&mut self, prefix: &str, in_key: &str, cfg: StackResConv2dOperatorConfig) -> &mut Self {
    let split_cfg = SplitOperatorConfig{
      in_dims:      cfg.in_dims,
      num_out_arms: 2,
    };
    let conv_cfg = BNormConv2dOperatorConfig{
      in_dims:          cfg.in_dims,
      conv_size:        3,
      conv_stride:      1,
      conv_pad:         1,
      out_channels:     cfg.in_dims.2,
      bnorm_mov_avg:    cfg.bnorm_mov_avg,
      bnorm_epsilon:    cfg.bnorm_epsilon,
      act_func:         cfg.act_func,
      init_weights:     cfg.init_weights,
      fwd_backend:      cfg.fwd_backend,
      bwd_backend:      cfg.bwd_backend,
    };
    let join_cfg = JoinOperatorConfig{
      num_in_arms:  2,
      out_dims:     cfg.in_dims,
    };
    self.nodes.insert(format!("{}__split", prefix), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::CopySplit(split_cfg),
    });
    self.nodes.insert(format!("{}__conv1", prefix), ConfigNode{
      in_keys:  vec![format!("{}__split", prefix)],
      config:   OperatorConfig::BNormConv2d(conv_cfg),
    });
    self.nodes.insert(format!("{}__conv2", prefix), ConfigNode{
      in_keys:  vec![format!("{}__conv1", prefix)],
      config:   OperatorConfig::BNormConv2d(conv_cfg),
    });
    self.nodes.insert(prefix.to_owned(), ConfigNode{
      in_keys:  vec![format!("{}__split", prefix), format!("{}__conv2", prefix)],
      config:   OperatorConfig::AddJoin(join_cfg),
    });
    self
  }

  pub fn proj_stack_res_conv2d(&mut self, prefix: &str, in_key: &str, cfg: ProjStackResConv2dOperatorConfig) -> &mut Self {
    // FIXME(20160529): should constrain so that projection only changes the
    // number of output channels.
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
      act_func:         cfg.act_func,
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
      act_func:         cfg.act_func,
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
      act_func:         cfg.act_func,
      init_weights:     cfg.init_weights,
      fwd_backend:      cfg.fwd_backend,
      bwd_backend:      cfg.bwd_backend,
    };
    let join_cfg = JoinOperatorConfig{
      num_in_arms:  2,
      out_dims:     cfg.in_dims,
    };
    self.nodes.insert(format!("{}__split", prefix), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::CopySplit(split_cfg),
    });
    self.nodes.insert(format!("{}__conv0", prefix), ConfigNode{
      in_keys:  vec![format!("{}__split", prefix)],
      config:   OperatorConfig::BNormConv2d(proj_conv_cfg),
    });
    self.nodes.insert(format!("{}__conv1", prefix), ConfigNode{
      in_keys:  vec![format!("{}__split", prefix)],
      config:   OperatorConfig::BNormConv2d(conv1_cfg),
    });
    self.nodes.insert(format!("{}__conv2", prefix), ConfigNode{
      in_keys:  vec![format!("{}__conv1", prefix)],
      config:   OperatorConfig::BNormConv2d(conv2_cfg),
    });
    self.nodes.insert(prefix.to_owned(), ConfigNode{
      in_keys:  vec![format!("{}__conv0", prefix), format!("{}__conv2", prefix)],
      config:   OperatorConfig::AddJoin(join_cfg),
    });
    self
  }

  pub fn pool2d(&mut self, key: &str, in_key: &str, cfg: Pool2dOperatorConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::Pool2d(cfg),
    });
    self
  }

  pub fn softmax_kl_loss(&mut self, key: &str, in_key: &str, cfg: CategoricalLossConfig) -> &mut Self {
    self.nodes.insert(key.to_owned(), ConfigNode{
      in_keys:  vec![in_key.to_owned()],
      config:   OperatorConfig::SoftmaxKLLoss(cfg),
    });
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
    config.input_keys.sort();

    let mut id_counter = 0;
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
    assert_eq!(id_counter, config.nodes.len());

    let mut fwd_edges = BTreeMap::new();
    let mut bwd_edges = BTreeMap::new();
    for id in 0 .. config.nodes.len() {
      let id_key = &id_keys[id];
      for r_key in config.nodes[id_key].in_keys.iter() {
        let r_id = key_ids[r_key];
        fwd_edges.entry(id).or_insert(vec![])
          .push(r_id);
        bwd_edges.entry(r_id).or_insert(vec![])
          .push(id);
      }
    }
    for id in 0 .. config.nodes.len() {
      fwd_edges.get_mut(&id).unwrap().sort();
      bwd_edges.get_mut(&id).unwrap().sort();
    }

    fn topsort(edges: &BTreeMap<usize, Vec<usize>>) -> Vec<usize> {
      let mut toporder = Vec::with_capacity(edges.len());
      let mut unmarkset = BTreeSet::new();
      let mut tmpmarkset = BTreeSet::new();
      for id in 0 .. edges.len() {
        unmarkset.insert(id);
      }
      while !unmarkset.is_empty() {
        let id = *unmarkset.iter().next().unwrap();
        topsort_visit(id, &edges, &mut unmarkset, &mut tmpmarkset, &mut toporder);
      }
      toporder.reverse();
      toporder
    }

    fn topsort_visit(id: usize, edges: &BTreeMap<usize, Vec<usize>>, unmarkset: &mut BTreeSet<usize>, tmpmarkset: &mut BTreeSet<usize>, toporder: &mut Vec<usize>) {
      if tmpmarkset.contains(&id) {
        panic!("GraphOperator: not a dag");
      }
      if unmarkset.contains(&id) {
        tmpmarkset.insert(id);
        for &next_id in edges[&id].iter() {
          topsort_visit(next_id, edges, unmarkset, tmpmarkset, toporder);
        }
        unmarkset.remove(&id);
        tmpmarkset.remove(&id);
        toporder.push(id);
      }
    }

    let fwd_toporder = topsort(&fwd_edges);
    let bwd_toporder = topsort(&bwd_edges);

    let mut operators_map: BTreeMap<usize, Box<Operator>> = BTreeMap::new();
    let mut input_ops = vec![];
    let mut loss_ops = vec![];
    for &id in fwd_toporder.iter() {
      let key = &id_keys[id];
      let op_var = {
        let mut prev_ops = vec![];
        for &prev_id in bwd_edges[&id].iter() {
          prev_ops.push(&*operators_map[&prev_id]);
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
