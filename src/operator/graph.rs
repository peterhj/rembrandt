use operator::{
  Operator,
  OpCapability,
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
};
use operator::input::{
  VarData3dOperatorConfig,
};
use operator::loss::{
  CategoricalLossConfig,
};
use operator::pool::{
};

use std::collections::{BTreeMap, BTreeSet};
use std::collections::btree_map::{Entry};

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
  operators:    Vec<Box<Operator>>,
  input_ops:    Vec<usize>,
  loss_ops:     Vec<usize>,
  fwd_toporder: Vec<usize>,
  bwd_toporder: Vec<usize>,
}

impl GraphOperator {
  pub fn new(mut config: GraphOperatorConfig, capability: OpCapability) -> GraphOperator {
    config.input_keys.sort();

    let mut id_counter = 0;
    let mut key_ids = BTreeMap::new();
    let mut id_keys = Vec::with_capacity(config.nodes.len());
    let mut source_set = BTreeSet::new();
    for key in config.input_keys.iter() {
      key_ids.insert(key.clone(), id_counter);
      id_keys.push(key.clone());
      source_set.insert(id_counter);
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
    // FIXME(20160529): create fwd and bwd edges.

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

    let mut fwd_toporder = Vec::with_capacity(config.nodes.len());
    let mut fwd_unmarkset = BTreeSet::new();
    let mut fwd_tmpmarkset = BTreeSet::new();
    for id in 0 .. config.nodes.len() {
      fwd_unmarkset.insert(id);
    }
    while !fwd_unmarkset.is_empty() {
      //let id = fwd_unmarkset.pop().unwrap();
      let id = *fwd_unmarkset.iter().next().unwrap();
      topsort_visit(id, &fwd_edges, &mut fwd_unmarkset, &mut fwd_tmpmarkset, &mut fwd_toporder);
    }
    fwd_toporder.reverse();

    let mut bwd_toporder = Vec::with_capacity(config.nodes.len());
    let mut bwd_unmarkset = BTreeSet::new();
    let mut bwd_tmpmarkset = BTreeSet::new();
    for id in 0 .. config.nodes.len() {
      bwd_unmarkset.insert(id);
    }
    while !bwd_unmarkset.is_empty() {
      //let id = bwd_unmarkset.pop().unwrap();
      let id = *fwd_unmarkset.iter().next().unwrap();
      topsort_visit(id, &bwd_edges, &mut bwd_unmarkset, &mut bwd_tmpmarkset, &mut bwd_toporder);
    }
    bwd_toporder.reverse();

    unimplemented!();
  }
}
