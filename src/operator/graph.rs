use operator::{Operator};

pub struct GraphOperatorConfig {
}

struct Node {
  edges:    Vec<usize>,
}

pub struct GraphOperator {
  operators:    Vec<Box<Operator>>,
  input_ops:    Vec<usize>,
  loss_ops:     Vec<usize>,
  fwd_toporder: Vec<usize>,
  bwd_toporder: Vec<usize>,
}
