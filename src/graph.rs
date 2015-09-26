use std::collections::{BTreeMap, BTreeSet};

pub struct Graph<T> {
  pub vertexes:   BTreeMap<usize, T>,
  pub in_edges:   BTreeMap<usize, BTreeSet<usize>>,
  pub out_edges:  BTreeMap<usize, BTreeSet<usize>>,
  counter:        usize,
}

impl<T> Graph<T> {
  pub fn new() -> Graph<T> {
    Graph{
      vertexes:   BTreeMap::new(),
      in_edges:   BTreeMap::new(),
      out_edges:  BTreeMap::new(),
      counter:    0,
    }
  }

  fn alloc_id(&mut self) -> usize {
    let id = self.counter;
    self.counter += 1;
    id
  }

  pub fn insert_vertex(&mut self, vertex: T) -> usize {
    let id = self.alloc_id();
    self.vertexes.insert(id, vertex);
    id
  }

  pub fn insert_edge(&mut self, src: usize, dst: usize) {
    if !self.in_edges.contains_key(&dst) {
      self.in_edges.insert(dst, BTreeSet::new());
    }
    self.in_edges.get_mut(&dst).unwrap().insert(src);
    if !self.out_edges.contains_key(&src) {
      self.out_edges.insert(src, BTreeSet::new());
    }
    self.out_edges.get_mut(&src).unwrap().insert(dst);
  }

  pub fn topological_order(&self) -> Vec<usize> {
    let mut roots: Vec<usize> = self.vertexes.keys()
      .filter_map(|&key| if !self.in_edges.contains_key(&key) { Some(key) } else { None })
      .collect();
    let mut order = Vec::new();
    let mut marked_edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    let mut root_idx = 0;
    while root_idx < roots.len() {
      let root = roots[root_idx];
      order.push(root);
      if let Some(dsts) = self.out_edges.get(&root) {
        for &dst in dsts.iter() {
          if marked_edges.contains(&(root, dst)) {
            continue;
          }
          marked_edges.insert((root, dst));
          let mut num_unmarked = 0;
          for &src in self.in_edges.get(&dst).unwrap().iter() {
            if !marked_edges.contains(&(src, dst)) {
              num_unmarked += 1;
            }
          }
          if num_unmarked == 0 {
            roots.push(dst);
          }
        }
      }
      root_idx += 1;
    }
    // FIXME(20150925): only valid if DAG.
    order
  }
}
