pub enum InterpolateFilter {
  Nearest,
  Linear,
  BSpline,
  CatmullRom,
  MitchellNetravali,
  Lanczos2,
}

pub struct Interpolate2dOperatorConfig {
  pub in_dims:  (usize, usize, usize),
  pub out_dims: (usize, usize, usize),
  pub filter:   InterpolateFilter,
}

pub struct Interpolate2dOperator {
  batch_cap:    usize,
  config:       Interpolate2dOperatorConfig,
}
