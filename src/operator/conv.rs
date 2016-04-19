use operator::{
  ActivationFunction,
  ParamsInit,
  Conv2dFwdBackend,
  Conv2dBwdBackend,
};

#[derive(Clone, Copy)]
pub struct StackResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}

#[derive(Clone, Copy)]
pub struct ProjStackResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub out_dims:     (usize, usize, usize),
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}

#[derive(Clone, Copy)]
pub struct BotResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}

impl BotResConv2dOperatorConfig {
  fn params_len(&self) -> usize {
    let (_, _, in_channels) = self.in_dims;
    let weights1_len = 1 * 1 * in_channels * in_channels / 4;
    let bias1_len = in_channels / 4;
    let weights2_len = 3 * 3 * in_channels / 4 * in_channels / 4;
    let bias2_len = in_channels / 4;
    let weights3_len = 1 * 1 * in_channels / 4 * in_channels;
    let bias3_len = in_channels;
    weights1_len + bias1_len +
        weights2_len + bias2_len +
        weights3_len + bias3_len
  }
}

/*
pub struct BotResConv2dOperator<Comm> {
  batch_cap:    usize,
  _capability:  OpCapability,
  params_off:   usize,
  config:       BotResConv2dOperatorConfig,

  context:      Rc<DeviceContext>,

  in_act:       SharedDeviceBuf<f32>,
  in_delta:     Option<SharedDeviceBuf<f32>>,
  out_act:      SharedDeviceBuf<f32>,
  out_delta:    SharedDeviceBuf<f32>,

  weights1:     DeviceArray2d<f32>,
  bias1:        DeviceArray2d<f32>,
  weights2:     DeviceArray2d<f32>,
  bias2:        DeviceArray2d<f32>,
  weights3:     DeviceArray2d<f32>,
  bias3:        DeviceArray2d<f32>,

  workspace:    DeviceBuffer<u8>,
  conv1_fwd:    CudnnConvFwdOp,
  add_bias1:    CudnnAddOp,
  conv2_fwd:    CudnnConvFwdOp,
  add_bias2:    CudnnAddOp,
  conv3_fwd:    CudnnConvFwdOp,
  add_bias3:    CudnnAddOp,

  backward:     Option<BotResConv2dBwdOperator<Comm>>,
  //hv_backward:  Option<BotResConv2dHvBwdOperator>,
}

struct BotResConv2dBwdOperator<Comm> {
  grad_w1:      DeviceArray2d<f32>,
  grad_b1:      DeviceArray2d<f32>,
  acc_grad_w1:  DeviceArray2d<f32>,
  acc_grad_b1:  DeviceArray2d<f32>,
  conv1_bwd_w:  CudnnConvBwdFilterOp,
  conv1_bwd_d:  CudnnConvBwdDataOp,

  grad_w2:      DeviceArray2d<f32>,
  grad_b2:      DeviceArray2d<f32>,
  acc_grad_w2:  DeviceArray2d<f32>,
  acc_grad_b2:  DeviceArray2d<f32>,
  conv2_bwd_w:  CudnnConvBwdFilterOp,
  conv2_bwd_d:  CudnnConvBwdDataOp,

  grad_w3:      DeviceArray2d<f32>,
  grad_b3:      DeviceArray2d<f32>,
  acc_grad_w3:  DeviceArray2d<f32>,
  acc_grad_b3:  DeviceArray2d<f32>,
  conv3_bwd_w:  CudnnConvBwdFilterOp,
  conv3_bwd_d:  CudnnConvBwdDataOp,

  comm_worker:  Rc<RefCell<Comm>>,
}

/*struct BotResConv2dHvBwdOperator {
  dir_weights:  DeviceArray2d<f32>,
  dir_bias:     DeviceArray2d<f32>,
}*/

impl<Comm> BotResConv2dOperator<Comm> where Comm: CommWorker {
  pub fn new(batch_size: usize, capability: OpCapability, params_offset: usize, config: BotResConv2dOperatorConfig, prev_op: Option<&Operator>, comm_worker: Option<Rc<RefCell<Comm>>>, context: Rc<DeviceContext>) -> BotResConv2dOperator<Comm> {
    let BotResConv2dOperatorConfig{
      in_dims,
      .. } = config;
    let (in_width, in_height, in_channels) = in_dims;

    let ctx = &(*context).as_ref();

    let mut workspace_size = 0;

    /*let fwd1_algo = match config.fwd1_backend {
      BotResConv2dFwdBackend::CudnnImplicitPrecompGemm => cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
      BotResConv2dFwdBackend::CudnnFftTiling           => cudnnConvolutionFwdAlgo_t::FftTiling,
      _ => unimplemented!(),
    };*/
    let conv1_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(1, 1, in_channels, in_channels / 4).unwrap(),
        CudnnConvDesc::create_2d_symmetric(1, 0).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv1_fwd.work_size);

    let conv2_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(3, 3, in_channels / 4, in_channels / 4).unwrap(),
        CudnnConvDesc::create_2d_symmetric(1, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv2_fwd.work_size);

    let conv3_fwd = CudnnConvFwdOp::create_algo(
        cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
        CudnnFilterDesc::<f32>::create_4d(1, 1, in_channels / 4, in_channels).unwrap(),
        CudnnConvDesc::create_2d_symmetric(1, 0).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
        &*ctx.get_dnn(),
    ).unwrap();
    workspace_size = max(workspace_size, conv3_fwd.work_size);

    let backward = if capability.backward_enabled() {
      let conv_bwd_w = CudnnConvBwdFilterOp::create_fastest(
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
          CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(1, 1, out_channels, 1).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv_bwd_w.work_size);

      let conv_bwd_d = CudnnConvBwdDataOp::create_fastest(
          CudnnFilterDesc::<f32>::create_4d(conv_size, conv_size, in_channels, out_channels).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(out_width, out_height, out_channels, batch_size).unwrap(),
          CudnnConvDesc::create_2d_symmetric(conv_stride, conv_pad).unwrap(),
          CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
          &*ctx.get_dnn(),
      ).unwrap();
      workspace_size = max(workspace_size, conv_bwd_d.work_size);

      Some(BotResConv2dBwdOperator{
        grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        acc_grad_weights: DeviceArray2d::<f32>::zeros((conv_size * conv_size * in_channels, out_channels), ctx),
        acc_grad_bias:    DeviceArray2d::<f32>::zeros((1, out_channels), ctx),
        conv_bwd_w:   conv_bwd_w,
        conv_bwd_d:   conv_bwd_d,
        comm_worker:  comm_worker.unwrap(),
      })
    } else {
      None
    };

    let add_bias1 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels / 4, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
    );
    let add_bias2 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels / 4, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels / 4, batch_size).unwrap(),
    );
    let add_bias3 = CudnnAddOp::new(
        CudnnTensorDesc::<f32>::create_4d(1, 1, in_channels, 1).unwrap(),
        CudnnTensorDesc::<f32>::create_4d(in_width, in_height, in_channels, batch_size).unwrap(),
    );

    BotResConv2dOperator{
      batch_cap:    batch_size,
      _capability:  capability,
      params_off:   params_offset,
      config:       config,
      context:      context.clone(),
      in_act:       match prev_op.unwrap().get_output_vars() {
        Some(vars) => vars,
        None => panic!("BotResConv2dOperator missing required prev operator output vars"),
      },
      in_delta:     prev_op.unwrap().get_output_deltas(),
      out_act:      Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      out_delta:    Rc::new(RefCell::new(DeviceBuffer::<f32>::zeros(out_length * batch_size, ctx))),
      weights1:     DeviceArray2d::<f32>::zeros((1 * 1 * in_channels, in_channels / 4), ctx),
      bias1:        DeviceArray2d::<f32>::zeros((1, in_channels / 4), ctx),
      weights2:     DeviceArray2d::<f32>::zeros((3 * 3 * in_channels / 4, in_channels / 4), ctx),
      bias2:        DeviceArray2d::<f32>::zeros((1, in_channels / 4), ctx),
      weights3:     DeviceArray2d::<f32>::zeros((1 * 1 * in_channels / 4, in_channels), ctx),
      bias3:        DeviceArray2d::<f32>::zeros((1, in_channels), ctx),
      workspace:    DeviceBuffer::<u8>::zeros(workspace_size, ctx),
      conv1_fwd:    conv1_fwd,
      add_bias1:    add_bias1,
      conv2_fwd:    conv2_fwd,
      add_bias2:    add_bias2,
      conv3_fwd:    conv3_fwd,
      add_bias3:    add_bias3,
      backward:     backward,
      //hv_backward:  None,
    }
  }
}

impl<Comm> Operator for BotResConv2dOperator<Comm> where Comm: CommWorker {
  fn batch_size(&self) -> usize {
    self.batch_cap
  }

  fn get_output_vars(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_act.clone())
  }

  fn get_output_deltas(&self) -> Option<SharedDeviceBuf<f32>> {
    Some(self.out_delta.clone())
  }

  fn init_params(&mut self, shared_seed: [u64; 2]) {
    let BotResConv2dOperatorConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut rng = Xorshiftplus128Rng::from_seed(shared_seed);
    let mut init_weights = Array2d::zeros((conv_size * conv_size * in_channels, out_channels));
    match self.config.init_weights {
      ParamsInit::Disabled => {
        panic!("BotResConv2dOperator: params init explicitly disabled");
      }
      ParamsInit::Uniform{half_range} => {
        let dist = Range::new(-half_range as f64, half_range as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
      ParamsInit::Normal{std} => {
        let dist = Normal::new(0.0, std as f64);
        for w in init_weights.as_view_mut().as_mut_slice().iter_mut() {
          *w = dist.ind_sample(&mut rng) as f32;
        }
      }
    }
    let init_bias = Array2d::zeros((1, out_channels));
    self.weights.as_view_mut(ctx).sync_load(&init_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&init_bias.as_view());
  }

  fn read_params(&mut self, blob: &[u8]) -> usize {
    let BotResConv2dOperatorConfig{in_dims, conv_size, out_channels, ..} = self.config;
    let ctx = &(*self.context).as_ref();
    let (_, _, in_channels) = in_dims;
    let mut reader = Cursor::new(blob);
    let load_weights = Array2d::deserialize(&mut reader)
      .ok().expect("BotResConv2dOperator failed to deserialize weights!");
    let load_bias = Array2d::deserialize(&mut reader)
      .ok().expect("BotResConv2dOperator failed to deserialize bias!");
    assert_eq!((conv_size * conv_size * in_channels, out_channels), load_weights.as_view().bound());
    assert_eq!((1, out_channels), load_bias.as_view().bound());
    self.weights.as_view_mut(ctx).sync_load(&load_weights.as_view());
    self.bias.as_view_mut(ctx).sync_load(&load_bias.as_view());
    let progress = reader.position() as usize;
    progress
  }

  fn write_params(&mut self, blob: &mut Vec<u8>) {
    let ctx = &(*self.context).as_ref();
    let weights = self.weights.as_view(ctx);
    let bias = self.bias.as_view(ctx);
    let mut save_weights = Array2d::zeros(weights.bound());
    let mut save_bias = Array2d::zeros(bias.bound());
    weights.sync_store(&mut save_weights.as_view_mut());
    bias.sync_store(&mut save_bias.as_view_mut());
    save_weights.serialize(blob).unwrap();
    save_bias.serialize(blob).unwrap();
  }

  fn forward(&mut self, batch_size: usize, _phase: OpPhase) {
    assert!(batch_size <= self.batch_cap);
    let out_dims = self.config.get_out_dims();
    let out_length = out_dims.len();

    // FIXME(20160413): the bottleneck residual consists of 3 convolutions.

    let &mut BotResConv2dOperator{
      ref context,
      ref mut in_act, ref mut out_act,
      ref mut weights, ref mut bias,
      ref mut workspace,
      .. } = self;

    let ctx = &(**context).as_ref();
    let mut out_act = out_act.borrow_mut().as_ref_mut(ctx);

    self.conv_fwd.set_batch_size(batch_size).unwrap();
    match unsafe { self.conv_fwd.forward(
        in_act.borrow_mut().as_ref(ctx).as_ptr(),
        weights.as_view(ctx).as_ptr(),
        out_act.as_mut_ptr(),
        workspace.as_ref_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ) } {
      Ok(_) => {}
      Err(e) => { panic!("conv2d forward failed: {:?}", e); }
    }
    self.add_bias.set_batch_size(batch_size).unwrap();
    unsafe { self.add_bias.forward(
        bias.as_view(ctx).as_ptr(),
        out_act.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_inplace(
            out_act.as_mut_ptr(),
            out_length as i32,
            batch_size as i32,
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }
  }

  fn backward(&mut self, batch_size: usize) {
    assert!(self.backward.is_some());
    assert!(batch_size <= self.batch_cap);
    /*let BotResConv2dOperatorConfig{
      in_dims, conv_size, conv_stride, conv_pad,
      .. } = self.config;*/
    //let (in_width, in_height, in_channels) = in_dims;
    //let in_length = in_dims.len();
    let out_dims = self.config.get_out_dims();
    //let (out_width, out_height, out_channels) = out_dims;
    let out_length = out_dims.len();

    // FIXME(20160413): the bottleneck residual consists of 3 convolutions;
    // the output delta is copied to the input delta before applying the conv
    // backward pass.

    let &mut BotResConv2dOperator{
      ref context,
      ref mut in_act, ref mut in_delta,
      ref mut out_act, ref mut out_delta,
      ref mut weights, //ref mut bias,
      ref mut workspace,
      ref mut backward,
      .. } = self;
    let mut backward = backward.as_mut().unwrap();
    let &mut BotResConv2dBwdOperator{
      ref mut grad_weights, ref mut grad_bias,
      .. } = backward;

    let ctx = &(**context).as_ref();
    let in_act = in_act.borrow_mut().as_ref(ctx);
    let out_act = out_act.borrow_mut().as_ref(ctx);
    let mut out_delta = out_delta.borrow_mut().as_ref_mut(ctx);
    let mut workspace = workspace.as_ref_mut(ctx);

    match self.config.act_func {
      ActivationFunction::Identity => {}
      ActivationFunction::Rect => {
        unsafe { rembrandt_kernel_batch_map_rect_backprop_inplace(
            out_act.as_ptr(),
            out_length as i32,
            batch_size as i32,
            out_delta.as_mut_ptr(),
            ctx.stream.ptr,
        ) };
      }
      _ => unimplemented!(),
    }

    backward.conv_bwd_w.set_batch_size(batch_size).unwrap();
    unsafe { backward.conv_bwd_w.backward_filter(
        1.0,
        in_act.as_ptr(),
        out_delta.as_ptr(),
        1.0,
        grad_weights.as_view_mut(ctx).as_mut_ptr(),
        workspace.as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    unsafe { backward.conv_bwd_w.backward_bias(
        1.0,
        out_delta.as_ptr(),
        1.0,
        grad_bias.as_view_mut(ctx).as_mut_ptr(),
        &*ctx.get_dnn(),
    ).unwrap() };
    if let &mut Some(ref mut in_delta) = in_delta {
      backward.conv_bwd_d.set_batch_size(batch_size).unwrap();
      let mut in_delta = in_delta.borrow_mut().as_ref_mut(ctx);
      unsafe { backward.conv_bwd_d.backward_data(
          1.0,
          weights.as_view(ctx).as_ptr(),
          out_delta.as_ptr(),
          0.0,
          in_delta.as_mut_ptr(),
          workspace.as_mut_ptr(),
          &*ctx.get_dnn(),
      ).unwrap() };
    }
  }

  fn regularize(&mut self, reg: Regularization) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    match reg {
      Regularization::L2{l2_reg_coef} => {
        assert!(l2_reg_coef >= 0.0);
        if l2_reg_coef > 0.0 {
          backward.grad_weights.as_view_mut(ctx)
            .matrix_sum(l2_reg_coef, &self.weights.as_view(ctx));
          backward.grad_bias.as_view_mut(ctx)
            .row_vector_sum(l2_reg_coef, &self.bias.as_view(ctx));
        }
      }
    }
  }

  fn accumulate_grads(&mut self, scale: f32, momentum: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.acc_grad_weights.as_view_mut(ctx)
      .matrix_scale(momentum);
    backward.acc_grad_bias.as_view_mut(ctx)
      .row_vector_scale(momentum);
    backward.acc_grad_weights.as_view_mut(ctx)
      .matrix_sum(scale, &backward.grad_weights.as_view(ctx));
    backward.acc_grad_bias.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.grad_bias.as_view(ctx));
  }

  fn update_params(&mut self, scale: f32) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    self.weights.as_view_mut(ctx)
      .matrix_sum(scale, &backward.acc_grad_weights.as_view(ctx));
    self.bias.as_view_mut(ctx)
      .row_vector_sum(scale, &backward.acc_grad_bias.as_view(ctx));
  }

  fn save_params(&mut self) {
    unimplemented!();
  }

  fn restore_params(&mut self) {
    unimplemented!();
  }

  fn set_grads_with_params_diff(&mut self) {
    unimplemented!();
  }

  fn sync_grads(&mut self) {
    unimplemented!();
  }

  fn stage_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.load(self.params_off, &mut self.weights, ctx);
    comm_worker.load(self.params_off, &mut self.bias, ctx);
  }

  fn sync_params(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let backward = self.backward.as_ref().unwrap();
    let mut comm_worker = backward.comm_worker.borrow_mut();
    comm_worker.store(self.params_off, &mut self.weights, ctx);
    comm_worker.store(self.params_off, &mut self.bias, ctx);
  }

  fn reset_grads(&mut self, scale: f32) {
    unimplemented!();
  }

  fn reset(&mut self) {
    assert!(self.backward.is_some());
    let ctx = &(*self.context).as_ref();
    let mut backward = self.backward.as_mut().unwrap();
    backward.grad_weights.as_view_mut(ctx)
      .matrix_scale(0.0);
    backward.grad_bias.as_view_mut(ctx)
      .row_vector_scale(0.0);
  }
}
*/

#[derive(Clone, Copy)]
pub struct ProjBotResConv2dOperatorConfig {
  pub in_dims:      (usize, usize, usize),
  pub out_dims:     (usize, usize, usize),
  pub act_func:     ActivationFunction,
  pub init_weights: ParamsInit,
  pub fwd_backend:  Conv2dFwdBackend,
  pub bwd_backend:  Conv2dBwdBackend,
}
