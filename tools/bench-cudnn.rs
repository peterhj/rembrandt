extern crate cuda_dnn;

use cuda_dnn::v3::*;

fn main() {
  let cudnn = CudnnHandle::create()
    .ok().expect("failed to create cudnn handle!");

  let n = 1024;
  let h = 16;

  let src_desc = CudnnTensorDesc::<f32>::create_4d(8, 8, 12, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(5, 5, 12, h).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 2).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(8, 8, h, n).unwrap();
  let perf1 = CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(8, 8, h, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(3, 3, h, 6).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 1).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(8, 8, 6, n).unwrap();
  let perf2 = CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let time_ms = perf1.time_ms + perf2.time_ms;
  println!("DEBUG: 8x8 batch size:  {}", n);
  println!("DEBUG: approx nodes/s:  {}", (n as f32) / (time_ms / 1000.0));

  let n = 8;
  let input = 16;
  let hidden = 16;

  let src_desc = CudnnTensorDesc::<f32>::create_4d(19, 19, input, n).unwrap();
  //let filter_desc = CudnnFilterDesc::<f32>::create_4d(9, 9, input, hidden).unwrap();
  //let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 4).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(5, 5, input, hidden).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 2).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(19, 19, hidden, n).unwrap();
  let perf1 = CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(19, 19, hidden, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(3, 3, hidden, hidden).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 1).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(19, 19, hidden, n).unwrap();
  let perf2 = CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(19, 19, hidden, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(3, 3, hidden, 1).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 1).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(19, 19, 1, n).unwrap();
  let perf3 = CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let time_ms = perf1.time_ms + perf3.time_ms;
  //let time_ms = perf1.time_ms + perf2.time_ms + perf3.time_ms;

  /*let time_3l_ms = perf1.time_ms + (3.0 - 1.0) * perf2.time_ms;
  let time_6l_ms = perf1.time_ms + (6.0 - 1.0) * perf2.time_ms;
  let time_12l_ms = perf1.time_ms + (12.0 - 1.0) * perf2.time_ms;*/

  println!("DEBUG: 19x19 batch size: {}", n);
  println!("DEBUG: layer timings:    {:.3} {:.3} {:.3}",
      perf1.time_ms, perf2.time_ms, perf3.time_ms);
  println!("DEBUG: approx evals/s:   {}", (n as f32) / (time_ms / 1000.0));
  /*println!("DEBUG: approx evals/s (2L):  {}", (n as f32) / (time_ms / 1000.0));
  println!("DEBUG: approx evals/s (3L):  {}", (n as f32) / (time_3l_ms / 1000.0));
  println!("DEBUG: approx evals/s (6L):  {}", (n as f32) / (time_6l_ms / 1000.0));
  println!("DEBUG: approx evals/s (12L): {}", (n as f32) / (time_12l_ms / 1000.0));*/
  println!("DEBUG: approx games/s:   {}", (n as f32 / 360.0) / (time_ms / 1000.0));

  let n = 1;

  let src_desc = CudnnTensorDesc::<f32>::create_4d(224, 224, 3, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(11, 11, 3, 96).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(4, 0).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(54, 54, 96, n).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(54, 54, 96, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(1, 1, 96, 96).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 0).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(54, 54, 96, n).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(27, 27, 96, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(5, 5, 96, 256).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 2).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(27, 27, 256, n).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(27, 27, 256, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(1, 1, 256, 256).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 0).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(27, 27, 256, n).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(13, 13, 256, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(3, 3, 256, 384).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 1).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(13, 13, 384, n).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(13, 13, 384, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(1, 1, 384, 384).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 0).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(13, 13, 384, n).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(6, 6, 384, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(3, 3, 384, 1024).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 1).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(6, 6, 1024, n).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(6, 6, 1024, n).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(1, 1, 1024, 1024).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 0).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(6, 6, 1024, n).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  /*let src_desc = CudnnTensorDesc::<f32>::create_4d(224, 224, 3).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(3, 3, 3, 64).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 1).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(224, 224, 64).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();

  let src_desc = CudnnTensorDesc::<f32>::create_4d(224, 224, 64).unwrap();
  let filter_desc = CudnnFilterDesc::<f32>::create_4d(3, 3, 64, 128).unwrap();
  let conv_desc = CudnnConvDesc::create_2d_symmetric(1, 1).unwrap();
  let dst_desc = CudnnTensorDesc::<f32>::create_4d(224, 224, 128).unwrap();
  CudnnConvFwdOp::create_fastest(src_desc, filter_desc, conv_desc, dst_desc, &cudnn).unwrap();*/
}
