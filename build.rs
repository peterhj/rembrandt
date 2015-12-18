fn main() {
  println!("cargo:rustc-flags=-L /usr/local/cuda/lib64");
  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l static=cudadevrt");
}
