fn main() {
  println!("cargo:rustc-flags=-L /lustre/atlas/proj-shared/csc103/phj/local/lib -l dmapp");
  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64");
  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l static=cudadevrt");
}
