pub fn checkpoint_params(&mut self, t: usize, prefix: &Path) {
  let prefix = PathBuf::from(prefix);
  match create_dir_all(&prefix) {
    Ok(_) => {}
    Err(_) => {}
  }

  let mut blob = Vec::new();
  op.encode_params(&mut blob);

  let mut blob_path = prefix.clone();
  blob_path.push(&format!("params.t_{}.blob", t));
  let mut blob_file = match OpenOptions::new()
    .create(true).truncate(true).write(true)
    .open(&blob_path)
  {
    Ok(file) => file,
    Err(e) => panic!("checkpoint_params: failed to open blob file: {:?}", e),
  };
  match blob_file.write_all(&blob) {
    Ok(_) => {}
    Err(e) => panic!("checkpoint_params: failed to write to blob file: {:?}", e),
  }

  let mut latest_blob_path = prefix.clone();
  latest_blob_path.push("params.latest.blob");
  match remove_file(&latest_blob_path) {
    Ok(_) => {}
    Err(_) => {}
  }
  let blob_filename = PathBuf::from(&blob_path.file_name().unwrap());
  match symlink(&blob_filename, &latest_blob_path) {
    Ok(_) => {}
    Err(e) => panic!("checkpoint_params: failed to symlink latest blob: {:?}", e),
  }

  let mut checkpoint_path = prefix.clone();
  checkpoint_path.push("checkpoint");
  let mut bak_checkpoint_path = prefix.clone();
  bak_checkpoint_path.push("checkpoint.0");
  copy(&checkpoint_path, &bak_checkpoint_path).ok();
  let mut checkpoint_file = match OpenOptions::new()
    .create(true).truncate(true).write(true)
    .open(&checkpoint_path)
  {
    Ok(file) => file,
    Err(e) => panic!("checkpoint_params: failed to open checkpoint file for reading: {:?}", e),
  };
  writeln!(checkpoint_file, "{}", t);
}

pub fn checkpoint_state(&mut self, prefix: &Path) {
  let prefix = PathBuf::from(prefix);
  match create_dir_all(&prefix) {
    Ok(_) => {}
    Err(_) => {}
  }

  let mut blob = Vec::new();
  op.encode_params(&mut blob);
  op.encode_state(&mut blob);

  let mut blob_path = prefix.clone();
  blob_path.push(&format!("state.latest.blob.{}", self.worker_data.worker_rank()));
  let mut blob_file = match OpenOptions::new()
    .create(true).truncate(true).write(true)
    .open(&blob_path)
  {
    Ok(file) => file,
    Err(e) => panic!("checkpoint_state: failed to open blob file: {:?}", e),
  };
  match blob_file.write_all(&blob) {
    Ok(_) => {}
    Err(e) => panic!("checkpoint_state: failed to write to blob file: {:?}", e),
  }
}

pub fn can_rollback(&mut self, prefix: &Path) -> Option<usize> {
  let prefix = PathBuf::from(prefix);

  let mut checkpoint_path = prefix.clone();
  checkpoint_path.push("checkpoint");
  if !checkpoint_path.exists() {
    return None;
  }

  let checkpoint_file = match OpenOptions::new().read(true).open(&checkpoint_path) {
    Ok(file) => file,
    Err(e) => panic!("rollback_params: failed to open checkpoint file: {:?}", e),
  };

  let mut latest_t: Option<usize> = None;
  for line in BufReader::new(checkpoint_file).lines() {
    let line = line.unwrap();
    latest_t = line.parse().ok();
    break;
  }
  latest_t
}

pub fn rollback_params(&mut self, t: Option<usize>, prefix: &Path) {
  let prefix = PathBuf::from(prefix);

  let mut checkpoint_path = prefix.clone();
  checkpoint_path.push("checkpoint");
  let checkpoint_file = match OpenOptions::new().read(true).open(&checkpoint_path) {
    Ok(file) => file,
    Err(e) => panic!("rollback_params: failed to open checkpoint file: {:?}", e),
  };

  let blob_path = match t {
    Some(t) => {
      let mut latest_t: Option<usize> = None;
      for line in BufReader::new(checkpoint_file).lines() {
        let line = line.unwrap();
        latest_t = line.parse().ok();
        break;
      }
      match latest_t {
        Some(latest_t) => {
          assert!(t <= latest_t);
        }
        None => {
          panic!("rollback_params: checkpoint file is empty, but requested iter {}", t);
        }
      }
      let mut blob_path = prefix.clone();
      blob_path.push(&format!("params.t_{}.blob", t));
      blob_path
    }
    None => {
      let mut blob_path = prefix.clone();
      blob_path.push(&format!("params.latest.blob"));
      blob_path
    }
  };
  let mut blob_file = match OpenOptions::new()
    .read(true)
    .open(&blob_path) 
  {
      Ok(file) => file,
      Err(e) => panic!("rollback_params: failed to open blob file"),
  };
  let mut blob = Vec::new();
  match blob_file.read_to_end(&mut blob) {
    Ok(_) => {}
    Err(_) => panic!("rollback_params: failed to read blob file"),
  }

  let mut offset = 0;
  op.decode_params(&blob[offset .. ]);
}

pub fn rollback_state(op: &mut Operator, prefix: &Path) {
  let prefix = PathBuf::from(prefix);

  let mut blob_path = prefix.clone();
  blob_path.push(&format!("state.latest.blob.{}", self.worker_data.worker_rank()));

  let mut blob_file = match OpenOptions::new()
    .read(true)
    .open(&blob_path) 
  {
      Ok(file) => file,
      Err(e) => panic!("rollback_state: failed to open blob file"),
  };
  let mut blob = Vec::new();
  match blob_file.read_to_end(&mut blob) {
    Ok(_) => {}
    Err(_) => panic!("rollback_state: failed to read blob file"),
  }

  let mut offset = 0;
  op.decode_params(&blob[offset .. ]);
  op.decode_state(&blob[offset .. ]);
}
