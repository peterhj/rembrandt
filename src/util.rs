use std::cmp::{min};

pub fn partition_range(upper_bound: usize, parts: usize) -> Vec<(usize, usize)> {
  let mut ranges = Vec::with_capacity(parts);
  let mut offset = 0;
  for p in 0 .. parts {
    let rem_parts = parts - p;
    let span = (upper_bound - offset + rem_parts - 1) / rem_parts;
    ranges.push((offset, offset + span));
    offset += span;
  }
  assert_eq!(offset, upper_bound);
  ranges
}

pub fn cyclic_partition_range(parts: usize, part_len: usize, upper_bound: usize) -> Vec<((usize, usize), Option<(usize, usize)>)> {
  assert!(part_len <= upper_bound);
  let mut ranges = Vec::with_capacity(parts);
  let mut offset = 0;
  for p in 0 .. parts {
    let front_len = min(part_len, upper_bound - offset);
    let back_len = if front_len < part_len {
      part_len - front_len
    } else {
      0
    };
    match back_len {
      0 => {
        ranges.push(((offset, offset + front_len), None));
      }
      back_len => {
        ranges.push(((offset, offset + front_len), Some((0, back_len))));
      }
    }
    let rem_parts = parts - p;
    let span = (upper_bound - offset + rem_parts - 1) / rem_parts;
    offset += span;
  }
  assert_eq!(offset, upper_bound);
  ranges
}
