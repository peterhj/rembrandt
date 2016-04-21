extern crate byteorder;

use byteorder::{WriteBytesExt, LittleEndian};
use std::io::{stdout};

fn main() {
  let mut writer = stdout();
  writer.write_u8(b'N').unwrap();
  writer.write_u8(b'D').unwrap();
  writer.write_u8(0).unwrap();
  writer.write_u8(1).unwrap();
  writer.write_u32::<LittleEndian>(3).unwrap();
  writer.write_u64::<LittleEndian>(256).unwrap();
  writer.write_u64::<LittleEndian>(256).unwrap();
  writer.write_u64::<LittleEndian>(3).unwrap();
}
