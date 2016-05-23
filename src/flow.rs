pub trait MonotoneTime: Ord {
  fn increment(&self) -> Self where Self: Sized;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MonotoneTime64 {
  time: u64,
}

impl MonotoneTime for MonotoneTime64 {
  fn increment(&self) -> MonotoneTime64 {
    MonotoneTime64{
      time: self.time + 1,
    }
  }
}
