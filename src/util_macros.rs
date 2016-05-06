// XXX: See <https://doc.rust-lang.org/std/macro.panic!.html> for an example of
// how to write a macro.
macro_rules! unwrap {
  ($expr:expr) => (match $expr {
    Option::Some(x) => x,
    Option::None => panic!("nothing to unwrap: {}", stringify!($expr)),
  });
  ($expr:expr, $msg:expr) => (match $expr {
    Option::Some(x) => x,
    Option::None => panic!($msg),
  });
  ($expr:expr, $fmt:expr, $($arg:tt)+) => (match $expr {
    Option::Some(x) => x,
    Option::None => panic!($fmt, $($arg)+),
  });
}
