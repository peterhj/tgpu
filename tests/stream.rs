extern crate tgpu;

use tgpu::*;

#[test]
fn stream_basic() {
  let _stream = TGpuStream::default();
}

#[test]
fn stream_do_stuff() {
  let mut stream = TGpuStream::default();
  let x = stream.run(|_stream| {
    42_u32
  });
  assert_eq!(42, x.wait());
}

#[test]
fn stream_do_stuff_2() {
  let mut stream = TGpuStream::default();
  let x = stream.run(|_| {
    1_i32
  });
  let y = {
    let x = x.clone();
    stream.run(move |ref mut stream| {
      let x = x.sync(stream);
      *x
    })
  };
  let z = {
    let x = x.clone();
    let y = y.clone();
    stream.run(|ref mut stream| {
      let x = x.sync(stream);
      let y = y.sync(stream);
      *x + *y
    })
  };
  let w = {
    let x1 = x.clone();
    let x2 = x.clone();
    let y = y.clone();
    stream.run(|ref mut stream| {
      let x1 = x1.sync(stream);
      let x2 = x2.sync(stream);
      let y = y.sync(stream);
      *x1 + *x2 + *y
    })
  };
  assert_eq!(1, x.wait());
  assert_eq!(1, y.wait());
  assert_eq!(2, z.wait());
  assert_eq!(3, w.wait());
}
