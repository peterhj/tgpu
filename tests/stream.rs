extern crate gpurepr;
extern crate tgpu;

use gpurepr::*;
use tgpu::*;

#[test]
fn stream_basic() {
  let _stream = TGpuStream::default();
}

#[test]
fn stream_do_stuff() {
  let mut stream = TGpuStream::default();
  let x = stream.run_wait(|_stream| {
    42_u32
  });
  assert_eq!(42, x);
}

#[test]
fn stream_do_stuff_2() {
  let mut stream = TGpuStream::default();
  let x = stream.run_wait(|_| {
    1_i32
  });
  let y = {
    let x = x.clone();
    stream.run_wait(move |ref mut stream| {
      //let x = x.sync(stream);
      x
    })
  };
  let z = {
    let x = x.clone();
    let y = y.clone();
    stream.run_wait(|ref mut stream| {
      //let x = x.sync(stream);
      //let y = y.sync(stream);
      x + y
    })
  };
  let w = {
    let x1 = x.clone();
    let x2 = x.clone();
    let y = y.clone();
    stream.run_wait(|ref mut stream| {
      //let x1 = x1.sync(stream);
      //let x2 = x2.sync(stream);
      //let y = y.sync(stream);
      x1 + x2 + y
    })
  };
  assert_eq!(1, x);
  assert_eq!(1, y);
  assert_eq!(2, z);
  assert_eq!(3, w);
}

#[test]
fn stream_mem() {
  let mut stream = TGpuStream::default();
  fn do_something(x: impl GpuDelayed<GpuVMem<u32>>) {
    // TODO
  }
  let x = stream.run(|ref mut stream| {
    unsafe { GpuVMem::<u32>::alloc(1024, stream.device()) }
  });
  stream.run(|ref mut stream| {
    let x = x.sync(stream);
    do_something(x);
  });
}

#[test]
fn stream_mem_2() {
  let mut stream = TGpuStream::default();
  fn do_something_mut(x: impl GpuDelayedMut<GpuVMem<u32>>) {
    // TODO
  }
  let x = stream.run(|ref mut stream| {
    unsafe { GpuVMem::<u32>::alloc(1024, stream.device()) }
  });
  stream.run(|ref mut stream| {
    let x = x.sync_mut(stream);
    do_something_mut(x);
  });
}
