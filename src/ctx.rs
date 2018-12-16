use cudart::{CudaDevice};

use std::cell::{Cell, RefCell};
use std::marker::{Unpin};

thread_local! {
  static ROOT_DEVICE:   Cell<Option<i32>> = Cell::new(None);
  static DEVICE_STACK:  RefCell<Vec<i32>> = RefCell::new(Vec::new());
}

pub struct GpuCtxGuard {
  dev:  i32,
  pop:  i32,
}

impl !Send for GpuCtxGuard {}
impl !Sync for GpuCtxGuard {}
impl !Unpin for GpuCtxGuard {}

impl Drop for GpuCtxGuard {
  fn drop(&mut self) {
    DEVICE_STACK.with(|dev_stack| {
      let mut dev_stack = dev_stack.borrow_mut();
      match dev_stack.pop() {
        None => panic!("bug"),
        Some(d) => assert_eq!(d, self.dev),
      }
      match CudaDevice::set_current(self.pop) {
        Err(e) => {
          panic!("set current device failed: {:?} ({})", e, e.get_string());
        }
        Ok(_) => {}
      }
    });
  }
}

impl GpuCtxGuard {
  pub fn new(dev: i32) -> GpuCtxGuard {
    DEVICE_STACK.with(|dev_stack| {
      ROOT_DEVICE.with(|root_dev| {
        let mut dev_stack = dev_stack.borrow_mut();
        let depth = dev_stack.len();
        let pop = match depth {
          0 => {
            match root_dev.get() {
              None => {
                let curr_dev = match CudaDevice::get_current() {
                  Err(e) => {
                    panic!("get current device failed: {:?} ({})", e, e.get_string());
                  }
                  Ok(dev) => {
                    dev
                  }
                };
                root_dev.set(Some(curr_dev));
                curr_dev
              }
              Some(d) => d,
            }
          }
          _ => {
            dev_stack[depth - 1]
          }
        };
        match CudaDevice::set_current(dev) {
          Err(e) => {
            panic!("set current device failed: {:?} ({})", e, e.get_string());
          }
          Ok(_) => {}
        }
        dev_stack.push(dev);
        GpuCtxGuard{dev, pop}
      })
    })
  }
}
