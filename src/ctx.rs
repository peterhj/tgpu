//use cudart::{CudaDevice};

use std::cell::{Cell, RefCell};
use std::marker::{Unpin};

thread_local! {
  static ROOT_DEVICE:   Cell<Option<i32>> = Cell::new(None);
  static DEVICE_STACK:  RefCell<Vec<i32>> = RefCell::new(Vec::new());
}

pub struct GpuDevGuard {
  dev:  i32,
  pop:  i32,
}

impl !Send for GpuDevGuard {}
impl !Sync for GpuDevGuard {}
impl !Unpin for GpuDevGuard {}

impl Drop for GpuDevGuard {
  fn drop(&mut self) {
    DEVICE_STACK.with(|dev_stack| {
      let mut dev_stack = dev_stack.borrow_mut();
      match dev_stack.pop() {
        None => panic!("bug"),
        Some(d) => assert_eq!(d, self.dev),
      }
      /*match CudaDevice::set_current(self.pop) {
        // TODO
      }*/
    });
  }
}

impl GpuDevGuard {
  pub fn new(dev: i32) -> GpuDevGuard {
    DEVICE_STACK.with(|dev_stack| {
      ROOT_DEVICE.with(|root_dev| {
        let mut dev_stack = dev_stack.borrow_mut();
        let depth = dev_stack.len();
        let pop = match depth {
          0 => {
            match root_dev.get() {
              None => {
                let curr_dev = unimplemented!()/*match CudaDevice::get_current() {
                  // TODO
                }*/;
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
        /*match CudaDevice::set_current(dev) {
          // TODO
        }*/
        dev_stack.push(dev);
        GpuDevGuard{dev, pop}
      })
    })
  }
}
