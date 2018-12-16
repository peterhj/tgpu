#![feature(integer_atomics)]
#![feature(optin_builtin_traits)]
#![feature(pin)]
#![feature(specialization)]

#[cfg(feature = "gpu")]
extern crate cudart;
#[macro_use] extern crate lazy_static;
#[cfg(feature = "gpu")]
extern crate parking_lot;

#[cfg(feature = "gpu")]
use cudart::{CudaStream, CudaEvent};
#[cfg(feature = "gpu")]
use parking_lot::{Mutex};

use std::cmp::{Ordering, max};
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "gpu")]
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

pub mod ctx;

lazy_static! {
  static ref UID64_CTR: AtomicU64 = AtomicU64::new(0);
}

pub type DefaultEpoch = u64;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Uid(u64);

impl Uid {
  pub fn fresh() -> Uid {
    let old_uid = UID64_CTR.fetch_add(1, AtomicOrdering::Relaxed);
    let new_uid = old_uid + 1;
    assert!(new_uid != 0);
    Uid(new_uid)
  }
}

#[derive(Debug)]
pub enum TotalOrdering {
  Equal,
  Before,
  After,
}

pub trait TotalOrd {
  fn next(&self) -> Self where Self: Sized;
  fn total_cmp(&self, other: &Self) -> TotalOrdering;
}

#[derive(Clone, Debug)]
pub struct TotalTime<E=DefaultEpoch> {
  epoch:    E,
}

impl<E: Ord> TotalOrd for TotalTime<E> {
  default fn next(&self) -> TotalTime<E> {
    unimplemented!();
  }

  fn total_cmp(&self, other: &TotalTime<E>) -> TotalOrdering {
    match self.epoch.cmp(&other.epoch) {
      Ordering::Equal   => TotalOrdering::Equal,
      Ordering::Less    => TotalOrdering::Before,
      Ordering::Greater => TotalOrdering::After,
    }
  }
}

impl TotalOrd for TotalTime<DefaultEpoch> {
  fn next(&self) -> TotalTime<DefaultEpoch> {
    let new_e = self.epoch + 1;
    assert!(new_e != 0);
    TotalTime{epoch: new_e}
  }
}

#[derive(Debug)]
pub enum PCausalOrdering {
  Equal,
  Before,
  After,
  MaybeConcurrent,
}

pub trait PCausalOrd {
  fn maybe_advance(&mut self, other: &Self);
  fn causal_cmp(&self, other: &Self) -> PCausalOrdering;
}

#[derive(Clone, Debug)]
pub struct LamportTime<E=DefaultEpoch> {
  uid:  Uid,
  tt:   TotalTime<E>,
}

impl<E: Ord> PCausalOrd for LamportTime<E> {
  default fn maybe_advance(&mut self, _other: &LamportTime<E>) {
    unimplemented!();
  }

  fn causal_cmp(&self, other: &LamportTime<E>) -> PCausalOrdering {
    if self.uid == other.uid {
      match self.tt.total_cmp(&other.tt) {
        TotalOrdering::Equal    => PCausalOrdering::Equal,
        TotalOrdering::Before   => PCausalOrdering::Before,
        TotalOrdering::After    => PCausalOrdering::After,
      }
    } else {
      PCausalOrdering::MaybeConcurrent
    }
  }
}

impl PCausalOrd for LamportTime<DefaultEpoch> {
  fn maybe_advance(&mut self, other: &LamportTime<DefaultEpoch>) {
    self.tt.epoch = max(self.tt.epoch, other.tt.next().epoch);
  }
}

#[derive(Clone)]
pub struct TGpuEvent<E=DefaultEpoch> {
  dev:      i32,
  t:        LamportTime<E>,
  #[cfg(feature = "gpu")]
  revent:   Arc<Mutex<Option<CudaEvent>>>,
}

pub struct TGpuStreamRef<'a, E=DefaultEpoch> {
  this: &'a mut TGpuStream<E>,
}

impl<'a, E> TGpuStreamRef<'a, E> {
  #[cfg(feature = "gpu")]
  pub fn cuda_stream(&mut self) -> &mut CudaStream {
    &mut self.this.rstream
  }
}

pub struct TGpuStream<E=DefaultEpoch> {
  dev:      i32,
  t:        LamportTime<E>,
  horizons: HashMap<Uid, (LamportTime<E>, LamportTime<E>)>,
  qlatest:  Option<LamportTime<E>>,
  queue:    VecDeque<TGpuEvent<E>>,
  #[cfg(feature = "gpu")]
  rstream:  CudaStream,
}

impl<E: Ord> TGpuStream<E> {
  fn fresh_time(&self) -> LamportTime<E> {
    LamportTime{
      uid:  self.t.uid.clone(),
      tt:   self.t.tt.next(),
    }
  }
}

impl<E: Ord + Clone> TGpuStream<E> {
  pub fn run<F, V>(&mut self, fun: F) -> TGpuThunk<V, E>
  where F: FnOnce(&mut V, TGpuStreamRef<E>) + 'static, V: Default {
    self.wrap_run(V::default(), fun)
  }

  pub fn wrap_run<F, V>(&mut self, mut val: V, fun: F) -> TGpuThunk<V, E>
  where F: FnOnce(&mut V, TGpuStreamRef<E>) + 'static {
    self.t = self.fresh_time();
    (fun)(&mut val, TGpuStreamRef{this: self});
    let ev = self.post_event();
    TGpuThunk{
      dev:  self.dev,
      ev,
      val,
    }
  }

  fn post_event(&mut self) -> TGpuEvent<E> {
    // TODO
    match self.qlatest.take() {
      None => {}
      Some(qt) => {
        match qt.causal_cmp(&self.t) {
          PCausalOrdering::Before => {}
          PCausalOrdering::Equal |
          PCausalOrdering::After => {
            panic!("causal violation");
          }
          PCausalOrdering::MaybeConcurrent => {
            panic!("bug");
          }
        }
      }
    }
    self.qlatest = Some(self.t.clone());
    #[cfg(not(feature = "gpu"))]
    unimplemented!();
    #[cfg(feature = "gpu")]
    let recycle: Option<()> = match self.queue.front_mut() {
      None => {
        None
      }
      Some(ev) => {
        let mut revent = ev.revent.lock();
        assert!(revent.is_some());
        match revent.as_mut().unwrap().query() {
          Err(e) => {
            panic!("query failed: {:?} ({})", e, e.get_string());
          }
          Ok(CudaEventStatus::NotReady) => {
            None
          }
          Ok(CudaEventStatus::Complete) => {
            let mut revent = revent.take();
            match revent.record(&mut self.rstream) {
              Err(e) => {
                panic!("record failed: {:?} ({})", e, e.get_string());
              }
              Ok(_) => {}
            }
            Some(revent)
          }
        }
      }
    };
    unimplemented!();
    /*let new_ev = if let Some(revent) = recycle {
      // TODO
      match self.queue.pop_front() {
        None => panic!("bug"),
        Some(_) => {}
      }
      unimplemented!();
    } else {
      // TODO
      unimplemented!();
      /*#[cfg(feature = "gpu")]
      {
        let mut revent = new_ev.revent.lock();
        match revent.record(&mut stream.rstream) {
          Err(e) => {
            panic!("record failed: {:?} ({})", e, e.get_string());
          }
          Ok(_) => {}
        }
      }*/
    };*/
    //self.queue.push_back(new_ev.clone());
    //new_ev
  }

  fn maybe_sync(&mut self, ev: &mut TGpuEvent<E>) {
    let do_sync = match self.horizons.get(&ev.t.uid) {
      None => true,
      Some(&(ref ht, ref vt)) => {
        match vt.causal_cmp(&self.t) {
          PCausalOrdering::Before |
          PCausalOrdering::Equal => {}
          PCausalOrdering::After => {
            panic!("causal violation");
          }
          PCausalOrdering::MaybeConcurrent => {
            panic!("bug");
          }
        }
        match ht.causal_cmp(&ev.t) {
          PCausalOrdering::Before => {
            true
          }
          PCausalOrdering::Equal |
          PCausalOrdering::After => {
            false
          }
          PCausalOrdering::MaybeConcurrent => {
            panic!("bug");
          }
        }
      },
    };
    if do_sync {
      #[cfg(feature = "gpu")]
      {
        let mut revent = ev.revent.lock();
        match self.rstream.wait_event(&mut revent) {
          Err(e) => panic!("wait_event failed: {:?} ({})", e, e.get_string()),
          Ok(_) => {}
        }
      }
      self.t.maybe_advance(&ev.t);
      self.horizons.insert(ev.t.uid.clone(), (ev.t.clone(), self.t.clone()));
    }
  }
}

#[derive(Clone)]
pub struct TGpuThunk<V=(), E=DefaultEpoch> {
  dev:  i32,
  ev:   TGpuEvent<E>,
  val:  V,
}

impl<V, E: Ord + Clone> TGpuThunk<V, E> {
  pub fn wait(mut self, stream: TGpuStreamRef<E>) -> TGpuThunkRef<V, E> {
    match self.ev.t.causal_cmp(&stream.this.t) {
      PCausalOrdering::Before |
      PCausalOrdering::Equal => {}
      PCausalOrdering::After => {
        panic!("causal violation");
      }
      PCausalOrdering::MaybeConcurrent => {
        stream.this.maybe_sync(&mut self.ev);
      }
    }
    TGpuThunkRef{
      dev:    self.dev,
      ev:     self.ev,
      val:    self.val,
    }
  }
}

pub struct TGpuThunkRef<V=(), E=DefaultEpoch> {
  dev:  i32,
  ev:   TGpuEvent<E>,
  val:  V,
}
