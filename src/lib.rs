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
use std::collections::{HashMap};
#[cfg(feature = "gpu")]
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

pub mod ctx;

lazy_static! {
  static ref UID_64_CTR:    AtomicU64 = AtomicU64::new(0);
}

pub type DefaultEpoch = u64;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Uid(u64);

impl Uid {
  pub fn fresh() -> Uid {
    let old_uid = UID_64_CTR.fetch_add(1, AtomicOrdering::Relaxed);
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
  Concurrent,
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
      PCausalOrdering::Concurrent
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
  t:        Option<LamportTime<E>>,
  #[cfg(feature = "gpu")]
  revent:   Arc<Mutex<CudaEvent>>,
}

impl<E: Ord + Clone> TGpuEvent<E> {
  pub fn time(&self) -> LamportTime<E> {
    match &self.t {
      None => panic!(),
      Some(ref t) => t.clone(),
    }
  }

  pub fn post(&mut self, new_t: LamportTime<E>, stream: &mut TGpuStream<E>) {
    assert_eq!(self.dev, stream.dev, "bug");
    assert_eq!(new_t.uid, stream.t.uid, "bug");
    match self.t.take() {
      None => {}
      Some(_) => {
        panic!("bug: double post");
      }
    }
    #[cfg(feature = "gpu")]
    {
      let mut revent = self.revent.lock();
      match revent.record(&mut stream.rstream) {
        Err(e) => panic!("record failed: {:?} ({})", e, e.get_string()),
        Ok(_) => {}
      }
    }
    self.t = Some(new_t.clone());
  }
}

pub struct TGpuEventPool<E=DefaultEpoch> {
  _dummy:   E,
  //rfree:    VecDeque<CudaEvent>,
}

impl<E> TGpuEventPool<E> {
  pub fn make(&mut self) -> TGpuEvent<E> {
    // TODO
    unimplemented!();
  }
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
  horizons: HashMap<Uid, LamportTime<E>>,
  events:   TGpuEventPool<E>,
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
    let mut ev = self.events.make();
    self.t = self.fresh_time();
    (fun)(&mut val, TGpuStreamRef{this: self});
    ev.post(self.t.clone(), self);
    TGpuThunk{
      dev:  self.dev,
      t:    self.t.clone(),
      ev,
      val,
    }
  }

  fn maybe_wait_for(&mut self, ev: &mut TGpuEvent<E>) {
    let t = ev.time();
    let advance = match self.horizons.get(&t.uid) {
      None => true,
      Some(ht) => match ht.causal_cmp(&t) {
        PCausalOrdering::Before => {
          true
        }
        PCausalOrdering::Equal |
        PCausalOrdering::After => {
          false
        }
        PCausalOrdering::Concurrent => {
          panic!("bug");
        }
      },
    };
    if advance {
      #[cfg(feature = "gpu")]
      {
        let mut revent = ev.revent.lock();
        match self.rstream.wait_event(&mut revent) {
          Err(e) => panic!("wait_event failed: {:?} ({})", e, e.get_string()),
          Ok(_) => {}
        }
      }
      self.t.maybe_advance(&t);
      self.horizons.insert(t.uid.clone(), t);
    }
  }
}

#[derive(Clone)]
pub struct TGpuThunk<V=(), E=DefaultEpoch> {
  dev:  i32,
  t:    LamportTime<E>,
  ev:   TGpuEvent<E>,
  //fun:  Option<Box<dyn FnOnce(&mut V, &mut TGpuStream<TT>)>>,
  val:  V,
}

impl<V, E: Ord + Clone> TGpuThunk<V, E> {
  pub fn wait(mut self, stream: TGpuStreamRef<E>) -> TGpuThunkRef<V, E> {
    match self.t.causal_cmp(&stream.this.t) {
      PCausalOrdering::Equal |
      PCausalOrdering::Before => {}
      PCausalOrdering::After => {
        panic!("causal violation");
      }
      PCausalOrdering::Concurrent => {
        stream.this.maybe_wait_for(&mut self.ev);
      }
    }
    TGpuThunkRef{
      dev:    self.dev,
      t:      self.t,
      ev:     self.ev,
      val:    self.val,
    }
  }
}

pub struct TGpuThunkRef<V=(), E=DefaultEpoch> {
  dev:  i32,
  t:    LamportTime<E>,
  ev:   TGpuEvent<E>,
  val:  V,
}
