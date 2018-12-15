#![feature(integer_atomics)]
#![feature(optin_builtin_traits)]
#![feature(pin)]
#![feature(specialization)]

//extern crate cudart;
#[macro_use] extern crate lazy_static;
extern crate parking_lot;

//use cudart::{CudaStream, CudaEvent};
use parking_lot::{Mutex, RwLock, MappedRwLockReadGuard, MappedRwLockWriteGuard};

use std::cmp::{Ordering, max};
use std::collections::{HashMap};
use std::ops::{Deref};
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
  default fn maybe_advance(&mut self, other: &LamportTime<E>) {
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
    self.tt.epoch = max(self.tt.epoch, other.tt.epoch + 1);
  }
}

// TODO
#[derive(Clone)]
pub struct TGpuEvent<E=DefaultEpoch> {
  dev:      i32,
  uid:      Uid,
  t:        Option<LamportTime<E>>,
  //event:    CudaEvent,
}

impl<E: Ord + Clone> TGpuEvent<E> {
  pub fn time(&self) -> LamportTime<E> {
    match &self.t {
      None => panic!(),
      Some(ref t) => t.clone(),
    }
  }

  pub fn post(&mut self, new_t: LamportTime<E>, tstream: &mut TGpuStream<E>) {
    assert_eq!(self.uid, tstream.uid, "bug");
    match self.t.take() {
      None => {}
      Some(_) => {
        panic!("bug: double post");
      }
    }
    /*match self.event.record(tstream.cuda_stream()) {
      // TODO
    }*/
    self.t = Some(new_t.clone());
  }
}

pub struct TGpuEventPool<E=DefaultEpoch> {
  _dummy:   E,
  //free:     VecDeque<CudaEvent>,
}

impl<E> TGpuEventPool<E> {
  pub fn make(&mut self) -> TGpuEvent<E> {
    // TODO
    unimplemented!();
  }
}

pub struct TGpuStream<E=DefaultEpoch> {
  dev:      i32,
  uid:      Uid,
  t:        LamportTime<E>,
  horizons: HashMap<Uid, LamportTime<E>>,
  events:   TGpuEventPool<E>,
  //stream:   CudaStream,
}

impl<E> TGpuStream<E> {
  /*pub fn cuda_stream(&mut self) -> &mut CudaStream {
    &mut self.stream
  }*/
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
  where F: FnOnce(&mut V, &mut TGpuStream<E>) + 'static, V: Default {
    // TODO
    let mut ev = self.events.make();
    let mut val = V::default();
    self.t = self.fresh_time();
    (fun)(&mut val, self);
    ev.post(self.t.clone(), self);
    TGpuThunk{
      dev:  self.dev,
      uid:  self.uid.clone(),
      t:    self.t.clone(),
      ev,
      val,
    }
  }

  fn maybe_wait_for(&mut self, ev: &mut TGpuEvent<E>) {
    let et = ev.time();
    let advance = match self.horizons.get(&ev.uid).map(|t| t.clone()) {
      None => true,
      Some(ht) => match ht.causal_cmp(&et) {
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
      // TODO
      /*match self.stream.wait_event(&mut ev.event) {
        Err(e) => panic!("wait_event failed: {:?} ({})", e, e.get_string()),
        Ok(_) => {}
      }*/
      self.t.maybe_advance(&et);
      self.horizons.insert(ev.uid.clone(), et);
    }
  }
}

#[derive(Clone)]
pub struct TGpuThunk<V=(), E=DefaultEpoch> {
  dev:  i32,
  uid:  Uid,
  t:    LamportTime<E>,
  ev:   TGpuEvent<E>,
  //fun:  Option<Box<dyn FnOnce(&mut V, &mut TGpuStream<TT>)>>,
  val:  V,
}

impl<V, E: Ord + Clone> TGpuThunk<V, E> {
  pub fn wait(mut self, tstream: &mut TGpuStream<E>) -> TGpuThunkRef<V, E> {
    match self.t.causal_cmp(&tstream.t) {
      PCausalOrdering::Equal |
      PCausalOrdering::Before => {}
      PCausalOrdering::After => {
        panic!("causal violation");
      }
      PCausalOrdering::Concurrent => {
        tstream.maybe_wait_for(&mut self.ev);
      }
    }
    TGpuThunkRef{
      dev:    self.dev,
      uid:    self.uid,
      t:      self.t,
      ev:     self.ev,
      val:    self.val,
    }
  }
}

pub struct TGpuThunkRef<V=(), E=DefaultEpoch> {
  dev:  i32,
  uid:  Uid,
  t:    LamportTime<E>,
  ev:   TGpuEvent<E>,
  val:  V,
}
