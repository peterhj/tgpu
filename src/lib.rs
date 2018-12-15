#![feature(fnbox)]
#![feature(integer_atomics)]

//extern crate cudart;
#[macro_use] extern crate lazy_static;
extern crate parking_lot;

//use cudart::{CudaStream, CudaEvent};
use parking_lot::{Mutex, RwLock, MappedRwLockReadGuard, MappedRwLockWriteGuard};

//use std::boxed::{FnBox};
use std::cmp::{Ordering};
use std::collections::{HashMap};
use std::ops::{Deref};
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

lazy_static! {
  static ref UID_64_CTR:    AtomicU64 = AtomicU64::new(0);
  static ref EPOCH_64_CTR:  AtomicU64 = AtomicU64::new(0);
}

pub trait Fresh {
  fn fresh() -> Self;
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Uid(u64);

impl Fresh for Uid {
  fn fresh() -> Uid {
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
  fn total_cmp(&self, other: &Self) -> TotalOrdering;
}

#[derive(Clone, Debug)]
pub struct TotalTime<E=u64> {
  epoch:    E,
}

impl<E: Ord> TotalOrd for TotalTime<E> {
  fn total_cmp(&self, other: &TotalTime<E>) -> TotalOrdering {
    match self.epoch.cmp(&other.epoch) {
      Ordering::Equal   => TotalOrdering::Equal,
      Ordering::Less    => TotalOrdering::Before,
      Ordering::Greater => TotalOrdering::After,
    }
  }
}

impl Fresh for TotalTime<u64> {
  fn fresh() -> TotalTime<u64> {
    let old_e = EPOCH_64_CTR.fetch_add(1, AtomicOrdering::SeqCst);
    let new_e = old_e + 1;
    assert!(new_e != 0);
    TotalTime{epoch: new_e}
  }
}

pub struct TGpuEvent<TT=TotalTime> {
  uid:      Uid,
  tt:       TT,
  //inner:    CudaEvent,
}

pub struct TGpuEventPool<TT=TotalTime> {
  _dummy:   TT,
  //free:     VecDeque<CudaEvent>,
}

pub struct TGpuStream<TT=TotalTime> {
  uid:      Uid,
  tt:       Option<TT>,
  links:    HashMap<Uid, TT>,
  events:   TGpuEventPool<TT>,
  //inner:    CudaStream,
}

impl<TT: TotalOrd + Fresh + Clone> TGpuStream<TT> {
  pub fn run<F, V>(&mut self, fun: F) -> TGpuThunk<V, TT> where F: FnOnce(&mut V, &mut TGpuStream<TT>) + 'static, V: Default {
    // TODO
    //let boxed_fun = Box::new(fun);
    let new_tt = TT::fresh();
    self.tt = Some(new_tt.clone());
    let mut new_thk = TGpuThunk{
      uid:  self.uid.clone(),
      tt:   new_tt,
      //fun:  Some(boxed_fun),
      fun:  None,
      val:  V::default(),
    };
    //new_thk.force(self);
    (fun)(&mut new_thk.val, self);
    new_thk
  }

  pub fn maybe_wait_for(&mut self, ev: &mut TGpuEvent<TT>) {
  }
}

pub struct TGpuThunk<V, TT=TotalTime> {
  uid:  Uid,
  tt:   TT,
  // TODO
  //ev:   TGpuEvent<TT>,
  fun:  Option<Box<dyn FnOnce(&mut V, &mut TGpuStream<TT>)>>,
  val:  V,
}

impl<V, TT: TotalOrd> TGpuThunk<V, TT> {
  pub fn wait(mut self, tstream: &mut TGpuStream<TT>) -> TGpuThunkRef<V, TT> {
    if self.uid == tstream.uid {
      if let Some(tstream_tt) = tstream.tt.as_ref() {
        match self.tt.total_cmp(tstream_tt) {
          TotalOrdering::Equal |
          TotalOrdering::Before => {
            TGpuThunkRef{
              uid:    self.uid,
              tt:     self.tt,
              //ev:     self.ev,
              val:    self.val,
            }
          }
          TotalOrdering::After => {
            panic!("probable causal violation (one stream)");
          }
        }
      } else {
        panic!("causal violation");
      }
    } else {
      if let Some(tstream_tt) = tstream.tt.as_ref() {
        match self.tt.total_cmp(tstream_tt) {
          TotalOrdering::Equal => {
            panic!("bug: invariant violation");
          }
          TotalOrdering::Before => {
            // TODO
            unimplemented!();
            /*tstream.maybe_wait_for(&mut self.ev);
            TGpuThunkRef{
              uid:    self.uid,
              tt:     self.tt,
              //ev:     self.ev,
              val:    self.val,
            }*/
          }
          TotalOrdering::After => {
            panic!("probable causal violation (two streams)");
          }
        }
      } else {
        panic!("causal violation");
      }
    }
  }

  /*pub fn force(&mut self, tstream: &mut TGpuStream<TT>) {
    match self.fun.take() {
    }
  }*/
}

pub struct TGpuThunkRef<V, TT=TotalTime> {
  uid:  Uid,
  tt:   TT,
  // TODO
  //ev:   TGpuEvent<TT>,
  val:  V,
}
