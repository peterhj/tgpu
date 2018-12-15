#![feature(integer_atomics)]
#![feature(optin_builtin_traits)]
#![feature(pin)]

//extern crate cudart;
#[macro_use] extern crate lazy_static;
extern crate parking_lot;

//use cudart::{CudaStream, CudaEvent};
use parking_lot::{Mutex, RwLock, MappedRwLockReadGuard, MappedRwLockWriteGuard};

use std::cmp::{Ordering};
use std::collections::{HashMap};
use std::ops::{Deref};
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

pub mod ctx;

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
  dev:      i32,
  uid:      Uid,
  tt:       Option<TT>,
  //event:    CudaEvent,
}

impl<TT: TotalOrd + Clone> TGpuEvent<TT> {
  pub fn total_time(&self) -> TT {
    match self.tt {
      None => panic!(),
      Some(ref tt) => tt.clone(),
    }
  }

  pub fn post(&mut self, new_tt: TT, tstream: &mut TGpuStream<TT>) {
    assert_eq!(self.uid, tstream.uid, "bug");
    match self.tt.take() {
      None => {}
      Some(_) => {
        panic!("bug: double post");
      }
    }
    match tstream.tt.take() {
      None => {}
      Some(tstream_tt) => {
        match tstream_tt.total_cmp(&new_tt) {
          TotalOrdering::Equal |
          TotalOrdering::After => {
            panic!("causal violation");
          }
          TotalOrdering::Before => {}
        }
      }
    }
    /*match self.event.record(tstream.cuda_stream()) {
      // TODO
    }*/
    self.tt = Some(new_tt.clone());
    tstream.tt = Some(new_tt);
  }
}

pub struct TGpuEventPool<TT=TotalTime> {
  _dummy:   TT,
  //free:     VecDeque<CudaEvent>,
}

pub struct TGpuStream<TT=TotalTime> {
  dev:      i32,
  uid:      Uid,
  tt:       Option<TT>,
  horizons: HashMap<Uid, TT>,
  events:   TGpuEventPool<TT>,
  //stream:   CudaStream,
}

impl<TT> TGpuStream<TT> {
  /*pub fn cuda_stream(&mut self) -> &mut CudaStream {
    &mut self.stream
  }*/
}

impl<TT: TotalOrd + Fresh + Clone> TGpuStream<TT> {
  pub fn run<F, V>(&mut self, fun: F) -> TGpuThunk<V, TT>
  where F: FnOnce(&mut V, &mut TGpuStream<TT>) + 'static, V: Default {
    // TODO
    //let mut ev = self.events.make(self.uid.clone());
    let mut val = V::default();
    (fun)(&mut val, self);
    let new_tt = TT::fresh();
    //ev.post(new_tt, &mut self.stream);
    self.tt = Some(new_tt.clone());
    TGpuThunk{
      dev:  self.dev,
      uid:  self.uid.clone(),
      tt:   new_tt,
      // TODO
      //ev:   _,
      val,
    }
  }

  pub fn maybe_wait_for(&mut self, ev: &mut TGpuEvent<TT>) {
    let ett = ev.total_time();
    let advance = match self.horizons.get(&ev.uid).map(|tt| tt.clone()) {
      None => true,
      Some(htt) => match htt.total_cmp(&ett) {
        TotalOrdering::Before => true,
        TotalOrdering::Equal |
        TotalOrdering::After => false,
      },
    };
    if advance {
      // TODO
      /*match self.stream.wait_event(&mut ev.event) {
        Err(e) => panic!("wait_event failed: {:?} ({})", e, e.get_string()),
        Ok(_) => {}
      }*/
      self.horizons.insert(ev.uid.clone(), ett);
    }
  }
}

pub struct TGpuThunk<V, TT=TotalTime> {
  dev:  i32,
  uid:  Uid,
  tt:   TT,
  // TODO
  //ev:   TGpuEvent<TT>,
  //fun:  Option<Box<dyn FnOnce(&mut V, &mut TGpuStream<TT>)>>,
  val:  V,
}

impl<V: Clone, TT: Clone> Clone for TGpuThunk<V, TT> {
  fn clone(&self) -> TGpuThunk<V, TT> {
    TGpuThunk{
      dev:  self.dev,
      uid:  self.uid.clone(),
      tt:   self.tt.clone(),
      //ev:   self.ev.clone(),
      val:  self.val.clone(),
    }
  }
}

impl<V, TT: TotalOrd> TGpuThunk<V, TT> {
  pub fn wait(mut self, tstream: &mut TGpuStream<TT>) -> TGpuThunkRef<V, TT> {
    if self.uid == tstream.uid {
      if let Some(tstream_tt) = tstream.tt.as_ref() {
        match self.tt.total_cmp(tstream_tt) {
          TotalOrdering::Equal |
          TotalOrdering::Before => {
            TGpuThunkRef{
              dev:    self.dev,
              uid:    self.uid,
              tt:     self.tt,
              //ev:     self.ev,
              val:    self.val,
            }
          }
          TotalOrdering::After => {
            panic!("causal violation (one stream)");
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
              dev:    self.dev,
              uid:    self.uid,
              tt:     self.tt,
              //ev:     self.ev,
              val:    self.val,
            }*/
          }
          TotalOrdering::After => {
            panic!("causal violation (two streams)");
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
  dev:  i32,
  uid:  Uid,
  tt:   TT,
  // TODO
  //ev:   TGpuEvent<TT>,
  val:  V,
}
