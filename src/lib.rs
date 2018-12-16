#![feature(integer_atomics)]
#![feature(optin_builtin_traits)]
#![feature(pin)]
#![feature(specialization)]

extern crate cudart;
extern crate gpurepr;
#[macro_use] extern crate lazy_static;
extern crate parking_lot;

use cudart::{CudaStream, CudaEvent, CudaEventStatus};
use gpurepr::{GpuDelay, GpuDelayed, GpuDelayedMut, GpuRegion, GpuRegionMut};
use gpurepr::ctx::{GpuCtxGuard};
use parking_lot::{Mutex};

use std::cmp::{Ordering, max};
use std::collections::{HashMap, VecDeque};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

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
  fn bottom() -> Self where Self: Sized;
  fn next(&self) -> Self where Self: Sized;
  fn total_cmp(&self, other: &Self) -> TotalOrdering;
}

#[derive(Clone, Debug)]
pub struct TotalTime<E=DefaultEpoch> {
  epoch:    E,
}

impl<E: Ord> TotalOrd for TotalTime<E> {
  default fn bottom() -> TotalTime<E> {
    unimplemented!();
  }

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
  fn bottom() -> TotalTime<DefaultEpoch> {
    TotalTime{epoch: 0}
  }

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
  revent:   Arc<Mutex<Option<CudaEvent>>>,
}

pub struct TGpuStreamRef<'a, E=DefaultEpoch> {
  this: &'a mut TGpuStream<E>,
}

impl<'a, E> TGpuStreamRef<'a, E> {
  pub fn device(&mut self) -> i32 {
    self.this.dev
  }

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
  rstream:  CudaStream,
}

impl Default for TGpuStream {
  fn default() -> TGpuStream {
    TGpuStream::new(0)
  }
}

impl<E: Ord> TGpuStream<E> {
  pub fn new(dev: i32) -> TGpuStream<E> {
    let _ctx = GpuCtxGuard::new(dev);
    let rstream = match CudaStream::create() {
      Err(e) => {
        panic!("create stream failed: {:?} ({})", e, e.get_string());
      }
      Ok(rstr) => {
        rstr
      }
    };
    let t = LamportTime{
      uid:  Uid::fresh(),
      tt:   TotalTime::<E>::bottom(),
    };
    TGpuStream{
      dev,
      t,
      horizons: HashMap::new(),
      qlatest:  None,
      queue:    VecDeque::new(),
      rstream,
    }
  }

  fn fresh_time(&self) -> LamportTime<E> {
    LamportTime{
      uid:  self.t.uid.clone(),
      tt:   self.t.tt.next(),
    }
  }
}

impl<E: Ord + Clone> TGpuStream<E> {
  pub fn run<V, F>(&mut self, fun: F) -> TGpuUnsafeThunk<V, E>
  where F: FnOnce(TGpuStreamRef<E>) -> V {
    self.t = self.fresh_time();
    let val = (fun)(TGpuStreamRef{this: self});
    let ev = self.post_event();
    TGpuUnsafeThunk{
      dev:  self.dev,
      ev,
      val,
    }
  }

  fn post_event(&mut self) -> TGpuEvent<E> {
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
    let recycle_revent = match self.queue.front_mut() {
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
            let mut revent = revent.take().unwrap();
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
    let new_revent = if let Some(revent) = recycle_revent {
      match self.queue.pop_front() {
        None => {
          panic!("bug");
        }
        Some(_) => {}
      }
      revent
    } else {
      let mut new_revent = match CudaEvent::create_fastest() {
        Err(e) => {
          panic!("create event failed: {:?} ({})", e, e.get_string());
        }
        Ok(rev) => {
          rev
        }
      };
      match new_revent.record(&mut self.rstream) {
        Err(e) => {
          panic!("record failed: {:?} ({})", e, e.get_string());
        }
        Ok(_) => {}
      }
      new_revent
    };
    let new_ev = TGpuEvent{
      dev:    self.dev,
      t:      self.t.clone(),
      revent: Arc::new(Mutex::new(Some(new_revent))),
    };
    self.qlatest = Some(self.t.clone());
    self.queue.push_back(new_ev.clone());
    new_ev
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
      {
        let mut revent = ev.revent.lock();
        assert!(revent.is_some());
        match self.rstream.wait_event(revent.as_mut().unwrap()) {
          Err(e) => panic!("wait_event failed: {:?} ({})", e, e.get_string()),
          Ok(_) => {}
        }
      }
      self.t.maybe_advance(&ev.t);
      self.horizons.insert(ev.t.uid.clone(), (ev.t.clone(), self.t.clone()));
    }
  }
}

pub struct TGpuUnsafeThunkRef<'stream, V> {
  val:  V,
  _mrk: PhantomData<&'stream ()>,
}

// TODO
impl<'stream, V> Deref for TGpuUnsafeThunkRef<'stream, V> {
  type Target = V;

  fn deref(&self) -> &V {
    &self.val
  }
}

// TODO
impl<'stream, V> DerefMut for TGpuUnsafeThunkRef<'stream, V> {
  fn deref_mut(&mut self) -> &mut V {
    &mut self.val
  }
}

// TODO
impl<'stream, V> GpuDelayed<V> for TGpuUnsafeThunkRef<'stream, V>
where V: GpuDelay + GpuRegion<<V as GpuDelay>::Target> {
  fn dptr(&self) -> *const V::Target {
    unsafe { self.val.as_devptr() }
  }
}

// TODO
impl<'stream, V> GpuDelayedMut<V> for TGpuUnsafeThunkRef<'stream, V>
where V: GpuDelay + GpuRegionMut<<V as GpuDelay>::Target> {
  fn dptr_mut(&self) -> *mut V::Target {
    unsafe { self.val.as_devptr_mut() }
  }
}

pub struct TGpuThunkRef<'stream, V> {
  val:  V,
  _mrk: PhantomData<&'stream ()>,
}

impl<'stream, V> Deref for TGpuThunkRef<'stream, V> {
  type Target = V;

  fn deref(&self) -> &V {
    &self.val
  }
}

impl<'stream, V> GpuDelayed<V> for TGpuThunkRef<'stream, V>
where V: GpuDelay + GpuRegion<<V as GpuDelay>::Target> {
  fn dptr(&self) -> *const V::Target {
    unsafe { self.val.as_devptr() }
  }
}

pub struct TGpuThunkRefMut<'stream, V> {
  val:  V,
  _mrk: PhantomData<&'stream ()>,
}

impl<'stream, V> Deref for TGpuThunkRefMut<'stream, V> {
  type Target = V;

  fn deref(&self) -> &V {
    &self.val
  }
}

impl<'stream, V> GpuDelayed<V> for TGpuThunkRefMut<'stream, V>
where V: GpuDelay + GpuRegion<<V as GpuDelay>::Target> {
  fn dptr(&self) -> *const V::Target {
    unsafe { self.val.as_devptr() }
  }
}

impl<'stream, V> GpuDelayedMut<V> for TGpuThunkRefMut<'stream, V>
where V: GpuDelay + GpuRegionMut<<V as GpuDelay>::Target> {
  fn dptr_mut(&self) -> *mut V::Target {
    unsafe { self.val.as_devptr_mut() }
  }
}

#[derive(Clone)]
pub struct TGpuUnsafeThunk<V=(), E=DefaultEpoch> {
  dev:  i32,
  ev:   TGpuEvent<E>,
  val:  V,
}

impl<V, E: Ord + Clone> TGpuUnsafeThunk<V, E> {
  pub fn sync<'stream>(mut self, stream: &mut TGpuStreamRef<'stream, E>) -> TGpuUnsafeThunkRef<'stream, V> {
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
    TGpuUnsafeThunkRef{
      val:    self.val,
      _mrk:   PhantomData,
    }
  }

  pub fn status(&self) -> CudaEventStatus {
    let mut revent = self.ev.revent.lock();
    match &mut *revent {
      &mut None => {
        CudaEventStatus::Complete
      }
      &mut Some(ref mut rev) => {
        match rev.query() {
          Err(e) => {
            panic!("query failed: {:?} ({})", e, e.get_string());
          }
          Ok(status) => {
            status
          }
        }
      }
    }
  }

  pub fn unchecked_into(self) -> V {
    self.val
  }

  pub fn wait(self) -> V {
    let mut revent = self.ev.revent.lock();
    match &mut *revent {
      &mut None => {
        self.val
      }
      &mut Some(ref mut rev) => {
        match rev.synchronize() {
          Err(e) => {
            panic!("synchronize failed: {:?} ({})", e, e.get_string());
          }
          Ok(_) => {
            self.val
          }
        }
      }
    }
  }
}
