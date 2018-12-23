#![feature(integer_atomics)]
#![feature(optin_builtin_traits)]
//#![feature(pin)]
#![feature(specialization)]

extern crate cudart;
extern crate gpurepr;
#[macro_use] extern crate lazy_static;
extern crate parking_lot;

use cudart::{CudaStream, CudaEvent, CudaEventStatus};
use gpurepr::{GpuDev, GpuDom, GpuDelay, GpuDelayed, GpuDelayedMut, GpuRegion, GpuRegionMut};
use gpurepr::ctx::{GpuCtxGuard};
use parking_lot::{Condvar, Mutex};

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

#[derive(Clone, PartialEq, Eq, Debug)]
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

#[derive(Clone, PartialEq, Eq, Debug)]
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
  dev:      GpuDev,
  t:        LamportTime<E>,
  revent:   Arc<Mutex<Option<CudaEvent>>>,
}

pub struct TGpuEventHorizons<E, X> {
  vector:   HashMap<Uid, (LamportTime<E>, X)>,
}

impl<E, X> TGpuEventHorizons<E, X> {
  pub fn new() -> TGpuEventHorizons<E, X> {
    TGpuEventHorizons{vector: HashMap::new()}
  }
}

impl<E: Ord + Clone, X> TGpuEventHorizons<E, X> {
  pub fn can_update(&self, t: LamportTime<E>) -> bool {
    match self.vector.get(&t.uid) {
      None => true,
      Some(&(ref ht, _)) => {
        /*match vt.causal_cmp(&self.t) {
          PCausalOrdering::Before |
          PCausalOrdering::Equal => {}
          PCausalOrdering::After => {
            panic!("causal violation");
          }
          PCausalOrdering::MaybeConcurrent => {
            panic!("bug");
          }
        }*/
        match ht.causal_cmp(&t) {
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
      }
    }
  }

  pub fn update_unchecked<F: FnOnce(Option<X>) -> X>(&mut self, t: LamportTime<E>, fun: F) {
    // TODO
    self.vector.insert(t.uid.clone(), (t.clone(), (fun)(None)));
  }
}

pub struct TGpuStreamRef<'a, E=DefaultEpoch> {
  this: &'a mut TGpuStream<E>,
}

impl<'a, E> TGpuStreamRef<'a, E> {
  fn new(this: &'a mut TGpuStream<E>) -> TGpuStreamRef<'a, E> {
    this._reset_syncs();
    TGpuStreamRef{this}
  }

  pub fn device(&mut self) -> GpuDev {
    self.this.dev
  }

  pub fn cuda_stream(&mut self) -> &mut CudaStream {
    &mut self.this.rstream
  }
}

pub struct TGpuStream<E=DefaultEpoch> {
  dev:      GpuDev,
  lock:     LamportTime<E>,
  t:        LamportTime<E>,
  horizons: TGpuEventHorizons<E, LamportTime<E>>,
  qlatest:  Option<LamportTime<E>>,
  evqueue:  VecDeque<TGpuEvent<E>>,
  thsts:    Vec<Arc<(Mutex<TGpuThunkState<E>>, Condvar)>>,
  mthsts:   Vec<Arc<(Mutex<TGpuThunkState<E>>, Condvar)>>,
  rstream:  CudaStream,
}

impl Default for TGpuStream {
  fn default() -> TGpuStream {
    TGpuStream::new(GpuDev(0))
  }
}

impl<E> TGpuStream<E> {
  fn _reset_syncs(&mut self) {
    assert!(self.thsts.is_empty());
    assert!(self.mthsts.is_empty());
  }

  fn _push_sync(&mut self, thst: Arc<(Mutex<TGpuThunkState<E>>, Condvar)>) {
    self.thsts.push(thst);
  }

  fn _push_sync_mut(&mut self, thst: Arc<(Mutex<TGpuThunkState<E>>, Condvar)>) {
    self.mthsts.push(thst);
  }
}

impl<E: Ord + Eq + Clone> TGpuStream<E> {
  pub fn new(dev: GpuDev) -> TGpuStream<E> {
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
      lock:     t.clone(),
      t,
      horizons: TGpuEventHorizons::new(),
      qlatest:  None,
      evqueue:  VecDeque::new(),
      thsts:    Vec::new(),
      mthsts:   Vec::new(),
      rstream,
    }
  }

  fn fresh_time(&self) -> LamportTime<E> {
    LamportTime{
      uid:  self.t.uid.clone(),
      tt:   self.t.tt.next(),
    }
  }

  fn maybe_sync(&mut self, ev: &mut TGpuEvent<E>) {
    let do_sync = self.horizons.can_update(ev.t.clone());
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
      let new_t = self.t.clone();
      self.horizons.update_unchecked(ev.t.clone(), |_| {
        // TODO: check invariant.
        /*match vt.causal_cmp(&self.t) {
          PCausalOrdering::Before |
          PCausalOrdering::Equal => {}
          PCausalOrdering::After => {
            panic!("causal violation");
          }
          PCausalOrdering::MaybeConcurrent => {
            panic!("bug");
          }
        }*/
        new_t
      });
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
    let recycle_revent = match self.evqueue.front_mut() {
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
      match self.evqueue.pop_front() {
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
    let post_ev = TGpuEvent{
      dev:    self.dev,
      t:      self.t.clone(),
      revent: Arc::new(Mutex::new(Some(new_revent))),
    };
    self.qlatest = Some(self.t.clone());
    self.evqueue.push_back(post_ev.clone());
    for thst_cond in self.thsts.drain(..) {
      {
        let mut thst = thst_cond.0.lock();
        if !(thst.lock.is_some() && thst.lock == Some(self.lock.clone())) {
          panic!("bug");
        }
        match thst.prop {
          Some(TGpuThunkProposal::Sync_) => {}
          _ => panic!("bug"),
        }
        if thst.luse.can_update(post_ev.t.clone()) {
          thst.luse.update_unchecked(post_ev.t.clone(), |_| post_ev.clone());
        }
        thst.prop = None;
      }
      thst_cond.1.notify_all();
    }
    for thst in self.mthsts.drain(..) {
      {
        let mut thst = thst.0.lock();
        if !(thst.lock.is_some() && thst.lock == Some(self.lock.clone())) {
          panic!("bug");
        }
        match thst.prop {
          Some(TGpuThunkProposal::SyncMut) => {}
          _ => panic!("bug"),
        }
        thst.ldef = post_ev.clone();
        thst.luse.vector.clear();
        thst.prop = None;
      }
      thst.1.notify_all();
    }
    post_ev
  }

  pub fn run_wait<V, F>(&mut self, fun: F) -> V
  where F: FnOnce(TGpuStreamRef<E>) -> V {
    self.t = self.fresh_time();
    self.lock = self.t.clone();
    let val = (fun)(TGpuStreamRef::new(self));
    let mut ev = self.post_event();
    let uthk = TGpuUnsafeThunk{val};
    uthk._wait(&mut ev)
  }

  pub fn run<V: GpuDelay, F>(&mut self, fun: F) -> TGpuThunk<V, E>
  where F: FnOnce(TGpuStreamRef<E>) -> V {
    self.t = self.fresh_time();
    self.lock = self.t.clone();
    let val = (fun)(TGpuStreamRef::new(self));
    let ev = self.post_event();
    let thst = Arc::new((Mutex::new(TGpuThunkState{
      lock: None,
      prop: None,
      ldef: ev,
      luse: TGpuEventHorizons::new(),
    }), Condvar::new()));
    TGpuThunk{
      dev:  self.dev,
      val,
      thst,
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
where V: GpuDelay + GpuRegion<<V as GpuDelay>::Data> {
  fn domain(&self) -> GpuDom {
    GpuDom::Dev(self.val.device())
  }

  fn delayed_ptr(&self) -> *const <V as GpuDelay>::Data {
    self.val.as_devptr()
  }
}

// TODO
impl<'stream, V> GpuDelayedMut<V> for TGpuUnsafeThunkRef<'stream, V>
where V: GpuDelay + GpuRegionMut<<V as GpuDelay>::Data> {
  fn delayed_ptr_mut(&self) -> *mut <V as GpuDelay>::Data {
    self.val.as_devptr_mut()
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
where V: GpuDelay + GpuRegion<<V as GpuDelay>::Data> {
  fn domain(&self) -> GpuDom {
    GpuDom::Dev(self.val.device())
  }

  fn delayed_ptr(&self) -> *const <V as GpuDelay>::Data {
    self.val.as_devptr()
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
where V: GpuDelay + GpuRegion<<V as GpuDelay>::Data> {
  fn domain(&self) -> GpuDom {
    GpuDom::Dev(self.val.device())
  }

  fn delayed_ptr(&self) -> *const <V as GpuDelay>::Data {
    self.val.as_devptr()
  }
}

impl<'stream, V> GpuDelayedMut<V> for TGpuThunkRefMut<'stream, V>
where V: GpuDelay + GpuRegionMut<<V as GpuDelay>::Data> {
  fn delayed_ptr_mut(&self) -> *mut <V as GpuDelay>::Data {
    self.val.as_devptr_mut()
  }
}

pub struct TGpuUnsafeThunk<V> {
  val:  V,
}

impl<V> TGpuUnsafeThunk<V> {
  pub fn _wait<E>(self, ev: &mut TGpuEvent<E>) -> V {
    let mut revent = ev.revent.lock();
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

enum TGpuThunkProposal {
  Sync_,
  SyncMut,
}

struct TGpuThunkState<E> {
  lock: Option<LamportTime<E>>,
  prop: Option<TGpuThunkProposal>,
  ldef: TGpuEvent<E>,
  luse: TGpuEventHorizons<E, TGpuEvent<E>>,
}

#[derive(Clone)]
pub struct TGpuThunk<V=(), E=DefaultEpoch> {
  dev:  GpuDev,
  val:  V,
  thst: Arc<(Mutex<TGpuThunkState<E>>, Condvar)>,
}

impl<V, E: Ord + Clone> TGpuThunk<V, E> {
  pub fn _sync<'stream>(mut self, stream: &mut TGpuStreamRef<'stream, E>) -> TGpuUnsafeThunkRef<'stream, V> {
    let mut thst = self.thst.0.lock();
    match thst.ldef.t.causal_cmp(&stream.this.t) {
      PCausalOrdering::Before |
      PCausalOrdering::Equal => {}
      PCausalOrdering::After => {
        panic!("causal violation");
      }
      PCausalOrdering::MaybeConcurrent => {
        stream.this.maybe_sync(&mut thst.ldef);
      }
    }
    TGpuUnsafeThunkRef{
      val:    self.val,
      _mrk:   PhantomData,
    }
  }

  pub fn status(&self) -> CudaEventStatus {
    let mut thst = self.thst.0.lock();
    let mut revent = thst.ldef.revent.lock();
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

  pub fn wait(self) -> V {
    let mut thst = self.thst.0.lock();
    let mut revent = thst.ldef.revent.lock();
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

impl<V: GpuDelay, E: Ord + Eq + Clone> TGpuThunk<V, E> {
  pub fn sync<'stream>(self, stream: &mut TGpuStreamRef<'stream, E>) -> TGpuThunkRef<'stream, V> {
    let mut thst = self.thst.0.lock();
    loop {
      match (&thst.lock, &thst.prop) {
        (&None, &None) => {
          thst.lock = Some(stream.this.lock.clone());
          thst.prop = Some(TGpuThunkProposal::Sync_);
          break;
        }
        (&Some(ref lock), &Some(ref prop)) => {
          if *lock == stream.this.lock {
            match prop {
              &TGpuThunkProposal::Sync_ => {
                break;
              }
              &TGpuThunkProposal::SyncMut => {
                panic!("double sync");
              }
            }
          } else {
            self.thst.1.wait(&mut thst);
          }
        }
        _ => panic!("invalid state"),
      }
    }
    match thst.ldef.t.causal_cmp(&stream.this.t) {
      PCausalOrdering::Before |
      PCausalOrdering::Equal => {}
      PCausalOrdering::After => {
        panic!("causal violation");
      }
      PCausalOrdering::MaybeConcurrent => {
        stream.this.maybe_sync(&mut thst.ldef);
      }
    }
    stream.this._push_sync(self.thst.clone());
    TGpuThunkRef{
      val:    self.val,
      _mrk:   PhantomData,
    }
  }

  pub fn sync_mut<'stream>(self, stream: &mut TGpuStreamRef<'stream, E>) -> TGpuThunkRefMut<'stream, V> {
    let mut thst = self.thst.0.lock();
    loop {
      match (&thst.lock, &thst.prop) {
        (&None, &None) => {
          thst.lock = Some(stream.this.lock.clone());
          thst.prop = Some(TGpuThunkProposal::SyncMut);
          break;
        }
        (&Some(ref lock), &Some(_)) => {
          if *lock == stream.this.lock {
            panic!("double sync");
          } else {
            self.thst.1.wait(&mut thst);
          }
        }
        _ => panic!("invalid state"),
      }
    }
    for (_, (_, ref mut luse_ev)) in thst.luse.vector.iter_mut() {
      match luse_ev.t.causal_cmp(&stream.this.t) {
        PCausalOrdering::Before |
        PCausalOrdering::Equal => {}
        PCausalOrdering::After => {
          panic!("causal violation");
        }
        PCausalOrdering::MaybeConcurrent => {
          stream.this.maybe_sync(luse_ev);
        }
      }
    }
    match thst.ldef.t.causal_cmp(&stream.this.t) {
      PCausalOrdering::Before |
      PCausalOrdering::Equal => {}
      PCausalOrdering::After => {
        panic!("causal violation");
      }
      PCausalOrdering::MaybeConcurrent => {
        stream.this.maybe_sync(&mut thst.ldef);
      }
    }
    stream.this._push_sync_mut(self.thst.clone());
    TGpuThunkRefMut{
      val:    self.val,
      _mrk:   PhantomData,
    }
  }
}
