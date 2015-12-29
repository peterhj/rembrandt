extern crate iron;
extern crate mount;
#[macro_use]
extern crate lazy_static;

use iron::prelude::*;
use iron::{BeforeMiddleware, Handler};
use iron::mime::{Mime};
use iron::status;
use iron::typemap;
use mount::{Mount};

use std::sync::{Arc, RwLock};
use std::thread::{JoinHandle, spawn};

lazy_static! {
  static ref ASSET_HTML_INDEX:  &'static [u8] = include_bytes!("../assets/html/index.html");
  //static ref ASSET_CSS_BOOT:    &'static [u8] = include_bytes!("../assets/css/bootstrap.min.v3.3.6.css");
  static ref ASSET_CSS_MG:      &'static [u8] = include_bytes!("../assets/css/mg.v2.7.0.css");
  static ref ASSET_JS_JQUERY:   &'static [u8] = include_bytes!("../assets/js/jquery.min.v2.1.4.js");
  //static ref ASSET_JS_BOOT:     &'static [u8] = include_bytes!("../assets/js/bootstrap.min.v3.3.6.js");
  static ref ASSET_JS_D3:       &'static [u8] = include_bytes!("../assets/js/d3.min.v3.5.12.js");
  static ref ASSET_JS_MG:       &'static [u8] = include_bytes!("../assets/js/mg.min.v2.7.0.js");

  static ref MIME_HTML: Mime = "text/html".parse().unwrap();
  static ref MIME_CSS:  Mime = "text/css".parse().unwrap();
  static ref MIME_JS:   Mime = "application/javascript".parse().unwrap();
  static ref MIME_JSON: Mime = "application/json".parse().unwrap();
}

fn index_h(_: &mut Request) -> IronResult<Response> {
  Ok(Response::with((status::Ok, MIME_HTML.clone(), *ASSET_HTML_INDEX)))
}

fn css_mg_h(_: &mut Request) -> IronResult<Response> {
  Ok(Response::with((status::Ok, MIME_CSS.clone(), *ASSET_CSS_MG)))
}

fn js_jquery_h(_: &mut Request) -> IronResult<Response> {
  Ok(Response::with((status::Ok, MIME_JS.clone(), *ASSET_JS_JQUERY)))
}

fn js_d3_h(_: &mut Request) -> IronResult<Response> {
  Ok(Response::with((status::Ok, MIME_JS.clone(), *ASSET_JS_D3)))
}

fn js_mg_h(_: &mut Request) -> IronResult<Response> {
  Ok(Response::with((status::Ok, MIME_JS.clone(), *ASSET_JS_MG)))
}

fn api_v0_train_loss_json_h(req: &mut Request) -> IronResult<Response> {
  let key = req.extensions.get::<ReportServerStateKey>().unwrap();
  let inner_state = key.state.inner.read().unwrap();
  // TODO(20151229)
  Ok(Response::with((status::Ok, MIME_JSON.clone(), "[{\"x\": 1, \"y\": 1}, {\"x\": 2, \"y\": 2}]")))
}

#[derive(Clone, Copy)]
pub enum ReportServerUpdate {
  TrainingIteration{t: i32, loss: f32, acc: f32},
  ValidationRun{t: i32, loss: f32, acc: f32},
}

struct ReportServerInnerState {
  train_loss_series:    Vec<(i32, f32)>,
  train_acc_series:     Vec<(i32, f32)>,
  valid_loss_series:    Vec<(i32, f32)>,
  valid_acc_series:     Vec<(i32, f32)>,
}

pub struct ReportServerState {
  //rx:       Mutex<Receiver<ReportServerUpdate>>,
  inner:    RwLock<ReportServerInnerState>,
}

impl ReportServerState {
  //pub fn new() -> (ReportServerState, Sender<ReportServerUpdate>) {
  pub fn new() -> ReportServerState {
    //let (tx, rx) = channel();
    //(
    ReportServerState{
      //rx:       Mutex::new(rx),
      inner:    RwLock::new(ReportServerInnerState{
        train_loss_series:  vec![],
        train_acc_series:   vec![],
        valid_loss_series:  vec![],
        valid_acc_series:   vec![],
      }),
    }//, tx)
  }

  pub fn update(&self, msg: ReportServerUpdate) {
    let mut inner = self.inner.write().unwrap();
    match msg {
      ReportServerUpdate::TrainingIteration{t, loss, acc} => {
        inner.train_loss_series.push((t, loss));
        inner.train_acc_series.push((t, acc));
      }
      ReportServerUpdate::ValidationRun{t, loss, acc} => {
        inner.valid_loss_series.push((t, loss));
        inner.valid_acc_series.push((t, acc));
      }
    }
  }
}

#[derive(Clone)]
struct ReportServerStateKey {
  state:    Arc<ReportServerState>,
}

impl ReportServerStateKey {
  pub fn new(state: Arc<ReportServerState>) -> ReportServerStateKey {
    ReportServerStateKey{
      state:    state,
    }
  }

  pub fn chain<H: Handler>(&self, h: H) -> Chain {
    let mut chain = Chain::new(h);
    chain.link_before(ReportServerStateKey{
      state:    self.state.clone(),
    });
    chain
  }
}

impl typemap::Key for ReportServerStateKey {
  type Value = ReportServerStateKey;
}

impl BeforeMiddleware for ReportServerStateKey {
  fn before(&self, req: &mut Request) -> IronResult<()> {
    req.extensions.insert::<ReportServerStateKey>(self.clone());
    Ok(())
  }
}

pub struct ReportServer;

impl ReportServer {
  pub fn spawn() -> (JoinHandle<()>, Arc<ReportServerState>) {
    let state = Arc::new(ReportServerState::new());
    let state_key = ReportServerStateKey::new(state.clone());
    let child = spawn(move || {
      let mut mount = Mount::new();
      mount.mount("/", index_h);
      //mount.mount("/css/bootstrap.min.css", css_bootstrap_h);
      mount.mount("/css/mg.css", css_mg_h);
      mount.mount("/js/jquery.min.js", js_jquery_h);
      //mount.mount("/js/bootstrap.min.js", js_bootstrap_h);
      mount.mount("/js/d3.min.js", js_d3_h);
      mount.mount("/js/mg.min.js", js_mg_h);
      mount.mount("/api/v0/train_loss.json", state_key.chain(api_v0_train_loss_json_h));
      Iron::new(mount).http("127.0.0.1:8080").unwrap();
    });
    (child, state)
  }
}
