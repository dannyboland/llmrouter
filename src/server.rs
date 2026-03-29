use bytes::Bytes;
use http_body_util::{BodyExt, Limited};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use std::convert::Infallible;
use std::future::Future;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{debug, error, info, warn};

use crate::gcp_auth::GcpTokenProvider;
use crate::model_map::ModelMap;
use crate::proxy;
use crate::router::{self, candidate_key, RoundRobinState};
use crate::session::SessionStore;
use crate::tracker::Tracker;

/// Shared application state.
pub struct AppState {
    pub model_map: ModelMap,
    pub tracker: Tracker,
    pub rr_state: RoundRobinState,
    pub client: reqwest::Client,
    pub explore_ratio: f64,
    pub gcp_token_provider: Option<Arc<GcpTokenProvider>>,
    pub session_store: SessionStore,
    pub max_body_bytes: usize,
}

type BoxBody = http_body_util::Either<
    http_body_util::Full<Bytes>,
    http_body_util::StreamBody<
        futures_util::stream::BoxStream<'static, Result<hyper::body::Frame<Bytes>, Infallible>>,
    >,
>;

fn json_error(status: hyper::StatusCode, message: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
        }
    });
    let bytes = Bytes::from(serde_json::to_vec(&body).unwrap());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(http_body_util::Either::Left(http_body_util::Full::new(
            bytes,
        )))
        .unwrap()
}

pub async fn handle_request(
    req: Request<hyper::body::Incoming>,
    state: Arc<AppState>,
) -> Result<Response<BoxBody>, Infallible> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();

    debug!(%method, %path, "incoming request");

    // GET endpoints
    if method == hyper::Method::GET {
        if path == "/health" || path == "/healthz" {
            let body = Bytes::from(r#"{"status":"ok"}"#);
            return Ok(json_response(200, body));
        }
        if path == "/status" {
            return Ok(build_status_response(&state));
        }
        return Ok(json_error(hyper::StatusCode::NOT_FOUND, "not found"));
    }

    if method != hyper::Method::POST {
        return Ok(json_error(
            hyper::StatusCode::METHOD_NOT_ALLOWED,
            "method not allowed",
        ));
    }

    let session_id = req
        .headers()
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let max_body = state.max_body_bytes;
    let body_bytes = match Limited::new(req, max_body).collect().await {
        Ok(collected) => collected.to_bytes(),
        Err(e) => {
            error!("failed to read request body: {e}");
            return Ok(json_error(
                hyper::StatusCode::PAYLOAD_TOO_LARGE,
                &format!("request body too large (max {} MB)", max_body / 1024 / 1024),
            ));
        }
    };

    let alias = match proxy::extract_json_field(&body_bytes, "model") {
        Some((_, val)) => val.to_string(),
        None => {
            return Ok(json_error(
                hyper::StatusCode::BAD_REQUEST,
                "missing 'model' field in request body",
            ));
        }
    };

    let candidates = match state.model_map.get(&alias) {
        Some(c) => c,
        None => {
            let available = state.model_map.alias_names();
            return Ok(json_error(
                hyper::StatusCode::BAD_REQUEST,
                &format!(
                    "unknown model alias '{}', available: {:?}",
                    alias, available
                ),
            ));
        }
    };

    let pinned = session_id.as_ref().and_then(|sid| {
        state.session_store.get(&alias, sid).and_then(|pinned_key| {
            let found = candidates
                .iter()
                .find(|c| candidate_key(c) == pinned_key)?;
            if state.tracker.is_degraded(&pinned_key) {
                debug!(session_id = %sid, provider = %pinned_key.0, "previous model in session now degraded, breaking affinity");
                state.session_store.remove(&alias, sid);
                None
            } else {
                Some(found)
            }
        })
    });

    let candidate = if let Some(c) = pinned {
        c
    } else {
        let c = match router::select_candidate(
            &alias,
            candidates,
            &state.tracker,
            &state.rr_state,
            state.explore_ratio,
        ) {
            Some(c) => c,
            None => {
                return Ok(json_error(
                    hyper::StatusCode::BAD_GATEWAY,
                    &format!("no healthy candidate for model alias '{alias}'"),
                ));
            }
        };
        if let Some(ref sid) = session_id {
            state.session_store.insert(&alias, sid, candidate_key(c));
        }
        c
    };

    let key = candidate_key(candidate);

    let is_streaming = proxy::extract_json_bool(&body_bytes, "stream").unwrap_or(false);

    if let Some(stats) = state.tracker.get(&key) {
        stats.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    info!(
        alias = %alias,
        provider = %candidate.provider_name,
        model = %candidate.model,
        streaming = is_streaming,
        "routing request"
    );

    let result = proxy::forward_request(
        &state.client,
        candidate,
        &path,
        body_bytes,
        is_streaming,
        state.gcp_token_provider.as_ref(),
    )
    .await;

    match result {
        Ok(proxy_result) => {
            state.tracker.record_ttfc(&key, proxy_result.ttfc);

            if proxy_result.status.is_success() {
                state.tracker.record_success(&key);
            } else {
                state.tracker.record_error(&key);
            }

            if let Some(stats) = state.tracker.get(&key) {
                stats.in_flight.fetch_sub(1, Ordering::Relaxed);
            }

            debug!(
                provider = %candidate.provider_name,
                model = %candidate.model,
                status = %proxy_result.status,
                ttfc_ms = proxy_result.ttfc.as_millis(),
                "response received"
            );

            let mut builder = Response::builder().status(proxy_result.status);
            for (k, v) in &proxy_result.headers {
                builder = builder.header(k, v);
            }
            builder = builder.header("x-llmrouter-provider", &candidate.provider_name);
            if let Some(ref sid) = session_id {
                builder = builder.header("x-llmrouter-session-id", sid.as_str());
            }

            let body = proxy::into_hyper_body(proxy_result.body);
            Ok(builder.body(body).unwrap())
        }
        Err(e) => {
            warn!(
                provider = %candidate.provider_name,
                model = %candidate.model,
                error = %e,
                "upstream request failed"
            );

            state.tracker.record_error(&key);

            if let Some(stats) = state.tracker.get(&key) {
                stats.in_flight.fetch_sub(1, Ordering::Relaxed);
            }

            Ok(json_error(
                hyper::StatusCode::BAD_GATEWAY,
                &format!("upstream error: {e}"),
            ))
        }
    }
}

fn json_response(status: u16, body: Bytes) -> Response<BoxBody> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(http_body_util::Either::Left(http_body_util::Full::new(
            body,
        )))
        .unwrap()
}

/// Run the server accept loop until the shutdown signal fires.
/// After shutdown, in-flight connections are allowed to complete gracefully.
pub async fn run_server(
    listener: TcpListener,
    state: Arc<AppState>,
    shutdown: impl Future<Output = ()>,
) {
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            result = listener.accept() => {
                let (stream, remote) = match result {
                    Ok(conn) => conn,
                    Err(e) => {
                        error!(error = %e, "accept error");
                        continue;
                    }
                };
                let io = TokioIo::new(stream);
                let state = state.clone();

                tokio::spawn(async move {
                    let service = service_fn(move |req| {
                        let state = state.clone();
                        async move { handle_request(req, state).await }
                    });

                    if let Err(e) = http1::Builder::new()
                        .serve_connection(io, service)
                        .await
                    {
                        if !e.to_string().contains("connection closed") {
                            error!(remote = %remote, error = %e, "connection error");
                        }
                    }
                });
            }
            _ = &mut shutdown => {
                info!("shutdown signal received, stopping accept loop");
                break;
            }
        }
    }
}

fn build_status_response(state: &AppState) -> Response<BoxBody> {
    let mut candidates = Vec::new();

    for (key, stats) in &state.tracker.stats {
        let ewma = stats.ewma_ms.load(Ordering::Relaxed);
        let degraded = state.tracker.is_degraded(key);
        let status = if ewma == u64::MAX {
            "cold"
        } else if degraded {
            "degraded"
        } else {
            "warm"
        };

        candidates.push(serde_json::json!({
            "provider": key.0,
            "model": key.1,
            "status": status,
            "ewma_ms": if ewma == u64::MAX { None } else { Some(ewma) },
            "in_flight": stats.in_flight.load(Ordering::Relaxed),
            "error_rate": stats.error_rate(),
        }));
    }

    let body = serde_json::json!({
        "candidates": candidates,
        "active_sessions": state.session_store.len(),
    });
    let bytes = Bytes::from(serde_json::to_vec_pretty(&body).unwrap());
    json_response(200, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_map::ModelMap;
    use crate::router::RoundRobinState;
    use crate::session::SessionStore;
    use crate::tracker::Tracker;
    use tokio::sync::oneshot;

    fn test_state() -> Arc<AppState> {
        let config: crate::config::Config = serde_yaml::from_str(
            r#"
providers:
  - name: test
    base_url: "http://localhost:9999"
    api_key: "fake"
models:
  fast:
    - provider: test
      model: test-model
"#,
        )
        .unwrap();

        let model_map = ModelMap::from_config(&config);
        let mut tracker = Tracker::new(0.3, 30, 0.5, 10_000);
        let mut rr_state = RoundRobinState::new();
        for (alias, candidates) in &config.models {
            rr_state.register_alias(alias.clone());
            for c in candidates {
                tracker.register((c.provider.clone(), c.model.clone()));
            }
        }

        Arc::new(AppState {
            model_map,
            tracker,
            rr_state,
            client: reqwest::Client::new(),
            explore_ratio: 0.2,
            gcp_token_provider: None,
            session_store: SessionStore::new(std::time::Duration::from_secs(1800), 100_000),
            max_body_bytes: 100 * 1024 * 1024,
        })
    }

    #[tokio::test]
    async fn health_endpoint_returns_ok() {
        let state = test_state();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (_tx, rx) = oneshot::channel::<()>();

        tokio::spawn(run_server(listener, state, async {
            let _ = rx.await;
        }));

        let resp = reqwest::get(format!("http://{addr}/health")).await.unwrap();
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn shutdown_signal_stops_accept_loop() {
        let state = test_state();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = oneshot::channel::<()>();

        let server_handle = tokio::spawn(run_server(listener, state, async {
            let _ = rx.await;
        }));

        // Confirm server is up
        let resp = reqwest::get(format!("http://{addr}/health")).await.unwrap();
        assert_eq!(resp.status(), 200);

        // Send shutdown signal
        tx.send(()).unwrap();

        // Server task should complete
        tokio::time::timeout(std::time::Duration::from_secs(2), server_handle)
            .await
            .expect("server did not shut down within 2s")
            .expect("server task panicked");

        // New connections should be refused
        let result = reqwest::get(format!("http://{addr}/health")).await;
        assert!(
            result.is_err(),
            "expected connection refused after shutdown"
        );
    }
}
