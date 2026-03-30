use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::info;

use llmrouter::gcp_auth::GcpTokenProvider;
use llmrouter::metrics::Metrics;
use llmrouter::model_map::{ModelMap, ProviderKind};
use llmrouter::router::RoundRobinState;
use llmrouter::server::AppState;
use llmrouter::session::SessionStore;
use llmrouter::tracker::Tracker;

#[derive(Parser)]
#[command(name = "llmrouter", about = "Lightweight LLM load-balancing sidecar")]
struct Cli {
    /// Path to config YAML file
    #[arg(short, long, default_value = "config.yaml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "llmrouter=info".parse().unwrap()),
        )
        .init();

    eprintln!(
        "\n\
         \x20     ┌► llm\n\
         \x20  ►──┼► llm\n\
         \x20     └► llm\n\
         \x20  llmrouter v{}\n",
        env!("CARGO_PKG_VERSION")
    );

    let cli = Cli::parse();
    let config = llmrouter::config::Config::load(&cli.config)?;

    info!(listen = %config.listen, "starting llmrouter");
    info!(
        providers = config.providers.len(),
        models = config.models.len(),
        "loaded config"
    );

    let model_map = ModelMap::from_config(&config);

    let mut tracker_inner = Tracker::new(
        config.routing.ewma_alpha,
        config.routing.error_decay_secs,
        config.routing.error_threshold,
        config.routing.max_error_window_entries,
    );
    let mut rr_state = RoundRobinState::new();

    for (alias, candidates) in &config.models {
        rr_state.register_alias(alias.clone());
        for c in candidates {
            tracker_inner.register((c.provider.clone(), c.model.clone()));
        }
    }

    let client = reqwest::Client::builder()
        .pool_max_idle_per_host(10)
        .connect_timeout(std::time::Duration::from_secs(
            config.routing.connect_timeout_secs,
        ))
        .read_timeout(std::time::Duration::from_secs(
            config.routing.read_timeout_secs,
        ))
        .build()?;

    let needs_gcp = config
        .providers
        .iter()
        .any(|p| p.resolved_kind() == ProviderKind::GcpMetadata);
    let gcp_token_provider = if needs_gcp {
        info!("GCP metadata auth enabled");
        Some(GcpTokenProvider::new(client.clone()))
    } else {
        None
    };

    let explore_ratio = config.routing.explore_ratio;
    let max_body_bytes = config.routing.max_body_bytes;
    let session_store = SessionStore::new(
        std::time::Duration::from_secs(config.routing.session_ttl_secs),
        config.routing.max_sessions,
    );

    let metrics = Metrics::new();
    let label_triples: Vec<_> = model_map
        .alias_names()
        .iter()
        .flat_map(|alias| {
            model_map
                .get(alias)
                .unwrap_or_default()
                .iter()
                .map(|c| (alias.to_string(), c.provider_name.clone(), c.model.clone()))
        })
        .collect();
    metrics.init_zero(&label_triples);

    let state = Arc::new(AppState {
        model_map,
        tracker: tracker_inner,
        rr_state,
        client,
        explore_ratio,
        gcp_token_provider,
        session_store,
        max_body_bytes,
        metrics,
    });

    let addr: SocketAddr = config.listen.parse()?;
    let listener = TcpListener::bind(addr).await?;
    info!(%addr, "listening");

    let shutdown = async {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate()).expect("failed to listen for SIGTERM");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = sigterm.recv() => {},
        }
        info!("received shutdown signal");
    };

    llmrouter::server::run_server(listener, state, shutdown).await;

    info!("server stopped");
    Ok(())
}
