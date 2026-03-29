use reqwest::Client;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::debug;

const METADATA_URL: &str =
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";

/// Cached GCP access token fetched from the metadata server.
struct CachedToken {
    token: String,
    expires_at: Instant,
}

/// Fetches and caches GCP access tokens from the GKE metadata server.
/// Safe to share across requests. The lock is held through the fetch to
/// prevent concurrent requests from redundantly hitting the metadata server.
pub struct GcpTokenProvider {
    client: Client,
    cache: Mutex<Option<CachedToken>>,
}

impl GcpTokenProvider {
    pub fn new(client: Client) -> Arc<Self> {
        Arc::new(Self {
            client,
            cache: Mutex::new(None),
        })
    }

    /// Get a valid access token, refreshing from the metadata server if needed.
    pub async fn get_token(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let mut cache = self.cache.lock().await;

        // Return cached token if still valid (refresh 60s before expiry)
        if let Some(ref cached) = *cache {
            if cached.expires_at > Instant::now() + Duration::from_secs(60) {
                return Ok(cached.token.clone());
            }
        }

        // Cache miss or expired — fetch while holding the lock so only
        // one request hits the metadata server at a time.
        debug!("refreshing GCP access token from metadata server");
        let resp = self
            .client
            .get(METADATA_URL)
            .header("Metadata-Flavor", "Google")
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("GCP metadata server returned {status}: {body}").into());
        }

        let body: serde_json::Value = resp.json().await?;
        let token = body["access_token"]
            .as_str()
            .ok_or("missing access_token in metadata response")?
            .to_string();
        let expires_in = body["expires_in"].as_u64().unwrap_or(3600);

        let expires_at = Instant::now() + Duration::from_secs(expires_in);
        debug!(expires_in, "GCP token refreshed");

        *cache = Some(CachedToken {
            token: token.clone(),
            expires_at,
        });

        Ok(token)
    }
}
