use crate::model_map::ProviderKind;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen: String,
    pub provider: HashMap<String, ProviderConfig>,
    pub model: HashMap<String, Vec<ModelCandidate>>,
    #[serde(default)]
    pub routing: RoutingConfig,
}

fn default_listen() -> String {
    "127.0.0.1:4000".to_string()
}

#[derive(Debug, Default, Deserialize)]
pub struct ProviderConfig {
    /// Base URL for the provider API. Optional if a provider shorthand is set.
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    /// Vertex AI shorthand — derives base_url from project/location.
    pub vertex_ai: Option<VertexAiConfig>,
    /// Azure OpenAI shorthand — derives base_url from resource name.
    pub azure_openai: Option<AzureOpenAiConfig>,
    /// Google AI Studio shorthand — derives base_url, uses API key as query param.
    pub google_ai: Option<GoogleAiConfig>,
    /// Anthropic shorthand — derives base_url, uses x-api-key header.
    pub anthropic: Option<AnthropicConfig>,
}

#[derive(Debug, Deserialize)]
pub struct VertexAiConfig {
    pub project_id: String,
    pub location: String,
}

#[derive(Debug, Deserialize)]
pub struct AzureOpenAiConfig {
    pub resource_name: String,
    pub api_version: String,
}

#[derive(Debug, Deserialize)]
pub struct GoogleAiConfig {
    /// API version prefix in the URL. Defaults to `v1beta`.
    #[serde(default = "GoogleAiConfig::default_api_version")]
    pub api_version: String,
}

impl GoogleAiConfig {
    fn default_api_version() -> String {
        "v1beta".to_string()
    }
}

#[derive(Debug, Deserialize)]
pub struct AnthropicConfig {
    /// Anthropic API version sent as `anthropic-version` header.
    /// Defaults to `2023-06-01`.
    #[serde(default = "AnthropicConfig::default_version")]
    pub version: String,
}

impl AnthropicConfig {
    fn default_version() -> String {
        "2023-06-01".to_string()
    }
}

impl ProviderConfig {
    pub fn resolved_base_url(&self) -> Option<String> {
        self.base_url.clone().or_else(|| {
            if let Some(ref v) = self.vertex_ai {
                return Some(format!(
                    "https://{}-aiplatform.googleapis.com/v1beta1/projects/{}/locations/{}/endpoints/openapi",
                    v.location, v.project_id, v.location
                ));
            }
            if let Some(ref az) = self.azure_openai {
                return Some(format!(
                    "https://{}.openai.azure.com/openai",
                    az.resource_name
                ));
            }
            if let Some(ref g) = self.google_ai {
                return Some(format!(
                    "https://generativelanguage.googleapis.com/{}/openai",
                    g.api_version
                ));
            }
            if self.anthropic.is_some() {
                return Some("https://api.anthropic.com/v1".to_string());
            }
            None
        })
    }

    pub fn resolved_kind(&self) -> ProviderKind {
        if self.vertex_ai.is_some() {
            return ProviderKind::GcpMetadata;
        }
        if let Some(ref az) = self.azure_openai {
            return ProviderKind::AzureOpenAi {
                api_version: az.api_version.clone(),
            };
        }
        if self.google_ai.is_some() {
            return ProviderKind::ApiKey;
        }
        if let Some(ref a) = self.anthropic {
            return ProviderKind::Anthropic {
                version: a.version.clone(),
            };
        }
        ProviderKind::ApiKey
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCandidate {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    pub ewma_alpha: f64,
    pub explore_ratio: f64,
    pub error_threshold: f64,
    pub error_decay_secs: u64,
    pub connect_timeout_secs: u64,
    pub read_timeout_secs: u64,
    pub max_body_bytes: usize,
    pub max_error_window_entries: usize,
    pub shutdown_timeout_secs: u64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.3,
            explore_ratio: 0.2,
            error_threshold: 0.5,
            error_decay_secs: 300,
            connect_timeout_secs: 10,
            read_timeout_secs: 60,
            max_body_bytes: 100 * 1024 * 1024, // 100 MB
            max_error_window_entries: 10_000,
            shutdown_timeout_secs: 30,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read config file '{}': {e}", path.display()))?;
        let interpolated =
            shellexpand::env(&raw).map_err(|e| anyhow::anyhow!("env var expansion failed: {e}"))?;
        let config: Config = toml::from_str(&interpolated)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> anyhow::Result<()> {
        if !(0.0..=1.0).contains(&self.routing.ewma_alpha) {
            anyhow::bail!(
                "ewma_alpha must be between 0.0 and 1.0, got {}",
                self.routing.ewma_alpha
            );
        }
        if !(0.0..=1.0).contains(&self.routing.explore_ratio) {
            anyhow::bail!(
                "explore_ratio must be between 0.0 and 1.0, got {}",
                self.routing.explore_ratio
            );
        }
        if !(0.0..=1.0).contains(&self.routing.error_threshold) {
            anyhow::bail!(
                "error_threshold must be between 0.0 and 1.0, got {}",
                self.routing.error_threshold
            );
        }
        for (name, p) in &self.provider {
            let shorthand_count = [
                p.vertex_ai.is_some(),
                p.azure_openai.is_some(),
                p.google_ai.is_some(),
                p.anthropic.is_some(),
            ]
            .iter()
            .filter(|&&b| b)
            .count();
            if shorthand_count > 1 {
                anyhow::bail!(
                    "provider '{}' has multiple provider shorthands; use only one of vertex_ai, azure_openai, google_ai, or anthropic",
                    name
                );
            }
            let base_url = match p.resolved_base_url() {
                Some(url) => url,
                None => {
                    anyhow::bail!(
                        "provider '{}' must have base_url or a provider shorthand configured",
                        name
                    );
                }
            };
            if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
                anyhow::bail!(
                    "provider '{}' has invalid base_url (must start with http:// or https://): {}",
                    name,
                    base_url
                );
            }
        }
        for (alias, candidates) in &self.model {
            for c in candidates {
                if !self.provider.contains_key(&c.provider) {
                    anyhow::bail!(
                        "model alias '{alias}' references unknown provider '{}'",
                        c.provider
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_ai_derives_base_url() {
        let provider = ProviderConfig {
            vertex_ai: Some(VertexAiConfig {
                project_id: "my-project".into(),
                location: "us-central1".into(),
            }),
            ..Default::default()
        };
        let url = provider.resolved_base_url().unwrap();
        assert!(url.contains("us-central1-aiplatform.googleapis.com"));
        assert!(url.contains("my-project"));
        assert!(url.contains("us-central1"));
    }

    #[test]
    fn vertex_ai_defaults_to_gcp_metadata_auth() {
        let provider = ProviderConfig {
            vertex_ai: Some(VertexAiConfig {
                project_id: "p".into(),
                location: "l".into(),
            }),
            ..Default::default()
        };
        assert_eq!(provider.resolved_kind(), ProviderKind::GcpMetadata);
    }

    #[test]
    fn api_key_provider_defaults_to_api_key_kind() {
        let provider = ProviderConfig {
            base_url: Some("https://api.openai.com/v1".into()),
            api_key: Some("sk-test".into()),
            ..Default::default()
        };
        assert_eq!(provider.resolved_kind(), ProviderKind::ApiKey);
        assert_eq!(
            provider.resolved_base_url().unwrap(),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn timeout_defaults_when_omitted() {
        let toml_str = r#"
[provider.openai]
base_url = "https://api.openai.com/v1"
api_key = "sk-test"

[model]
fast = [{ provider = "openai", model = "gpt-4o-mini" }]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.routing.connect_timeout_secs, 10);
        assert_eq!(config.routing.read_timeout_secs, 60);
    }

    #[test]
    fn timeout_explicit_values_override_defaults() {
        let toml_str = r#"
[provider.openai]
base_url = "https://api.openai.com/v1"
api_key = "sk-test"

[model]
fast = [{ provider = "openai", model = "gpt-4o-mini" }]

[routing]
connect_timeout_secs = 5
read_timeout_secs = 120
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.routing.connect_timeout_secs, 5);
        assert_eq!(config.routing.read_timeout_secs, 120);
    }

    #[test]
    fn explore_ratio_rejects_negative() {
        let toml_str = r#"
[provider.openai]
base_url = "https://api.openai.com/v1"
api_key = "sk-test"

[model]
fast = [{ provider = "openai", model = "gpt-4o-mini" }]

[routing]
explore_ratio = -0.1
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn explore_ratio_rejects_greater_than_one() {
        let toml_str = r#"
[provider.openai]
base_url = "https://api.openai.com/v1"
api_key = "sk-test"

[model]
fast = [{ provider = "openai", model = "gpt-4o-mini" }]

[routing]
explore_ratio = 1.5
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn explore_ratio_accepts_valid_range() {
        for ratio in &[0.0, 0.2, 0.5, 1.0] {
            let toml_str = format!(
                r#"
[provider.openai]
base_url = "https://api.openai.com/v1"
api_key = "sk-test"

[model]
fast = [{{ provider = "openai", model = "gpt-4o-mini" }}]

[routing]
explore_ratio = {}
"#,
                ratio
            );
            let config: Config = toml::from_str(&toml_str).unwrap();
            assert!(
                config.validate().is_ok(),
                "explore_ratio {} should be valid",
                ratio
            );
        }
    }

    #[test]
    fn azure_openai_derives_base_url() {
        let provider = ProviderConfig {
            api_key: Some("key".into()),
            azure_openai: Some(AzureOpenAiConfig {
                resource_name: "my-resource".into(),
                api_version: "2024-10-21".into(),
            }),
            ..Default::default()
        };
        let url = provider.resolved_base_url().unwrap();
        assert_eq!(url, "https://my-resource.openai.azure.com/openai");
    }

    #[test]
    fn azure_openai_resolves_kind() {
        let provider = ProviderConfig {
            api_key: Some("key".into()),
            azure_openai: Some(AzureOpenAiConfig {
                resource_name: "r".into(),
                api_version: "v".into(),
            }),
            ..Default::default()
        };
        assert!(matches!(
            provider.resolved_kind(),
            ProviderKind::AzureOpenAi { .. }
        ));
    }

    #[test]
    fn google_ai_shorthand_derives_url_and_kind() {
        let toml_str = r#"
[provider.gemini]
api_key = "key"
google_ai = {}

[model]
test = [{ provider = "gemini", model = "gemini-2.5-flash" }]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_ok());
        let p = &config.provider["gemini"];
        assert_eq!(
            p.resolved_base_url().unwrap(),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(p.resolved_kind(), ProviderKind::ApiKey);
    }

    #[test]
    fn anthropic_shorthand_derives_url_and_kind() {
        let toml_str = r#"
[provider.anthropic]
api_key = "key"
anthropic = {}

[model]
test = [{ provider = "anthropic", model = "claude-sonnet-4-20250514" }]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_ok());
        let p = &config.provider["anthropic"];
        assert_eq!(
            p.resolved_base_url().unwrap(),
            "https://api.anthropic.com/v1"
        );
        assert!(matches!(p.resolved_kind(), ProviderKind::Anthropic { .. }));
    }

    #[test]
    fn rejects_multiple_provider_shorthands() {
        let toml_str = r#"
[provider.bad]
api_key = "key"
google_ai = {}
anthropic = {}

[model]
test = [{ provider = "bad", model = "test" }]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_err());
    }
}
