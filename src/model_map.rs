use crate::config::{Config, ModelCandidate, ProviderConfig};
use std::collections::HashMap;

/// Provider kind with associated runtime config.
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderKind {
    /// Standard OpenAI-compatible: Bearer token auth.
    /// Also used by Google AI Studio (via google_ai shorthand).
    ApiKey,
    /// GCP Vertex AI: access token from metadata server.
    GcpMetadata,
    /// Azure OpenAI: api-key header + deployment URL rewriting.
    AzureOpenAi { api_version: String },
    /// Anthropic: x-api-key header + anthropic-version header.
    Anthropic { version: String },
}

/// Resolved candidate with provider details baked in.
#[derive(Debug, Clone)]
pub struct ResolvedCandidate {
    pub provider_name: String,
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub kind: ProviderKind,
}

/// Maps alias names to resolved candidate lists.
pub struct ModelMap {
    aliases: HashMap<String, Vec<ResolvedCandidate>>,
}

impl ModelMap {
    pub fn from_config(config: &Config) -> Self {
        let providers: HashMap<&str, &ProviderConfig> = config
            .providers
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();

        let mut aliases = HashMap::new();
        for (alias, candidates) in &config.models {
            let resolved: Vec<ResolvedCandidate> = candidates
                .iter()
                .filter_map(|c: &ModelCandidate| {
                    let prov = providers.get(c.provider.as_str())?;
                    let kind = prov.resolved_kind();
                    Some(ResolvedCandidate {
                        provider_name: c.provider.clone(),
                        model: c.model.clone(),
                        base_url: prov.resolved_base_url()?,
                        api_key: prov.api_key.clone(),
                        kind,
                    })
                })
                .collect();
            aliases.insert(alias.clone(), resolved);
        }

        Self { aliases }
    }

    pub fn get(&self, alias: &str) -> Option<&[ResolvedCandidate]> {
        self.aliases.get(alias).map(|v| v.as_slice())
    }

    pub fn alias_names(&self) -> Vec<&str> {
        self.aliases.keys().map(|s| s.as_str()).collect()
    }
}
