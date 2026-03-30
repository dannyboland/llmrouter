use std::time::Duration;

use crate::tracker::CandidateKey;

type SessionKey = (String, String);

/// Maps (alias, session_id) to a pinned candidate.
/// Entries are evicted after TTL (measured from last access).
pub struct SessionStore {
    cache: moka::sync::Cache<SessionKey, CandidateKey>,
}

impl SessionStore {
    pub fn new(ttl: Duration, max_entries: u64) -> Self {
        Self {
            cache: moka::sync::Cache::builder()
                .max_capacity(max_entries)
                .time_to_idle(ttl)
                .build(),
        }
    }

    pub fn get(&self, alias: &str, session_id: &str) -> Option<CandidateKey> {
        self.cache.get(&(alias.to_string(), session_id.to_string()))
    }

    pub fn insert(&self, alias: &str, session_id: &str, candidate: CandidateKey) {
        self.cache
            .insert((alias.to_string(), session_id.to_string()), candidate);
    }

    pub fn remove(&self, alias: &str, session_id: &str) {
        self.cache
            .remove(&(alias.to_string(), session_id.to_string()));
    }

    pub fn len(&self) -> usize {
        self.cache.entry_count() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    fn key(provider: &str, model: &str) -> CandidateKey {
        (provider.to_string(), model.to_string())
    }

    #[test]
    fn insert_and_get() {
        let store = SessionStore::new(Duration::from_secs(60), 1000);
        store.insert("fast", "sess-1", key("openai", "gpt-4o"));
        let result = store.get("fast", "sess-1");
        assert_eq!(result, Some(key("openai", "gpt-4o")));
    }

    #[test]
    fn get_returns_none_for_unknown() {
        let store = SessionStore::new(Duration::from_secs(60), 1000);
        assert_eq!(store.get("fast", "nonexistent"), None);
    }

    #[test]
    fn get_returns_none_after_ttl() {
        let store = SessionStore::new(Duration::from_millis(50), 1000);
        store.insert("fast", "sess-1", key("openai", "gpt-4o"));
        sleep(Duration::from_millis(60));
        assert_eq!(store.get("fast", "sess-1"), None);
    }

    #[test]
    fn remove_clears_binding() {
        let store = SessionStore::new(Duration::from_secs(60), 1000);
        store.insert("fast", "sess-1", key("openai", "gpt-4o"));
        store.remove("fast", "sess-1");
        assert_eq!(store.get("fast", "sess-1"), None);
    }

    #[test]
    fn touch_on_access_extends_lifetime() {
        let store = SessionStore::new(Duration::from_millis(80), 1000);
        store.insert("fast", "sess-1", key("openai", "gpt-4o"));
        sleep(Duration::from_millis(50));
        assert!(store.get("fast", "sess-1").is_some());
        sleep(Duration::from_millis(50));
        assert!(store.get("fast", "sess-1").is_some());
    }

    #[test]
    fn scoped_by_alias() {
        let store = SessionStore::new(Duration::from_secs(60), 1000);
        store.insert("fast", "sess-1", key("openai", "gpt-4o"));
        store.insert("smart", "sess-1", key("anthropic", "claude"));
        assert_eq!(store.get("fast", "sess-1"), Some(key("openai", "gpt-4o")));
        assert_eq!(
            store.get("smart", "sess-1"),
            Some(key("anthropic", "claude"))
        );
    }
}
