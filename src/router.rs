use crate::model_map::ResolvedCandidate;
use crate::tracker::{CandidateKey, Tracker};
use std::sync::atomic::Ordering;

/// Round-robin counters per alias. Two separate counters: one for the
/// explore-vs-exploit decision, one for distributing across explore candidates.
/// Keeping them separate avoids counter drift when explore picks consume ticks.
pub struct RoundRobinState {
    /// Counter for explore-vs-exploit decision.
    route_counters: std::collections::HashMap<String, std::sync::atomic::AtomicUsize>,
    /// Counter for distributing across explore candidates.
    explore_counters: std::collections::HashMap<String, std::sync::atomic::AtomicUsize>,
}

impl Default for RoundRobinState {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRobinState {
    pub fn new() -> Self {
        Self {
            route_counters: std::collections::HashMap::new(),
            explore_counters: std::collections::HashMap::new(),
        }
    }

    pub fn register_alias(&mut self, alias: String) {
        self.route_counters
            .entry(alias.clone())
            .or_insert_with(|| std::sync::atomic::AtomicUsize::new(0));
        self.explore_counters
            .entry(alias)
            .or_insert_with(|| std::sync::atomic::AtomicUsize::new(0));
    }

    /// Advance the main routing counter. Used for explore-vs-exploit decision
    /// and for all-cold round-robin.
    pub fn next(&self, alias: &str, count: usize) -> usize {
        if let Some(counter) = self.route_counters.get(alias) {
            counter.fetch_add(1, Ordering::Relaxed) % count
        } else {
            0
        }
    }

    /// Advance the explore-specific counter. Used to distribute across
    /// explore candidates without interfering with the routing counter.
    pub fn next_explore(&self, alias: &str, count: usize) -> usize {
        if let Some(counter) = self.explore_counters.get(alias) {
            counter.fetch_add(1, Ordering::Relaxed) % count
        } else {
            0
        }
    }
}

/// Pick the best candidate for the given alias.
///
/// Candidates are split into three buckets:
///   - **warm**: have EWMA data and error rate below threshold → lowest-latency selection
///   - **cold**: no EWMA data yet → discovered via exploration
///   - **degraded**: high error rate → excluded from selection entirely
///
/// The `explore_ratio` controls what fraction of requests round-robin across all
/// healthy candidates (warm + cold). This both discovers cold candidates and
/// periodically re-samples warm-but-not-fastest ones to detect latency changes.
///
/// Degraded candidates recover via the timer-based decay in the tracker: after
/// `error_decay_secs` their error window resets and they rejoin automatically.
pub fn select_candidate<'a>(
    alias: &str,
    candidates: &'a [ResolvedCandidate],
    tracker: &Tracker,
    rr: &RoundRobinState,
    explore_ratio: f64,
) -> Option<&'a ResolvedCandidate> {
    if candidates.is_empty() {
        return None;
    }

    let mut warm = Vec::new();
    let mut cold = Vec::new();
    let mut degraded = Vec::new();

    for (i, c) in candidates.iter().enumerate() {
        let item = (i, c);
        let key = candidate_key(c);
        let stats = match tracker.get(&key) {
            Some(s) => s,
            None => {
                cold.push(item);
                continue;
            }
        };
        if tracker.is_degraded(&key) {
            degraded.push(item);
        } else if stats.is_cold() {
            cold.push(item);
        } else {
            warm.push(item);
        }
    }

    if warm.is_empty() {
        let all_explore: Vec<_> = cold.into_iter().chain(degraded).collect();
        if all_explore.is_empty() {
            return None;
        }
        let idx = rr.next(alias, all_explore.len());
        return Some(all_explore[idx].1);
    }

    // Send a proportion of requests to round-robin across all healthy candidates
    // (warm + cold). This ensures cold candidates get discovered AND warm-but-not-
    // fastest candidates get periodic probes to detect latency improvements.
    let healthy: Vec<_> = warm.iter().chain(cold.iter()).copied().collect();
    if explore_ratio > 0.0 && healthy.len() > 1 {
        let denom = (1.0 / explore_ratio).round().max(2.0) as usize;
        if rr.next(alias, denom) == 0 {
            let idx = rr.next_explore(alias, healthy.len());
            return Some(healthy[idx].1);
        }
    }

    warm.iter()
        .min_by(|(_, a), (_, b)| {
            let ka = candidate_key(a);
            let kb = candidate_key(b);
            let sa = tracker.get(&ka).unwrap();
            let sb = tracker.get(&kb).unwrap();
            let ewma_a = sa.ewma_ms.load(Ordering::Relaxed);
            let ewma_b = sb.ewma_ms.load(Ordering::Relaxed);

            // If within 10ms, tie-break on in-flight
            if ewma_a.abs_diff(ewma_b) <= 10 {
                let inf_a = sa.in_flight.load(Ordering::Relaxed);
                let inf_b = sb.in_flight.load(Ordering::Relaxed);
                inf_a.cmp(&inf_b)
            } else {
                ewma_a.cmp(&ewma_b)
            }
        })
        .map(|(_, c)| *c)
}

pub fn candidate_key(c: &ResolvedCandidate) -> CandidateKey {
    (c.provider_name.clone(), c.model.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::Tracker;
    use std::time::Duration;

    fn make_candidate(provider: &str, model: &str) -> ResolvedCandidate {
        ResolvedCandidate {
            provider_name: provider.to_string(),
            model: model.to_string(),
            base_url: "http://localhost".to_string(),
            api_key: None,
            kind: crate::model_map::ProviderKind::ApiKey,
        }
    }

    fn setup(candidates: &[ResolvedCandidate]) -> (Tracker, RoundRobinState) {
        let mut tracker = Tracker::new(0.3, 30, 0.5, 10_000);
        let mut rr = RoundRobinState::new();
        rr.register_alias("test".to_string());
        for c in candidates {
            tracker.register(candidate_key(c));
        }
        (tracker, rr)
    }

    #[test]
    fn empty_candidates_returns_none() {
        let candidates: Vec<ResolvedCandidate> = vec![];
        let (tracker, rr) = setup(&candidates);
        assert!(select_candidate("test", &candidates, &tracker, &rr, 0.2).is_none());
    }

    #[test]
    fn all_cold_round_robins() {
        let candidates = vec![make_candidate("a", "m1"), make_candidate("b", "m2")];
        let (tracker, rr) = setup(&candidates);

        let first = select_candidate("test", &candidates, &tracker, &rr, 0.2).unwrap();
        let second = select_candidate("test", &candidates, &tracker, &rr, 0.2).unwrap();
        assert_ne!(first.provider_name, second.provider_name);
    }

    #[test]
    fn picks_lowest_ewma() {
        let candidates = vec![make_candidate("slow", "m1"), make_candidate("fast", "m2")];
        let (tracker, rr) = setup(&candidates);

        let slow_key = ("slow".to_string(), "m1".to_string());
        let fast_key = ("fast".to_string(), "m2".to_string());
        tracker.record_ttfc(&slow_key, Duration::from_millis(500));
        tracker.record_ttfc(&fast_key, Duration::from_millis(100));

        for _ in 0..10 {
            let picked = select_candidate("test", &candidates, &tracker, &rr, 0.0).unwrap();
            assert_eq!(picked.provider_name, "fast");
        }
    }

    #[test]
    fn tiebreak_on_in_flight() {
        let candidates = vec![make_candidate("a", "m1"), make_candidate("b", "m2")];
        let (tracker, rr) = setup(&candidates);

        let ka = ("a".to_string(), "m1".to_string());
        let kb = ("b".to_string(), "m2".to_string());
        tracker.record_ttfc(&ka, Duration::from_millis(100));
        tracker.record_ttfc(&kb, Duration::from_millis(105));

        tracker
            .get(&ka)
            .unwrap()
            .in_flight
            .store(5, Ordering::Relaxed);
        tracker
            .get(&kb)
            .unwrap()
            .in_flight
            .store(1, Ordering::Relaxed);

        let picked = select_candidate("test", &candidates, &tracker, &rr, 0.0).unwrap();
        assert_eq!(picked.provider_name, "b");
    }

    #[test]
    fn degraded_candidate_goes_to_explore() {
        let candidates = vec![make_candidate("good", "m1"), make_candidate("bad", "m2")];
        let (tracker, rr) = setup(&candidates);

        let good_key = ("good".to_string(), "m1".to_string());
        let bad_key = ("bad".to_string(), "m2".to_string());

        tracker.record_ttfc(&good_key, Duration::from_millis(200));
        tracker.record_ttfc(&bad_key, Duration::from_millis(50));

        for _ in 0..10 {
            tracker.record_error(&bad_key);
        }
        assert!(tracker.is_degraded(&bad_key));

        for _ in 0..20 {
            let picked = select_candidate("test", &candidates, &tracker, &rr, 0.2).unwrap();
            assert_eq!(picked.provider_name, "good");
        }
    }

    #[test]
    fn explore_ratio_sends_traffic_to_cold() {
        let candidates = vec![make_candidate("warm", "m1"), make_candidate("cold", "m2")];
        let (tracker, rr) = setup(&candidates);

        let warm_key = ("warm".to_string(), "m1".to_string());
        tracker.record_ttfc(&warm_key, Duration::from_millis(100));

        // With explore_ratio=0.5, roughly half should go to the cold candidate
        let mut cold_picks = 0;
        let n = 100;
        for _ in 0..n {
            let picked = select_candidate("test", &candidates, &tracker, &rr, 0.5).unwrap();
            if picked.provider_name == "cold" {
                cold_picks += 1;
            }
        }
        assert!(
            cold_picks > 10,
            "expected some cold picks, got {cold_picks}"
        );
        assert!(
            cold_picks < 90,
            "expected some warm picks, got {cold_picks} cold"
        );
    }

    #[test]
    fn traffic_shifts_when_latency_increases() {
        let candidates = vec![make_candidate("a", "m1"), make_candidate("b", "m2")];
        let (tracker, rr) = setup(&candidates);

        let ka = ("a".to_string(), "m1".to_string());
        let kb = ("b".to_string(), "m2".to_string());

        tracker.record_ttfc(&ka, Duration::from_millis(100));
        tracker.record_ttfc(&kb, Duration::from_millis(300));

        let picked = select_candidate("test", &candidates, &tracker, &rr, 0.0).unwrap();
        assert_eq!(picked.provider_name, "a");

        for _ in 0..20 {
            tracker.record_ttfc(&ka, Duration::from_millis(500));
        }

        let picked = select_candidate("test", &candidates, &tracker, &rr, 0.0).unwrap();
        assert_eq!(picked.provider_name, "b");
    }

    #[test]
    fn cold_candidate_with_errors_is_degraded_not_explored() {
        let candidates = vec![make_candidate("good", "m1"), make_candidate("bad", "m2")];
        let (tracker, rr) = setup(&candidates);

        let good_key = ("good".to_string(), "m1".to_string());
        let bad_key = ("bad".to_string(), "m2".to_string());

        tracker.record_ttfc(&good_key, Duration::from_millis(100));

        for _ in 0..10 {
            tracker.record_error(&bad_key);
        }

        // Cold (no EWMA) but degraded takes priority
        assert!(tracker.get(&bad_key).unwrap().is_cold());
        assert!(tracker.is_degraded(&bad_key));

        for _ in 0..20 {
            let picked = select_candidate("test", &candidates, &tracker, &rr, 0.5).unwrap();
            assert_eq!(picked.provider_name, "good");
        }
    }
}
