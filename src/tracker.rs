use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Key: (provider_name, model_name)
pub type CandidateKey = (String, String);

/// Tracks the outcome (success/failure) of recent requests in a time-based
/// moving window. Entries older than `window` are pruned on each access,
/// so errors naturally age out without an abrupt reset.
struct ErrorWindow {
    /// Recent outcomes: (timestamp, success). Oldest entries at the front.
    outcomes: VecDeque<(Instant, bool)>,
    /// How long outcomes are retained.
    window: Duration,
    /// Maximum entries to retain (prevents unbounded growth under high load).
    max_entries: usize,
}

impl ErrorWindow {
    fn new(window: Duration, max_entries: usize) -> Self {
        Self {
            outcomes: VecDeque::new(),
            window,
            max_entries,
        }
    }

    fn prune(&mut self) {
        let cutoff = Instant::now() - self.window;
        while let Some(&(ts, _)) = self.outcomes.front() {
            if ts < cutoff {
                self.outcomes.pop_front();
            } else {
                break;
            }
        }
    }

    fn record(&mut self, success: bool) {
        self.prune();
        while self.outcomes.len() >= self.max_entries {
            self.outcomes.pop_front();
        }
        self.outcomes.push_back((Instant::now(), success));
    }

    /// Error rate from 0.0 to 1.0 over entries within the window.
    /// Returns 0.0 if no requests are in the window.
    fn error_rate(&mut self) -> f64 {
        self.prune();
        let total = self.outcomes.len();
        if total == 0 {
            return 0.0;
        }
        let errors = self.outcomes.iter().filter(|(_, ok)| !ok).count();
        errors as f64 / total as f64
    }
}

pub struct CandidateStats {
    /// EWMA of time-to-first-chunk in milliseconds. u64::MAX means cold (no data).
    pub ewma_ms: AtomicU64,
    /// Number of in-flight requests.
    pub in_flight: AtomicU32,
    /// Time-based sliding window of recent request outcomes.
    error_window: Mutex<ErrorWindow>,
}

impl CandidateStats {
    pub fn new(error_window_duration: Duration, max_error_window_entries: usize) -> Self {
        Self {
            ewma_ms: AtomicU64::new(u64::MAX),
            in_flight: AtomicU32::new(0),
            error_window: Mutex::new(ErrorWindow::new(
                error_window_duration,
                max_error_window_entries,
            )),
        }
    }

    /// Lock the error window, recovering from poison by replacing with a fresh window.
    fn lock_window(&self) -> std::sync::MutexGuard<'_, ErrorWindow> {
        self.error_window.lock().unwrap_or_else(|poisoned| {
            tracing::warn!("error window mutex was poisoned, resetting");
            let window = poisoned.get_ref().window;
            let max_entries = poisoned.get_ref().max_entries;
            let mut guard = poisoned.into_inner();
            *guard = ErrorWindow::new(window, max_entries);
            self.error_window.clear_poison();
            self.error_window.lock().unwrap()
        })
    }

    pub fn is_cold(&self) -> bool {
        self.ewma_ms.load(Ordering::Relaxed) == u64::MAX
    }

    /// Error rate over the recent time window (0.0–1.0).
    pub fn error_rate(&self) -> f64 {
        self.lock_window().error_rate()
    }

    pub fn record_success(&self) {
        self.lock_window().record(true);
    }

    pub fn record_error(&self) {
        self.lock_window().record(false);
    }

    pub fn update_ewma(&self, observed_ms: u64, alpha: f64) {
        loop {
            let old = self.ewma_ms.load(Ordering::Relaxed);
            let new_val = if old == u64::MAX {
                observed_ms
            } else {
                let new_f = alpha * observed_ms as f64 + (1.0 - alpha) * old as f64;
                new_f.round() as u64
            };
            match self.ewma_ms.compare_exchange_weak(
                old,
                new_val,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue, // another thread updated; retry with fresh value
            }
        }
    }
}

pub struct Tracker {
    pub stats: HashMap<CandidateKey, Arc<CandidateStats>>,
    pub alpha: f64,
    /// Duration of the error tracking window.
    pub error_window_duration: Duration,
    /// Error rate threshold above which a candidate is considered degraded.
    pub error_threshold: f64,
    /// Maximum entries per error window.
    pub max_error_window_entries: usize,
}

impl Tracker {
    pub fn new(
        alpha: f64,
        error_decay_secs: u64,
        error_threshold: f64,
        max_error_window_entries: usize,
    ) -> Self {
        Self {
            stats: HashMap::new(),
            alpha,
            error_window_duration: Duration::from_secs(error_decay_secs),
            error_threshold,
            max_error_window_entries,
        }
    }

    pub fn register(&mut self, key: CandidateKey) {
        let duration = self.error_window_duration;
        let max_entries = self.max_error_window_entries;
        self.stats
            .entry(key)
            .or_insert_with(|| Arc::new(CandidateStats::new(duration, max_entries)));
    }

    pub fn get(&self, key: &CandidateKey) -> Option<&Arc<CandidateStats>> {
        self.stats.get(key)
    }

    pub fn record_ttfc(&self, key: &CandidateKey, ttfc: Duration) {
        if let Some(stats) = self.stats.get(key) {
            let ms = ttfc.as_millis() as u64;
            stats.update_ewma(ms, self.alpha);
        }
    }

    pub fn record_success(&self, key: &CandidateKey) {
        if let Some(stats) = self.stats.get(key) {
            stats.record_success();
        }
    }

    pub fn record_error(&self, key: &CandidateKey) {
        if let Some(stats) = self.stats.get(key) {
            stats.record_error();
        }
    }

    /// Whether this candidate is degraded (error rate above threshold
    /// within the recent time window).
    pub fn is_degraded(&self, key: &CandidateKey) -> bool {
        if let Some(stats) = self.stats.get(key) {
            return stats.error_rate() > self.error_threshold;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(provider: &str, model: &str) -> CandidateKey {
        (provider.to_string(), model.to_string())
    }

    #[test]
    fn ewma_first_observation_sets_directly() {
        let stats = CandidateStats::new(Duration::from_secs(30), 10_000);
        assert!(stats.is_cold());
        stats.update_ewma(100, 0.3);
        assert_eq!(stats.ewma_ms.load(Ordering::Relaxed), 100);
        assert!(!stats.is_cold());
    }

    #[test]
    fn ewma_blends_subsequent_observations() {
        let stats = CandidateStats::new(Duration::from_secs(30), 10_000);
        stats.update_ewma(100, 0.3);
        stats.update_ewma(200, 0.3);
        assert_eq!(stats.ewma_ms.load(Ordering::Relaxed), 130);
    }

    #[test]
    fn ewma_converges_toward_constant_signal() {
        let stats = CandidateStats::new(Duration::from_secs(30), 10_000);
        for _ in 0..50 {
            stats.update_ewma(500, 0.3);
        }
        assert_eq!(stats.ewma_ms.load(Ordering::Relaxed), 500);
    }

    #[test]
    fn error_rate_empty_is_zero() {
        let stats = CandidateStats::new(Duration::from_secs(30), 10_000);
        assert_eq!(stats.error_rate(), 0.0);
    }

    #[test]
    fn error_rate_tracks_recent_outcomes() {
        let stats = CandidateStats::new(Duration::from_secs(30), 10_000);
        stats.record_success();
        stats.record_success();
        stats.record_error();
        stats.record_success();
        stats.record_error();
        assert!((stats.error_rate() - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn errors_age_out_of_window() {
        let stats = CandidateStats::new(Duration::from_secs(1), 10_000);
        for _ in 0..10 {
            stats.record_error();
        }
        assert_eq!(stats.error_rate(), 1.0);

        std::thread::sleep(Duration::from_millis(1100));
        assert_eq!(stats.error_rate(), 0.0);
    }

    #[test]
    fn old_errors_retained_alongside_new_ones() {
        let stats = CandidateStats::new(Duration::from_secs(2), 10_000);
        for _ in 0..5 {
            stats.record_error();
        }

        std::thread::sleep(Duration::from_millis(200));
        for _ in 0..5 {
            stats.record_success();
        }

        assert!((stats.error_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn tracker_not_degraded_below_threshold() {
        let mut tracker = Tracker::new(0.3, 30, 0.5, 10_000);
        let k = key("openai", "gpt-4o-mini");
        tracker.register(k.clone());
        for _ in 0..6 {
            tracker.record_success(&k);
        }
        for _ in 0..4 {
            tracker.record_error(&k);
        }
        assert!(!tracker.is_degraded(&k));
    }

    #[test]
    fn tracker_degraded_above_threshold() {
        let mut tracker = Tracker::new(0.3, 30, 0.5, 10_000);
        let k = key("openai", "gpt-4o-mini");
        tracker.register(k.clone());
        for _ in 0..4 {
            tracker.record_success(&k);
        }
        for _ in 0..6 {
            tracker.record_error(&k);
        }
        assert!(tracker.is_degraded(&k));
    }

    #[test]
    fn tracker_degraded_recovers_when_errors_age_out() {
        let mut tracker = Tracker::new(0.3, 1, 0.5, 10_000);
        let k = key("openai", "gpt-4o-mini");
        tracker.register(k.clone());
        for _ in 0..10 {
            tracker.record_error(&k);
        }
        assert!(tracker.is_degraded(&k));

        std::thread::sleep(Duration::from_millis(1100));
        assert!(!tracker.is_degraded(&k));
    }

    #[test]
    fn tracker_degraded_recovers_with_successes() {
        let mut tracker = Tracker::new(0.3, 30, 0.5, 10_000);
        let k = key("openai", "gpt-4o-mini");
        tracker.register(k.clone());
        for _ in 0..10 {
            tracker.record_error(&k);
        }
        assert!(tracker.is_degraded(&k));
        for _ in 0..11 {
            tracker.record_success(&k);
        }
        assert!(!tracker.is_degraded(&k));
    }

    #[test]
    fn record_ttfc_updates_ewma() {
        let mut tracker = Tracker::new(0.3, 30, 0.5, 10_000);
        let k = key("groq", "llama");
        tracker.register(k.clone());
        tracker.record_ttfc(&k, Duration::from_millis(150));
        let ewma = tracker.get(&k).unwrap().ewma_ms.load(Ordering::Relaxed);
        assert_eq!(ewma, 150);
    }
}
