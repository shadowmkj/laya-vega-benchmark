#[derive(Debug, Clone, Default)]
pub struct Telemetry {
    pub samples: Vec<f32>,
    pub deadline_misses: u32,
    pub last_latency_ms: f32,
    pub last_confidence: f32,
    pub last_action: String,
}

impl Telemetry {
    pub const DEADLINE_MS: f32 = 45.0;

    pub fn new() -> Self {
        Self {
            samples: Vec::with_capacity(1000),
            deadline_misses: 0,
            last_latency_ms: 0.0,
            last_confidence: 0.0,
            last_action: "RUN".to_string(),
        }
    }

    pub fn record(&mut self, latency_ms: f32) {
        self.last_latency_ms = latency_ms;
        self.samples.push(latency_ms);
        if latency_ms > Self::DEADLINE_MS {
            self.deadline_misses += 1;
        }
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn p50(&self) -> f32 {
        self.percentile(0.50)
    }

    pub fn p90(&self) -> f32 {
        self.percentile(0.90)
    }

    pub fn p99(&self) -> f32 {
        self.percentile(0.99)
    }

    pub fn percentile(&self, p: f32) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((sorted.len() as f32 * p).ceil() as usize).saturating_sub(1);
        sorted[idx.min(sorted.len() - 1)]
    }

    pub fn jitter(&self) -> f32 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.samples.iter().sum::<f32>() / self.samples.len() as f32;
        let variance = self.samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>()
            / self.samples.len() as f32;
        variance.sqrt()
    }

    pub fn recent_sparkline_data(&self, max_points: usize) -> Vec<u64> {
        let n = self.samples.len();
        let start = n.saturating_sub(max_points);
        self.samples[start..]
            .iter()
            .map(|&s| s.round() as u64)
            .collect()
    }
}
