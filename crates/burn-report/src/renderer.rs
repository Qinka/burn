// TensorBoard renderer for Burn training

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use burn_train::metric::{MetricDefinition, MetricId};
use burn_train::renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, TrainingProgress,
};
use parking_lot::Mutex;

use crate::writer::{EventWriter, WriterError};

/// Configuration for the TensorBoard renderer
#[derive(Clone, Debug)]
pub struct TensorboardRendererConfig {
    /// Directory where TensorBoard logs will be written
    pub logdir: PathBuf,
}

impl TensorboardRendererConfig {
    /// Create a new configuration with a custom log directory
    pub fn new(logdir: PathBuf) -> Self {
        Self { logdir }
    }

    /// Create a new configuration with default log directory (./runs/{timestamp})
    pub fn default_logdir() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let logdir = PathBuf::from(format!("./runs/{}", timestamp));
        Self { logdir }
    }
}

impl Default for TensorboardRendererConfig {
    fn default() -> Self {
        Self::default_logdir()
    }
}

/// TensorBoard renderer for training metrics
pub struct TensorboardRenderer {
    writer: Arc<Mutex<EventWriter>>,
    metric_definitions: HashMap<MetricId, MetricDefinition>,
    train_metrics: HashMap<MetricId, f32>,
    valid_metrics: HashMap<MetricId, f32>,
}

impl TensorboardRenderer {
    /// Create a new TensorBoard renderer with the given configuration
    pub fn new(config: TensorboardRendererConfig) -> Result<Self, WriterError> {
        let writer = EventWriter::new(config.logdir)?;

        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
            metric_definitions: HashMap::new(),
            train_metrics: HashMap::new(),
            valid_metrics: HashMap::new(),
        })
    }

    /// Create a new TensorBoard renderer with a custom log directory
    pub fn with_logdir(logdir: PathBuf) -> Result<Self, WriterError> {
        Self::new(TensorboardRendererConfig::new(logdir))
    }

    /// Create a new TensorBoard renderer with default log directory
    pub fn with_default_logdir() -> Result<Self, WriterError> {
        Self::new(TensorboardRendererConfig::default())
    }

    /// Extract scalar value from MetricState
    fn extract_scalar(&self, state: &MetricState) -> Option<(MetricId, f32)> {
        match state {
            MetricState::Numeric(entry, numeric) => {
                let metric_id = entry.metric_id.clone();
                let value = numeric.current() as f32;
                Some((metric_id, value))
            }
            MetricState::Generic(_) => None,
        }
    }

    /// Get the metric name from its ID
    fn get_metric_name(&self, metric_id: &MetricId) -> String {
        self.metric_definitions
            .get(metric_id)
            .map(|def| def.name.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }
}

impl MetricsRendererTraining for TensorboardRenderer {
    fn update_train(&mut self, state: MetricState) {
        if let Some((metric_id, value)) = self.extract_scalar(&state) {
            self.train_metrics.insert(metric_id, value);
        }
    }

    fn update_valid(&mut self, state: MetricState) {
        if let Some((metric_id, value)) = self.extract_scalar(&state) {
            self.valid_metrics.insert(metric_id, value);
        }
    }

    fn render_train(&mut self, item: TrainingProgress) {
        let step = item.iteration as i64;
        let mut writer = self.writer.lock();

        for (metric_id, value) in &self.train_metrics {
            let name = self.get_metric_name(metric_id);
            let tag = format!("train/{}", name);
            if let Err(e) = writer.add_scalar(&tag, *value, step) {
                eprintln!("Failed to write train metric {}: {}", name, e);
            }
        }

        if let Err(e) = writer.flush() {
            eprintln!("Failed to flush writer: {}", e);
        }

        self.train_metrics.clear();
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        let step = item.iteration as i64;
        let mut writer = self.writer.lock();

        for (metric_id, value) in &self.valid_metrics {
            let name = self.get_metric_name(metric_id);
            let tag = format!("valid/{}", name);
            if let Err(e) = writer.add_scalar(&tag, *value, step) {
                eprintln!("Failed to write valid metric {}: {}", name, e);
            }
        }

        if let Err(e) = writer.flush() {
            eprintln!("Failed to flush writer: {}", e);
        }

        self.valid_metrics.clear();
    }
}

impl MetricsRendererEvaluation for TensorboardRenderer {
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {
        // Store test metrics - for now we'll just log them on render
    }

    fn render_test(&mut self, _item: EvaluationProgress) {
        let mut writer = self.writer.lock();

        // For test rendering, we just flush the writer
        if let Err(e) = writer.flush() {
            eprintln!("Failed to flush writer during test: {}", e);
        }
    }
}

impl MetricsRenderer for TensorboardRenderer {
    fn manual_close(&mut self) {
        let mut writer = self.writer.lock();
        if let Err(e) = writer.flush() {
            eprintln!("Failed to flush writer on close: {}", e);
        }
    }

    fn register_metric(&mut self, definition: MetricDefinition) {
        self.metric_definitions
            .insert(definition.metric_id.clone(), definition);
    }
}
