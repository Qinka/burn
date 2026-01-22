use crate::renderer::{TensorboardRenderer, TensorboardRendererConfig};
use crate::writer::WriterError;

#[test]
fn test_tensorboard_renderer_creation() {
    let temp_dir = tempfile::tempdir().unwrap();
    let logdir = temp_dir.path().to_path_buf();

    let config = TensorboardRendererConfig::new(logdir.clone());
    let renderer = TensorboardRenderer::new(config);

    assert!(renderer.is_ok());

    // Check that the logdir was created
    assert!(logdir.exists());
}

#[test]
fn test_default_logdir_config() {
    let config = TensorboardRendererConfig::default_logdir();
    assert!(config.logdir.to_string_lossy().contains("runs"));
}

#[test]
fn test_writer_error_display() {
    let error = WriterError::SystemTime;
    let error_str = format!("{}", error);
    assert!(error_str.contains("system time"));
}
