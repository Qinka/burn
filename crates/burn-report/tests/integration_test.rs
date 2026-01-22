use burn_report::EventWriter;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn test_event_writer_creates_file() {
    let temp_dir = tempdir().unwrap();
    let logdir = temp_dir.path().to_path_buf();

    let mut writer = EventWriter::new(logdir.clone()).unwrap();

    // Write some scalar events
    writer.add_scalar("loss", 0.5, 0).unwrap();
    writer.add_scalar("accuracy", 0.9, 0).unwrap();
    writer.add_scalar("loss", 0.3, 1).unwrap();
    writer.add_scalar("accuracy", 0.95, 1).unwrap();

    writer.flush().unwrap();

    // Verify that event files were created
    let entries: Vec<_> = fs::read_dir(&logdir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();

    assert!(!entries.is_empty(), "No event files created");

    // Verify filename pattern
    let has_event_file = entries.iter().any(|e| {
        e.file_name()
            .to_string_lossy()
            .starts_with("events.out.tfevents.")
    });
    assert!(has_event_file, "No TensorBoard event file found");

    // Verify file has content
    for entry in entries {
        let metadata = entry.metadata().unwrap();
        assert!(metadata.len() > 0, "Event file is empty");
    }
}

#[test]
fn test_event_writer_with_custom_logdir() {
    let temp_dir = tempdir().unwrap();
    let logdir = temp_dir.path().join("custom").join("logs");

    let mut writer = EventWriter::new(logdir.clone()).unwrap();
    writer.add_scalar("test_metric", 1.0, 0).unwrap();
    writer.flush().unwrap();

    // Verify nested directory was created
    assert!(logdir.exists());
    assert!(logdir.is_dir());
}

#[test]
fn test_event_writer_error_handling() {
    // Test with invalid path (on Unix, null bytes are invalid)
    #[cfg(unix)]
    {
        let invalid_path = PathBuf::from("/tmp/\0invalid");
        let result = EventWriter::new(invalid_path);
        assert!(result.is_err());
    }
}
