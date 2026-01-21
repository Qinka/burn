// TensorBoard event file writer

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::event::write_scalar_event;

/// Errors that can occur when writing TensorBoard events
#[derive(Debug, thiserror::Error)]
pub enum WriterError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to get system time")]
    SystemTime,
}

/// TensorBoard event file writer
pub struct EventWriter {
    writer: BufWriter<File>,
}

impl EventWriter {
    /// Create a new event writer with the specified log directory
    pub fn new(logdir: PathBuf) -> Result<Self, WriterError> {
        // Create the log directory if it doesn't exist
        fs::create_dir_all(&logdir)?;

        // Create event file with timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| WriterError::SystemTime)?
            .as_secs();

        let filename = format!("events.out.tfevents.{}.{}", timestamp, std::process::id());
        let filepath = logdir.join(filename);

        let file = File::create(filepath)?;
        let writer = BufWriter::new(file);

        Ok(Self { writer })
    }

    /// Write a scalar summary
    pub fn add_scalar(&mut self, tag: &str, value: f32, step: i64) -> Result<(), WriterError> {
        let wall_time = self.get_wall_time()?;
        write_scalar_event(&mut self.writer, wall_time, step, tag, value)?;
        Ok(())
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<(), WriterError> {
        self.writer.flush()?;
        Ok(())
    }

    /// Get the current wall time in seconds since epoch
    fn get_wall_time(&self) -> Result<f64, WriterError> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .map_err(|_| WriterError::SystemTime)
    }
}

impl Drop for EventWriter {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
