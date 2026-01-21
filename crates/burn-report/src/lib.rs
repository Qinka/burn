//! # Burn Report
//!
//! Reporting and visualization tools for Burn training, including TensorBoard support.
//!
//! This crate provides renderers for visualizing training metrics in real-time using
//! popular tools like TensorBoard.
//!
//! ## TensorBoard Renderer
//!
//! The `TensorboardRenderer` writes training metrics to TensorBoard event files,
//! which can be visualized using the TensorBoard web interface.
//!
//! ### Usage
//!
//! ```rust,no_run
//! use burn_report::{TensorboardRenderer, TensorboardRendererConfig};
//! use std::path::PathBuf;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a renderer with default log directory (./runs/{timestamp})
//! let renderer = TensorboardRenderer::with_default_logdir()?;
//!
//! // Or specify a custom log directory
//! let renderer = TensorboardRenderer::with_logdir(PathBuf::from("./my_logs"))?;
//!
//! // Or use a configuration
//! let config = TensorboardRendererConfig::new(PathBuf::from("./logs"));
//! let renderer = TensorboardRenderer::new(config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Viewing Results
//!
//! After training, you can view the results using TensorBoard:
//!
//! ```bash
//! tensorboard --logdir runs
//! ```
//!
//! Then open http://localhost:6006 in your browser.
//!
//! ## Limitations
//!
//! - Currently only supports scalar metrics (no histograms or images)
//! - Single-process training only (DDP aggregation not supported)
//! - CRC checksums in event files are placeholders (set to 0)

mod event;
mod renderer;
mod writer;

pub use renderer::{TensorboardRenderer, TensorboardRendererConfig};
pub use writer::{EventWriter, WriterError};
