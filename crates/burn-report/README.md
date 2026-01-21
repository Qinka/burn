# Burn Report

Reporting and visualization tools for Burn training, including TensorBoard support.

## Features

- **TensorBoard Renderer**: Write training metrics to TensorBoard event files for visualization
- **Thread-safe**: Safe to use in multi-threaded training scenarios
- **Automatic log directory management**: Creates timestamped log directories by default

## Usage

### Basic Example

```rust
use burn_report::{TensorboardRenderer, TensorboardRendererConfig};
use burn::train::SupervisedTraining;
use std::path::PathBuf;

// Create a renderer with default log directory (./runs/{timestamp})
let renderer = TensorboardRenderer::with_default_logdir()
    .expect("Failed to create TensorBoard renderer");

// Use it in your training
let training = SupervisedTraining::new("", dataloader_train, dataloader_test)
    .num_epochs(10)
    .renderer(renderer);
```

### Custom Log Directory

```rust
use burn_report::TensorboardRenderer;
use std::path::PathBuf;

// Specify a custom log directory
let renderer = TensorboardRenderer::with_logdir(PathBuf::from("./my_training_logs"))
    .expect("Failed to create TensorBoard renderer");
```

### Viewing Results

After training, view the results using TensorBoard:

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

## Supported Metrics

Currently, the TensorBoard renderer supports:
- **Scalar metrics**: Loss, accuracy, learning rate, etc.

## Limitations

- Histogram and image support not yet implemented
- Single-process training only (DDP aggregation not supported)
- CRC checksums in event files are placeholders (TensorBoard still reads them correctly)

## Log Directory Structure

By default, logs are written to:
```
./runs/{timestamp}/events.out.tfevents.{timestamp}.{pid}
```

You can customize the base directory using `TensorboardRendererConfig`.

## Dependencies

- `burn-core`: Core Burn framework
- `burn-train`: Training utilities
- `parking_lot`: Thread-safe synchronization
- `thiserror`: Error handling
