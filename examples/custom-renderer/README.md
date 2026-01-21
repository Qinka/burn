# Custom Renderer Example - TensorBoard Integration

This example demonstrates how to use a custom renderer for Burn training, specifically showcasing the TensorBoard renderer from the `burn-report` crate.

## Overview

The example trains a simple MNIST classifier using the Burn framework and logs training metrics to TensorBoard for visualization.

## Features

- **TensorBoard Integration**: Automatically logs training and validation metrics
- **Real-time Visualization**: View metrics as training progresses
- **Easy Setup**: Default configuration writes logs to `./runs/{timestamp}/`

## Running the Example

### 1. Run the training

```bash
cargo run --example custom-renderer --release
```

### 2. View the results with TensorBoard

While training is running (or after it completes), start TensorBoard:

```bash
tensorboard --logdir runs
```

Then open your browser to [http://localhost:6006](http://localhost:6006)

## What Gets Logged

The TensorBoard renderer automatically logs:
- Training loss
- Validation loss
- Learning rate
- Any other numeric metrics registered during training

All metrics are prefixed with:
- `train/` for training metrics
- `valid/` for validation metrics

## Customization

You can customize the log directory:

```rust
use burn_report::TensorboardRenderer;
use std::path::PathBuf;

// Use a custom log directory
let renderer = TensorboardRenderer::with_logdir(
    PathBuf::from("./my_custom_logs")
).expect("Failed to create renderer");
```

## Directory Structure

After running the example, you'll have:

```
runs/
└── {timestamp}/
    └── events.out.tfevents.{timestamp}.{pid}
```

## Requirements

- TensorBoard must be installed separately to view the logs
- Install with: `pip install tensorboard`

## Learn More

- [Burn Documentation](https://burn.dev)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [burn-report crate README](../../crates/burn-report/README.md)
