use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::AdamConfig,
    tensor::backend::AutodiffBackend,
    train::{Learner, SupervisedTraining},
};
use burn_report::TensorboardRenderer;
use guide::{data::MnistBatcher, model::ModelConfig};

#[derive(Config, Debug)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // Create the configuration.
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);

    B::seed(&device, config.seed);

    // Create the model and optimizer.
    let model = config.model.init::<B>(&device);
    let optim = config.optimizer.init();

    // Create the batcher.
    let batcher = MnistBatcher::default();

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    // Create TensorBoard renderer
    // Logs will be written to ./runs/{timestamp}/
    // You can view them with: tensorboard --logdir runs
    let renderer = TensorboardRenderer::with_default_logdir()
        .expect("Failed to create TensorBoard renderer");

    println!("Training with TensorBoard logging enabled.");
    println!("To view the logs, run: tensorboard --logdir runs");
    println!("Then open http://localhost:6006 in your browser.");

    // artifact dir does not need to be provided when log_to_file is false
    let training = SupervisedTraining::new("", dataloader_train, dataloader_test)
        .num_epochs(config.num_epochs)
        .renderer(renderer)
        .with_application_logger(None);
    // can be used to interrupt training
    let _interrupter = training.interrupter();

    let _model_trained = training.launch(Learner::new(model, optim, config.lr));
}
