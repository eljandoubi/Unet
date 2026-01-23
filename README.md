# UNet for Semantic Segmentation

This repository contains a PyTorch implementation of a UNet model for semantic segmentation, trained on the ADE20K dataset.

## Project Structure

```
src/
    dataloader.py  # Dataset class for ADE20K
    model.py       # UNet model architecture definition
    train.py       # Main training script
```

## Dependencies

The project uses the following major libraries:
- `torch`
- `torchvision`
- `transformers`
- `accelerate`
- `kagglehub`
- `python-dotenv`

See `pyproject.toml` for the complete list and version requirements.

## Installation

1. Clone the repository.
2. Install the dependencies:
   ```bash
   uv sync
   ```

## Usage

### Training

To start training the model, run the `train.py` script from the `src` directory:

```bash
cd src
python train.py
```

The script will:
1. Automatically download the ADE20K dataset using `kagglehub`.
2. Initialize the UNet model (modification of hyperparameters is currently done inside `train.py`).
3. Train the model and save checkpoints/logs to a directory named after the `experiment_name` (e.g., `UNET_wo_skip_ADE20K_test`).

### Configuration

Training parameters such as `batch_size`, `learning_rate`, `num_epochs`, and `experiment_name` are currently hardcoded in the `if __name__ == "__main__":` block at the bottom of `src/train.py`. Modify these values directly in the file to change the training configuration.

## Features

- **Model**: Custom UNet implementation with optional skip connections.
- **Data**: Automatic download and processing of the ADE20K dataset.
- **Training**: Supports gradient accumulation and mixed precision training via Hugging Face `accelerate`.
- **Logging**: Custom `LocalLogger` to track training and testing metrics (loss, accuracy).

