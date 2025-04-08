# DeepFake Generation

This repository contains tools and models for deepfake generation, using advanced deep learning techniques to create realistic face swaps and manipulations in images and videos.

## Overview

DeepFake Generation is a comprehensive toolkit that implements state-of-the-art deepfake generation algorithms. It includes pipelines for face detection, alignment, swapping, and post-processing to create high-quality synthetic media.

## Features

- **Face Detection & Alignment**: Robust detection and alignment of facial landmarks for preprocessing
- **Multiple Model Architectures**: Implementations of various deepfake generation models, including:
  - GAN-based face swapping
  - Autoencoder architectures
- **Training Pipeline**: Complete workflow for training custom models on your own datasets
- **Inference Tools**: Optimized code for generating deepfakes from pretrained models
- **Quality Enhancement**: Post-processing techniques to improve realism and consistency

## Installation

```bash
# Clone the repository
git clone https://github.com/hcgjhb/deepfake_generation.git
cd deepfake_generation

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for GPU acceleration (optional)
pip install -r requirements-gpu.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended for HD video processing)

## Usage

### Dataset Preparation

```python
from deepfake_generation.preprocessing import prepare_dataset

# Prepare dataset from a folder of images
prepare_dataset(
    source_dir="path/to/source_faces",
    target_dir="path/to/target_faces",
    output_dir="path/to/processed_dataset",
    image_size=256,
    detect_faces=True,
    align_faces=True
)
```

### Training a Model

```python
from deepfake_generation.training import Trainer
from deepfake_generation.models import AutoencoderModel

# Initialize model
model = AutoencoderModel(
    input_shape=(3, 256, 256),
    encoder_filters=[64, 128, 256, 512],
    decoder_filters=[512, 256, 128, 64],
    latent_dim=512
)

# Initialize trainer
trainer = Trainer(
    model=model,
    dataset_path="path/to/processed_dataset",
    batch_size=32,
    learning_rate=0.0001,
    epochs=100,
    save_dir="path/to/save_models"
)

# Start training
trainer.train()
```

## Examples

### Basic Face Swap

```python
from deepfake_generation.inference import quick_swap

# Quick single image face swap
quick_swap(
    source_image="examples/images/source.jpg",
    target_image="examples/images/target.jpg",
    output_path="examples/results/swapped.jpg",
    model="autoencoder",  # Options: "autoencoder", "gan", "diffusion"
    quality="high"  # Options: "fast", "balanced", "high"
)
```

### Video Face Swap with Custom Settings

```python
from deepfake_generation.inference import VideoProcessor
from deepfake_generation.models import load_model

# Load pretrained model
model = load_model("models/pretrained/hd_quality_v2.pth")

# Initialize video processor
processor = VideoProcessor(
    model=model,
    batch_size=4,
    temporal_smoothing=True,
    use_face_tracking=True
)

# Process video
processor.process(
    source_image="examples/images/source.jpg",
    target_video="examples/videos/target.mp4",
    output_video="examples/results/swapped_video.mp4",
    frame_skip=1,  # Process every frame
    face_detection_frequency=15,  # Re-detect faces every 15 frames
    blend_ratio=0.85
)
```

## Ethical Considerations

This software is provided for research and educational purposes only. Users are responsible for complying with applicable laws and regulations when using this software. The authors do not endorse using this technology to create misleading or deceptive content.

Please consider the following guidelines:
- Always disclose synthetic media as such
- Do not create deepfakes of individuals without their consent
- Do not use this technology to create defamatory, obscene, or harassing content
- Be aware of legal restrictions regarding synthetic media in your jurisdiction

## License

[MIT License](LICENSE)


## Acknowledgements

- Thanks to all contributors who have helped develop and improve this project
