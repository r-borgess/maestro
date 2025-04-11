<div align="center">
    <img src="assets/images/SVG/logo-full.svg" alt="maestro logo" width="1000">
</div>

# maestro: Deep Learning Pipeline Framework

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.19.0](https://img.shields.io/badge/tensorflow-2.19.0-orange.svg)](https://www.tensorflow.org/)

**A Comprehensive Framework for Training and Evaluating Deep Learning Models**
</div>

## Overview

maestro is a production-ready framework for building, training, and evaluating deep learning models with a focus on image classification tasks. It provides a robust, modular architecture that handles the full machine learning lifecycle while following best practices in deep learning engineering.

### Key Features

- **Modular and Extensible Design**: Clean architecture with separation of concerns
- **Automated Cross-Validation**: Supports both k-fold cross-validation and holdout validation
- **Transfer Learning**: Leverages pre-trained models from the Keras Applications API
- **Advanced Training Features**:
  - Learning rate scheduling
  - Early stopping
  - Class imbalance handling
  - Mixed precision training
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC, and more
- **Hardware Optimization**: Support for multi-GPU setups and memory optimization
- **Experiment Tracking**: Records all parameters, metrics, and results for reproducibility
- **Interactive UI**: Streamlit-based interface for configuration, training, and visualization

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/maestro.git
cd maestro
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Train a model with specific parameters:

```bash
python main.py --model ResNet50 --data-path /path/to/data --image-size 224 224 --batch-size 32
```

Train using a configuration file:

```bash
python main.py --config configs/my_config.json
```

Evaluate a trained model:

```bash
python evaluate.py --model-path results/ResNet50_20250403-150211/fold_1/best_model.keras --data-path /path/to/test_data
```

### Interactive UI

Launch the Streamlit interface for a more user-friendly experience:

```bash
streamlit run maestro_streamlit.py
```

The UI provides:
- Dataset exploration and visualization
- Model configuration with interactive controls
- Training monitoring
- Results visualization
- Model evaluation

## Configuration Options

maestro supports extensive configuration through both command-line arguments and JSON files:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Model architecture to use | Required |
| `data_path` | Path to dataset directory | Required |
| `image_size` | Image dimensions (height, width) | (224, 224) |
| `batch_size` | Batch size for training | 32 |
| `epochs` | Maximum number of epochs | 100 |
| `validation_strategy` | "cross_validation" or "holdout" | "cross_validation" |
| `k_folds` | Number of folds for cross-validation | 5 |
| `holdout_split` | Proportion for validation when using holdout | 0.2 |
| `test_split` | Proportion for test set | 0.2 |
| `learning_rate` | Initial learning rate | 0.0001 |
| `dropout_rate` | Dropout rate for regularization | 0.5 |
| `early_stopping_patience` | Patience for early stopping | 10 |
| `reduce_lr_patience` | Patience for reducing learning rate | 5 |
| `mixed_precision` | Enable mixed precision training | True |
| `save_dir` | Directory to save results | "./results" |
| `verbose` | Verbosity level (0, 1, or 2) | 1 |

## Dataset Structure

maestro expects datasets to be organized with the following structure:

```
data_directory/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Supported Models

maestro supports the following model architectures:

- ResNet50, ResNet101
- Xception
- VGG16, VGG19
- DenseNet121, DenseNet201
- MobileNet, MobileNetV2
- EfficientNetB3

## Output Structure

maestro organizes training results in the following structure:

```
results/
└── ModelName_YYYYMMDD-HHMMSS/
    ├── config.json
    ├── experiment_results.json
    ├── class_mapping.txt
    ├── fold_1/
    │   ├── best_model.keras
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   └── ...
    ├── fold_2/
    │   └── ...
    └── ...
```

## Extending maestro

### Adding New Models

Extend the `ModelFactory` class in `models.py` by adding your model to the `AVAILABLE_MODELS` dictionary:

```python
AVAILABLE_MODELS = {
    "Xception": Xception,
    "ResNet50": ResNet50,
    # Add your model here
    "MyNewModel": MyNewModelFunction
}
```

### Custom Data Preprocessing

Modify the data generators in `training.py` to add custom preprocessing:

```python
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    # Add custom preprocessing
    preprocessing_function=my_custom_preprocessing
)
```

## Examples

### Basic Training Example

```python
from config import TrainingConfig
from data import load_image_data
from training import Trainer

# Create configuration
config = TrainingConfig(
    model_name="ResNet50",
    data_path="./datasets/images",
    image_size=(224, 224),
    batch_size=32,
    epochs=100
)

# Load data
df = load_image_data(config.data_path)

# Create trainer and run training
trainer = Trainer(config)
metrics = trainer.train(df)

print(f"Final validation accuracy: {metrics['accuracy_mean']:.4f}")
```

### Evaluation Example

```python
import tensorflow as tf
from evaluation import evaluate_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model("./results/mymodel/best_model.keras")

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "./data/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Evaluate model
metrics, conf_matrix = evaluate_model(model, test_generator, "./evaluation_results")
print(f"Test accuracy: {metrics['accuracy']:.4f}")
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to maestro.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.