# Maestro

## Project Overview

Maestro is a comprehensive, production-ready pipeline for training deep learning models.

## Key Features

### Architecture
- **Modular Design**: Clean separation of concerns across multiple Python modules
- **Object-Oriented**: Encapsulated functionality in well-designed classes
- **Flexible Configuration**: Support for command-line and JSON-based configuration

### Model Training
- **Transfer Learning**: Leverages pre-trained models (ResNet, Xception, DenseNet, etc.)
- **Cross-Validation**: Implements stratified k-fold cross-validation
- **Class Imbalance Handling**: Automatic computation of class weights
- **Early Stopping**: Prevents overfitting by monitoring validation metrics
- **Learning Rate Scheduling**: Reduces learning rate on plateaus
- **Mixed Precision Training**: Improves performance on compatible GPUs

### Distributed Computing
- **Multi-GPU Support**: Scales across multiple GPUs on a single machine
- **Distributed Training**: Supports multi-node distributed training
- **Memory Optimization**: Careful management of GPU memory growth

### Data Handling
- **Data Augmentation**: Implements rotation, zoom, flip, and other transformations
- **Efficient Data Loading**: Optimized data loading with proper worker configuration
- **Validation**: Robust error handling for data loading issues

### Evaluation & Visualization
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC, Cohen's kappa
- **Visualizations**: Confusion matrices, ROC curves, precision-recall curves
- **TensorBoard Integration**: Real-time monitoring of training progress

### Experiment Management
- **Experiment Tracking**: Records all configuration parameters, system info, and results
- **Reproducibility**: Ensures experiments can be reproduced through saved configurations
- **Unique Experiment Directories**: Organized storage of all artifacts and results

## Code Organization

```
maestro/
├── config.py               # Configuration management
├── data.py                 # Data loading and preprocessing
├── models.py               # Model architecture definitions
├── training.py             # Training pipeline
├── evaluation.py           # Evaluation metrics and visualization
├── maestro_streamlit.py    # Streamlit GUI
├── utils.py                # Utility functions and experiment tracking
└── main.py                 # Entry point
```

## Usage

### Basic Usage
```bash
python maestro.py --model ResNet50 --data-path /path/to/data --image-size 224 224 --batch-size 32
```

### Using Configuration File
```bash
python maestro.py --config experiment_config.json
```

## Extending the Framework

### Adding New Models
Extend the `AVAILABLE_MODELS` dictionary in the `ModelFactory` class.

### Custom Data Preprocessing
Modify the data generators in the training module or extend the data loading functions.

### Custom Loss Functions
Implement new loss functions and add them to the model compilation step.

## Technical Details

- **Language & Framework**: Python with TensorFlow/Keras
- **Required Libraries**: TensorFlow, scikit-learn, pandas, matplotlib, seaborn, psutil
- **Hardware Requirements**: Compatible with CPU and GPU setups, optimized for NVIDIA GPUs
- **Input Data Format**: Directory structure with class subdirectories containing images

## Implementation Notes

The framework is designed with best practices in machine learning engineering:
- Strong typing with Python type hints
- Comprehensive docstrings
- Robust error handling
- Memory management for long training runs
- Scalability from single GPU to multi-node setups

Maestro serves as both a practical tool for medical image classification tasks and an educational resource for deep learning practitioners seeking to understand proper pipeline architecture.
