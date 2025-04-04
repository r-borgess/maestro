# config.py
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

@dataclass
class TrainingConfig:
    model_name: str
    data_path: str
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 100
    k_folds: int = 5
    learning_rate: float = 1e-4
    dropout_rate: float = 0.5  # Added dropout_rate parameter with default
    verbose: int = 1
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    mixed_precision: bool = True
    save_dir: str = './results'
    
    @classmethod
    def from_args(cls, args):
        return cls(
            model_name=args.model,
            data_path=args.data_path,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size,
            epochs=args.epochs,
            k_folds=args.k,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,  # Added dropout_rate
            verbose=args.verbose,
            early_stopping_patience=args.early_stopping_patience,
            reduce_lr_patience=args.reduce_lr_patience,
            mixed_precision=args.mixed_precision,
            save_dir=args.save_dir
        )
    
    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def save(self, json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def create_experiment_dir(self):
        """Create unique experiment directory based on parameters"""
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        exp_dir = os.path.join(
            self.save_dir, 
            f"{self.model_name}_{timestamp}"
        )
        os.makedirs(exp_dir, exist_ok=True)
        self.save(os.path.join(exp_dir, "config.json"))
        return exp_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument('--model', type=str, required=True, 
                        choices=["Xception", "ResNet50", "ResNet101", "VGG16", "VGG19", 
                                "EfficientNetB3", "DenseNet201", "MobileNet", "DenseNet121", "MobileNetV2"],
                        help="Name of the model architecture to use")
    parser.add_argument('--data-path', type=str, required=True,
                        help="Path to the data directory")
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                        help="Image dimensions (height, width)")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=100,
                        help="Maximum number of epochs to train")
    parser.add_argument('--k', type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument('--dropout-rate', type=float, default=0.5,  # Added dropout-rate argument
                        help="Dropout rate for regularization (0.0 to 1.0)")
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level")
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument('--reduce-lr-patience', type=int, default=5,
                        help="Patience for reducing learning rate")
    parser.add_argument('--mixed-precision', action='store_true',
                        help="Use mixed precision training")
    parser.add_argument('--save-dir', type=str, default='./results',
                        help="Directory to save results")
    parser.add_argument('--config', type=str,
                        help="Path to config JSON file (overrides other arguments)")
    
    args = parser.parse_args()
    
    # If config file is provided, load it and override CLI arguments
    if args.config:
        return TrainingConfig.from_json(args.config)
    else:
        return TrainingConfig.from_args(args)