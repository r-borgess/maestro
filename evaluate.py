#!/usr/bin/env python
# evaluate.py - Script for evaluating a trained model on a different dataset

import os
import json
import argparse
import logging
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s'
)

# Import local modules
from data import load_image_data
from evaluation import evaluate_model
from utils import set_tf_memory_growth
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def parse_args():
    """Parse command line arguments for model evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate a trained Maestro model on a dataset")
    parser.add_argument('--model-path', type=str, required=True,
                        help="Path to the trained model file (.keras or .h5)")
    parser.add_argument('--data-path', type=str, required=True,
                        help="Path to the evaluation dataset directory")
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help="Base directory to save evaluation results")
    parser.add_argument('--name', type=str, default=None,
                        help="Custom name for this evaluation (will be used as subdirectory name)")
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                        help="Image dimensions (height, width)")
    parser.add_argument('--class-mapping', type=str, default=None,
                        help="Path to class mapping file (optional)")
    parser.add_argument('--model-config', type=str, default=None,
                        help="Path to model configuration JSON (optional)")
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help="Verbosity level")
    
    return parser.parse_args()

def load_class_mapping(mapping_path: str) -> Dict[int, str]:
    """
    Load class mapping from a file
    
    Args:
        mapping_path: Path to class mapping file
        
    Returns:
        Dictionary mapping class indices to class names
    """
    class_mapping = {}
    
    with open(mapping_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ':' in line:
                idx, label = line.split(':', 1)
                class_mapping[int(idx.strip())] = label.strip()
                
    return class_mapping

def infer_classes_from_directory(data_path: str) -> Dict[str, int]:
    """
    Infer class names and indices from directory structure
    
    Args:
        data_path: Path to dataset directory
        
    Returns:
        Dictionary mapping class names to indices
    """
    class_dirs = [d for d in os.listdir(data_path) 
                 if os.path.isdir(os.path.join(data_path, d))]
    
    return {class_name: idx for idx, class_name in enumerate(sorted(class_dirs))}

def prepare_evaluation_data(data_path: str, 
                           image_size: Tuple[int, int], 
                           batch_size: int,
                           class_indices: Optional[Dict[str, int]] = None) -> Any:
    """
    Prepare data generator for evaluation
    
    Args:
        data_path: Path to dataset directory
        image_size: Image dimensions (height, width)
        batch_size: Batch size for evaluation
        class_indices: Optional dictionary mapping class names to indices
        
    Returns:
        Data generator for evaluation
    """
    # Load image data
    df = load_image_data(data_path)
    
    # Create data generator
    datagen = ImageDataGenerator(rescale=1./255)
    
    common_params = {
        'target_size': image_size,
        'batch_size': batch_size,
        'class_mode': 'binary',  # Will be adjusted automatically for multi-class
        'shuffle': False,  # No shuffling for evaluation
        'workers': min(os.cpu_count() or 4, 8),
        'max_queue_size': 10,
        'use_multiprocessing': True
    }
    
    generator = datagen.flow_from_dataframe(
        df,
        x_col='image_path',
        y_col='label',
        classes=class_indices,  # Pass provided class indices if available
        **common_params
    )
    
    return generator

def save_model_info(model: tf.keras.Model, output_dir: str) -> None:
    """
    Save model architecture and summary
    
    Args:
        model: Loaded model
        output_dir: Directory to save model information
    """
    # Save model architecture
    try:
        model_json = model.to_json()
        with open(os.path.join(output_dir, "model_architecture.json"), "w") as f:
            f.write(model_json)
    except Exception as e:
        logging.warning(f"Could not save model architecture: {e}")
    
    # Save model summary
    from contextlib import redirect_stdout
    with open(os.path.join(output_dir, "model_summary.txt"), "w") as f:
        with redirect_stdout(f):
            model.summary()

def main():
    """Main entry point for model evaluation"""
    args = parse_args()
    
    # Set memory growth for GPUs
    set_tf_memory_growth()
    
    # Create a specific output directory for this evaluation
    # If name is provided, use it; otherwise, generate from model name and timestamp
    if args.name:
        eval_name = args.name
    else:
        # Extract model name from path
        model_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
        # If couldn't extract meaningful name, use model filename
        if not model_name or model_name == '':
            model_name = os.path.basename(args.model_path).split('.')[0]
        
        # Extract dataset name from path
        dataset_name = os.path.basename(args.data_path)
        if not dataset_name or dataset_name == '':
            dataset_name = 'dataset'
            
        # Create timestamped name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        eval_name = f"{model_name}_{dataset_name}_{timestamp}"
    
    # Construct full output path
    output_dir = os.path.join(args.output_dir, eval_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save command line arguments for reproducibility
    with open(os.path.join(output_dir, 'evaluation_params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    try:
        # Load model
        logging.info(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)
        
        # Save model information
        save_model_info(model, output_dir)
        
        # Determine class mapping
        class_indices = None
        
        if args.class_mapping:
            # Load class mapping from file
            logging.info(f"Loading class mapping from {args.class_mapping}")
            class_mapping = load_class_mapping(args.class_mapping)
            # Convert to format expected by flow_from_dataframe
            class_indices = {v: k for k, v in class_mapping.items()}
        elif args.model_config:
            # Try to extract class info from model configuration
            logging.info(f"Loading configuration from {args.model_config}")
            try:
                with open(args.model_config, 'r') as f:
                    config = json.load(f)
                    
                # Check if original data path is in config
                if 'data_path' in config:
                    original_data_path = config['data_path']
                    logging.info(f"Inferring classes from original data path: {original_data_path}")
                    if os.path.exists(original_data_path):
                        class_indices = infer_classes_from_directory(original_data_path)
            except Exception as e:
                logging.warning(f"Could not extract class information from config: {e}")
        
        # If no class mapping provided, infer from evaluation dataset
        if class_indices is None:
            logging.info(f"Inferring classes from evaluation dataset: {args.data_path}")
            class_indices = infer_classes_from_directory(args.data_path)
        
        # Log class mapping
        logging.info(f"Using class mapping: {class_indices}")
        
        # Prepare evaluation data
        logging.info(f"Preparing evaluation data from {args.data_path}")
        eval_generator = prepare_evaluation_data(
            args.data_path,
            tuple(args.image_size),
            args.batch_size,
            class_indices
        )
        
        # Run evaluation
        logging.info("Evaluating model")
        metrics, conf_matrix = evaluate_model(
            model,
            eval_generator,
            output_dir
        )
        
        # Save metrics as JSON
        with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        # Save confusion matrix
        np.save(os.path.join(output_dir, 'confusion_matrix.npy'), conf_matrix)
        
        # Print summary metrics
        logging.info("Evaluation complete. Summary:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value}")
            
        return 0
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())