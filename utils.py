import os
import json
import time
import pickle
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("Loading utils module")

def set_tf_memory_growth() -> None:
    """Configure TensorFlow to grow GPU memory allocation as needed"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Memory growth enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            logging.error(f"Failed to set memory growth: {e}")

def save_model_summary(model: tf.keras.Model, filepath: str) -> None:
    """Save model summary to a text file"""
    # Redirect summary to file
    from contextlib import redirect_stdout
    with open(filepath, 'w') as f:
        with redirect_stdout(f):
            model.summary()

def save_history_plots(history: Dict[str, List], output_dir: str) -> None:
    """
    Save plots of training history
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Make plots for each metric
    metrics = [m for m in history.keys() if not m.startswith('val_')]
    
    for metric in metrics:
        if f'val_{metric}' in history:
            plt.figure(figsize=(10, 6))
            plt.plot(history[metric], label=f'Training {metric}')
            plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'{metric} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{metric}_plot.png'))
            plt.close()

def save_model_architecture(model: tf.keras.Model, filepath: str) -> None:
    """Save model architecture as JSON"""
    try:
        model_json = model.to_json()
        with open(filepath, "w") as json_file:
            json_file.write(model_json)
    except Exception as e:
        logging.error(f"Could not save model architecture: {e}")

def calculate_flops(model: tf.keras.Model) -> Tuple[float, str]:
    """
    Calculate FLOPs for a Keras model
    
    Args:
        model: Keras model
        
    Returns:
        Tuple of (flops_value, readable_flops)
    """
    try:
        # Run model once to build it
        input_shape = model.inputs[0].shape.as_list()
        if None in input_shape:
            input_shape = [1 if x is None else x for x in input_shape]
        
        dummy_input = np.zeros([1] + input_shape[1:])
        _ = model(dummy_input)
        
        # Calculate FLOPs
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        
        forward_pass = tf.function(
            model.call,
            input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)]
        )
        
        graph_info = profile(
            forward_pass.get_concrete_function(tf.zeros(input_shape)).graph,
            options=ProfileOptionBuilder.float_operation()
        )
        
        flops = graph_info.total_float_ops
        
        # Convert to readable format
        units = ['', 'K', 'M', 'G', 'T']
        readable_flops = flops
        unit_idx = 0
        
        while readable_flops > 1000 and unit_idx < len(units) - 1:
            readable_flops /= 1000
            unit_idx += 1
            
        return flops, f"{readable_flops:.2f} {units[unit_idx]}FLOPs"
    except Exception as e:
        logging.error(f"Error calculating FLOPs: {e}")
        return 0, "Unknown FLOPs"

class ExperimentTracker:
    """Class to track experiment settings and results"""
    
    def __init__(self, experiment_dir: str):
        """
        Initialize tracker
        
        Args:
            experiment_dir: Directory to save experiment data
        """
        self.experiment_dir = experiment_dir
        self.results_file = os.path.join(experiment_dir, "experiment_results.json")
        self.experiment_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {},
            "results": {},
            "system_info": self._get_system_info()
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import platform
        import psutil
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpu_info = []
        
        for gpu in gpus:
            gpu_info.append(str(gpu))
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "tensorflow_version": tf.__version__,
            "gpus": gpu_info
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set configuration parameters"""
        self.experiment_data["config"] = config
    
    def update_results(self, results: Dict[str, Any]) -> None:
        """Update results with new data"""
        self.experiment_data["results"].update(results)
    
    def save(self) -> None:
        """Save experiment data to JSON file"""
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        
        with open(self.results_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=4)
