import os
import gc
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Tuple, List, Any, Optional

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, CSVLogger, TensorBoard, 
    ReduceLROnPlateau, ModelCheckpoint, Callback
)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# Local imports
from config import TrainingConfig
from models import ModelFactory

print("Loading training module")

class TimingCallback(Callback):
    """Callback to measure training time per epoch"""
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        logs = logs or {}
        logs['time_per_epoch'] = epoch_time
        logging.info(f"Epoch {epoch+1} took {epoch_time:.2f} seconds")

class Trainer:
    """Class to handle model training with cross-validation"""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration object
        """
        self.config = config
        self.experiment_dir = config.create_experiment_dir()
        
        # Setup logging
        log_file = os.path.join(self.experiment_dir, 'training.log')
        self._setup_logging(log_file)
        
        # Set up mixed precision if requested
        if config.mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logging.info("Mixed precision training enabled")
        
        # Set up distribution strategy
        self.strategy = self._setup_distribution_strategy()

    def _setup_logging(self, log_file: str) -> None:
        """Setup logging configuration"""
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    def _setup_distribution_strategy(self) -> tf.distribute.Strategy:
        """Set up appropriate distribution strategy based on available hardware"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if not gpus:
            logging.info("No GPUs detected, using default strategy")
            return tf.distribute.get_strategy()
        
        # Configure memory growth to avoid OOM errors
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Enabled memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logging.error(f"Could not set memory growth: {e}")
        
        # IMPORTANT: Don't use MultiWorkerMirroredStrategy unless specifically 
        # configured for distributed training with proper cluster settings
        
        # Check if TF_CONFIG is set for a multi-worker setup
        if "TF_CONFIG" in os.environ:
            try:
                # Only use MultiWorkerMirroredStrategy if properly configured
                import json
                tf_config = json.loads(os.environ["TF_CONFIG"])
                
                # Only proceed with multi-worker if there are multiple workers defined
                # AND we're not trying to run everything on one machine
                if (
                    "cluster" in tf_config and 
                    "worker" in tf_config["cluster"] and 
                    len(tf_config["cluster"]["worker"]) > 1
                ):
                    logging.info("Using MultiWorkerMirroredStrategy for distributed training")
                    # Use a different port range to avoid conflicts
                    options = tf.distribute.experimental.CommunicationOptions(
                        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
                    )
                    return tf.distribute.MultiWorkerMirroredStrategy(
                        communication_options=options
                    )
                else:
                    logging.info("TF_CONFIG exists but not configured for multi-worker, using MirroredStrategy")
            except Exception as e:
                logging.error(f"Error setting up MultiWorkerMirroredStrategy: {e}")
                logging.info("Falling back to MirroredStrategy")
        
        # If using multiple GPUs on a single machine
        if len(gpus) > 1:
            logging.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
            return tf.distribute.MirroredStrategy()
        
        # Single GPU
        logging.info("Using default strategy with single device")
        return tf.distribute.get_strategy()
    
    def _create_data_generators(self, 
                              train_df: pd.DataFrame, 
                              val_df: pd.DataFrame,
                              test_df: pd.DataFrame) -> Tuple:
        """
        Create data generators for training, validation and testing
        
        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame  
            test_df: Test data DataFrame
            
        Returns:
            Tuple of (train_generator, val_generator, test_generator)
        """
        # Create data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            # Add normalization
            rescale=1./255
        )
        
        # Only rescaling for validation/test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Performance optimizations for data loading
        common_params = {
            'target_size': self.config.image_size,
            'batch_size': self.config.batch_size,
            'class_mode': 'binary',
            'workers': min(os.cpu_count() or 4, 8),  # Limit number of workers
            'max_queue_size': 10,
            'use_multiprocessing': True
        }
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='image_path',
            y_col='label',
            shuffle=True,
            **common_params
        )
        
        val_generator = test_datagen.flow_from_dataframe(
            val_df,
            x_col='image_path',
            y_col='label',
            shuffle=False,
            **common_params
        )
        
        test_generator = test_datagen.flow_from_dataframe(
            test_df,
            x_col='image_path',
            y_col='label',
            shuffle=False,
            **common_params
        )
        
        return train_generator, val_generator, test_generator
    
    def _create_callbacks(self, fold: int) -> List[Callback]:
        """
        Create callbacks for training
        
        Args:
            fold: Current fold number
            
        Returns:
            List of callbacks
        """
        fold_dir = os.path.join(self.experiment_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        
        callbacks = [
            # Early stopping based on validation AUC
            EarlyStopping(
                monitor='val_auc',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                mode='max'
            ),
            
            # Log training metrics to CSV
            CSVLogger(
                os.path.join(fold_dir, 'training_log.csv'),
                append=True
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(fold_dir, 'tensorboard'),
                histogram_freq=1,
                profile_batch='500,520'  # Profile performance on batches 500-520
            ),
            
            # Reduce learning rate when metrics plateau
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.2,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                mode='max'
            ),
            
            # Save best model
            ModelCheckpoint(
                os.path.join(fold_dir, 'best_model.keras'),
                save_best_only=True,
                monitor='val_auc',
                mode='max'
            ),
            
            # Custom timing callback
            TimingCallback()
        ]
        
        return callbacks
    
    def _compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced datasets
        
        Args:
            labels: Array of encoded labels
            
        Returns:
            Dictionary mapping class indices to weights
        """
        try:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels),
                y=labels
            )
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
            return class_weight_dict
        except Exception as e:
            logging.warning(f"Could not compute class weights: {e}. Using equal weights.")
            unique_classes = np.unique(labels)
            return {i: 1.0 for i in range(len(unique_classes))}
    
    def train_with_cross_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model with stratified k-fold cross-validation
        
        Args:
            df: DataFrame with image paths and labels
            
        Returns:
            Dictionary with training metrics
        """
        logging.info(f"Starting {self.config.k_folds}-fold cross validation with {self.config.model_name}")
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        df['encoded_label'] = label_encoder.fit_transform(df['label'])
        class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        
        # Save class mapping
        with open(os.path.join(self.experiment_dir, 'class_mapping.txt'), 'w') as f:
            for idx, label in class_mapping.items():
                f.write(f"{idx}: {label}\n")
        
        # Split into train and test sets
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['encoded_label'], 
            random_state=42
        )
        
        # Initialize metrics storage
        all_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc': [],
            'training_time': []
        }
        
        # Initialize confusion matrix
        from sklearn.metrics import confusion_matrix
        num_classes = len(np.unique(df['encoded_label']))
        combined_conf_matrix = np.zeros((num_classes, num_classes))
        
        # Create stratified k-fold splitter
        skf = StratifiedKFold(
            n_splits=self.config.k_folds,
            shuffle=True,
            random_state=42
        )
        
        # Loop through folds
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df['image_path'], train_df['encoded_label'])):
            fold_start_time = time.time()
            logging.info(f"Starting fold {fold+1}/{self.config.k_folds}")
            
            # Split data for this fold
            train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
            val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
            
            # Create data generators
            train_generator, val_generator, test_generator = self._create_data_generators(
                train_fold_df, val_fold_df, test_df
            )
            
            # Compute class weights
            class_weights = self._compute_class_weights(train_fold_df['encoded_label'].values)
            
            # Create callbacks
            callbacks = self._create_callbacks(fold)
            
            # Create model within strategy scope
            with self.strategy.scope():
                model = ModelFactory.create_model(
                    model_name=self.config.model_name,
                    input_shape=(*self.config.image_size, 3),
                    num_classes=1,  # Binary classification
                    dropout_rate=self.config.dropout_rate,  # Pass the dropout rate
                    learning_rate=self.config.learning_rate
                )

            # Train model
            history = model.fit(
                train_generator,
                epochs=self.config.epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=self.config.verbose,
                class_weight=class_weights
            )
            
            # Calculate total training time
            fold_time = time.time() - fold_start_time
            all_metrics['training_time'].append(fold_time)
            logging.info(f"Fold {fold+1} training completed in {fold_time:.2f} seconds")
            
            # Evaluate on test set
            from evaluation import evaluate_model
            fold_metrics, fold_conf_matrix = evaluate_model(
                model, 
                test_generator,
                os.path.join(self.experiment_dir, f'fold_{fold+1}')
            )
            
            # Store metrics
            for metric, value in fold_metrics.items():
                if metric in all_metrics:
                    all_metrics[metric].append(value)
            
            # Update combined confusion matrix
            combined_conf_matrix += fold_conf_matrix
            
            # Clean up to prevent memory leaks
            tf.keras.backend.clear_session()
            gc.collect()
        
        # Calculate and save final metrics
        final_metrics = self._save_final_metrics(all_metrics, combined_conf_matrix)
        
        return final_metrics
    
    def _save_final_metrics(self, 
                          all_metrics: Dict[str, List[float]], 
                          conf_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate and save final metrics
        
        Args:
            all_metrics: Dictionary with lists of metrics from each fold
            conf_matrix: Combined confusion matrix
            
        Returns:
            Dictionary with final metric summaries
        """
        # Calculate mean and std for each metric
        final_metrics = {}
        
        for metric, values in all_metrics.items():
            if len(values) > 0:
                final_metrics[f"{metric}_mean"] = np.mean(values)
                final_metrics[f"{metric}_std"] = np.std(values)
        
        # Add average epoch time if available
        if 'time_per_epoch' in all_metrics and all_metrics['time_per_epoch']:
            final_metrics['avg_time_per_epoch'] = np.mean(all_metrics['time_per_epoch'])
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame([final_metrics])
        metrics_df.to_csv(os.path.join(self.experiment_dir, 'final_metrics.csv'), index=False)
        
        # Save confusion matrix
        np.savetxt(
            os.path.join(self.experiment_dir, 'confusion_matrix.csv'),
            conf_matrix,
            delimiter=','
        )
        
        # Log final results
        logging.info("Cross-validation completed. Final metrics:")
        for metric, value in final_metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        return final_metrics