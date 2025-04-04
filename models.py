import tensorflow as tf
from tensorflow import keras
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from typing import Tuple, Dict, Optional, Any

from keras.applications import (
    Xception, ResNet50, ResNet101, VGG16, VGG19, 
    EfficientNetB3, DenseNet201, MobileNet, DenseNet121, MobileNetV2
)

print("Loading models module")

# models.py - ModelFactory class update
class ModelFactory:
    """Factory class for creating different model architectures"""
    
    AVAILABLE_MODELS = {
        "Xception": Xception,
        "ResNet50": ResNet50,
        "ResNet101": ResNet101,
        "VGG16": VGG16,
        "VGG19": VGG19,
        "EfficientNetB3": EfficientNetB3,
        "DenseNet201": DenseNet201,
        "MobileNet": MobileNet,
        "DenseNet121": DenseNet121,
        "MobileNetV2": MobileNetV2
    }
    
    @classmethod
    def create_model(cls, 
                    model_name: str, 
                    input_shape: Tuple[int, int, int],
                    num_classes: int = 1,
                    dropout_rate: float = 0.5,  # Updated to pass dropout_rate
                    learning_rate: float = 1e-4) -> Model:
        """
        Create a model with the specified architecture
        
        Args:
            model_name: Name of the model architecture
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes (1 for binary classification)
            dropout_rate: Dropout rate for regularization (0.0 to 1.0)
            learning_rate: Learning rate for the optimizer
            
        Returns:
            Compiled Keras model
        """
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(cls.AVAILABLE_MODELS.keys())}")
        
        # Create base model
        base_model = cls.AVAILABLE_MODELS[model_name](
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
        
        # Add custom layers
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(dropout_rate)(x)  # Use the provided dropout_rate
        
        # Output layer - sigmoid for binary classification
        if num_classes == 1:
            predictions = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            predictions = Dense(num_classes, activation='softmax')(x)
            loss = 'sparse_categorical_crossentropy'
        
        # Create and compile model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with appropriate metrics
        metrics = [
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]
        
        if num_classes == 1:
            # Add binary classification metrics
            metrics.extend([
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model