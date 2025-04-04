import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from typing import Dict, Tuple, List, Any, Optional, Union

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

def evaluate_model(model: tf.keras.Model, 
                  test_generator: Any,  # Use Any instead of specific type for compatibility
                  output_dir: str) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate model performance on test set and save visualizations
    
    Args:
        model: Trained Keras model
        test_generator: Generator for test data
        output_dir: Directory to save evaluation results
        
    Returns:
        Tuple of (metrics_dict, confusion_matrix)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Reset generator to start
    test_generator.reset()
    
    # Get predictions
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = test_generator.classes
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Calculate AUC
    if len(np.unique(y_true)) == 2:  # Binary classification
        metrics['auc'] = roc_auc_score(y_true, y_pred_prob)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_prob)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Save metrics to CSV
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(output_dir, 'classification_report.csv')
    )
    
    # Create visualizations
    _create_confusion_matrix_plot(cm, output_dir, test_generator.class_indices)
    
    # Create ROC curve for binary classification
    if len(np.unique(y_true)) == 2:
        _create_roc_curve_plot(y_true, y_pred_prob, output_dir)
        _create_precision_recall_curve_plot(y_true, y_pred_prob, output_dir)
    
    return metrics, cm

def roc_auc_score(y_true, y_pred_prob):
    """Calculate ROC AUC score for binary classification"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        return auc(fpr, tpr)
    except Exception as e:
        print(f"Error calculating ROC AUC: {e}")
        return 0.0

def _create_confusion_matrix_plot(cm: np.ndarray, 
                                output_dir: str,
                                class_indices: Dict[str, int]) -> None:
    """Create and save confusion matrix visualization"""
    plt.figure(figsize=(10, 8))
    
    # Get class names in correct order
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    
    # Create normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def _create_roc_curve_plot(y_true: np.ndarray, 
                          y_pred_prob: np.ndarray,
                          output_dir: str) -> None:
    """Create and save ROC curve visualization"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()

def _create_precision_recall_curve_plot(y_true: np.ndarray,
                                      y_pred_prob: np.ndarray,
                                      output_dir: str) -> None:
    """Create and save precision-recall curve visualization"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    plt.close()