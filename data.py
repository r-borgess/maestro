import os
import logging
import traceback
import pandas as pd
from typing import Dict, List, Optional, Tuple

print("Loading data module")

def load_image_data(data_path):
    """
    Load image paths and labels with improved error handling
    
    Args:
        data_path (str): Path to the dataset directory
        
    Returns:
        pd.DataFrame: DataFrame with image paths and labels
    """
    image_paths, labels = [], []
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    try:
        class_dirs = os.listdir(data_path)
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {data_path}")
            
        for label in class_dirs:
            class_dir = os.path.join(data_path, label)
            if not os.path.isdir(class_dir):
                continue
                
            files = [f for f in os.listdir(class_dir) 
                    if os.path.splitext(f.lower())[1] in valid_extensions]
            
            if len(files) == 0:
                logging.warning(f"No valid images found in class directory: {class_dir}")
                continue
                
            for filename in files:
                file_path = os.path.join(class_dir, filename)
                if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                    image_paths.append(file_path)
                    labels.append(label)
                else:
                    logging.warning(f"Skipping invalid file: {file_path}")
        
        if len(image_paths) == 0:
            raise ValueError("No valid images found in the dataset")
            
        df = pd.DataFrame({'image_path': image_paths, 'label': labels})
        logging.info(f"Loaded {len(df)} images from {len(df['label'].unique())} classes")
        return df
        
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        logging.error(traceback.format_exc())
        raise