# main.py
import os
import logging
import traceback
from typing import Optional

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s'
)

# Then import local modules
from config import parse_args, TrainingConfig
from data import load_image_data
from training import Trainer

print("Loading main module")

# At the beginning of your main.py script
os.makedirs("configs", exist_ok=True)
os.makedirs("results", exist_ok=True)

def setup_environment() -> None:
    """Setup environment variables and configurations"""
    # Set TF_CPP_MIN_LOG_LEVEL to 2 to filter out INFO and WARNING messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Clear TF_CONFIG if it's not needed for your use case
    # If you're not doing distributed training across multiple machines,
    # having TF_CONFIG set can cause problems
    if 'TF_CONFIG' in os.environ and not os.environ.get('ENABLE_DISTRIBUTED', False):
        logging.info("Clearing TF_CONFIG as distributed training is not enabled")
        del os.environ['TF_CONFIG']
    
    # Set memory growth for GPUs if available
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(f"Could not set memory growth: {e}")
            
def main() -> Optional[int]:
    """Main entry point for the training pipeline"""
    try:
        # Setup environment
        setup_environment()
        
        # Parse command line arguments
        config = parse_args()
        
        # Load data
        logging.info(f"Loading data from {config.data_path}")
        df = load_image_data(config.data_path)
        
        # Create trainer
        trainer = Trainer(config)
        
        # Run training - now using the unified train method that selects
        # between cross-validation and holdout methods
        metrics = trainer.train(df)
        
        logging.info("Training completed successfully!")
        
        # Return success
        return 0
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code if exit_code is not None else 1)
