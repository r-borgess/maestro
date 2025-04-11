import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import subprocess
import sys
from pathlib import Path
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import tempfile
from tensorflow.keras.utils import plot_model


# Add the Maestro modules to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Maestro modules
from config import TrainingConfig
from models import ModelFactory
from utils import set_tf_memory_growth, ExperimentTracker

# Set page configuration
st.set_page_config(
    page_title="Maestro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🧠 Maestro")
st.markdown("A framework for image classification experiments with TensorFlow")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select a page", 
    ["Dashboard", "Configure Experiment", "Training", "Model Evaluation", "Results Viewer"]
)

# Check and display GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
gpu_info = f"Found {len(gpus)} GPU(s)" if gpus else "No GPUs found, using CPU"
st.sidebar.info(gpu_info)

def prepare_dataframe_for_display(df):
    """
    Convert any list/tuple values to strings to make the DataFrame Arrow-compatible
    
    Args:
        df: pandas DataFrame
        
    Returns:
        pandas DataFrame with list/tuple values converted to strings
    """
    df_display = df.copy()
    
    for col in df_display.columns:
        # Check if the column contains lists or tuples
        if df_display[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            # Convert lists/tuples to strings
            df_display[col] = df_display[col].apply(lambda x: str(x) if isinstance(x, (list, tuple)) else x)
    
    return df_display

# Function to get available models
def get_available_models():
    return list(ModelFactory.AVAILABLE_MODELS.keys())

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_experiments():
    results_dir = Path("./results")
    if not results_dir.exists():
        return []
    
    experiments = [d.name for d in results_dir.iterdir() if d.is_dir()]
    return sorted(experiments, reverse=True)  # Most recent first

@st.cache_data(ttl=300)
def load_experiment_results(experiment_name):
    results_path = Path(f"./results/{experiment_name}/experiment_results.json")
    if not results_path.exists():
        return None
    
    with open(results_path, "r") as f:
        return json.load(f)
    
# Safe DataFrame display function
def safe_display_dataframe(data, use_container_width=True):
    """
    Safely display data as a DataFrame, handling different input types
    
    Args:
        data: Data to display (DataFrame, dict, list of dicts)
        use_container_width: Whether to use full container width
    """
    try:
        # If it's already a DataFrame, use it directly
        if isinstance(data, pd.DataFrame):
            df = data
        # If it's a dict, convert to DataFrame
        elif isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
        # If it's a list of dicts, convert to DataFrame
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        # Otherwise, try to create a DataFrame
        else:
            df = pd.DataFrame(data)
            
        # Apply the prepare_dataframe_for_display function
        display_df = prepare_dataframe_for_display(df)
        
        # Display the DataFrame
        st.dataframe(display_df, use_container_width=use_container_width)
        
    except Exception as e:
        st.error(f"Error displaying data: {e}")
        st.write("Raw data:")
        st.write(data)

# Function to visualize dataset
def visualize_dataset(data_path, num_samples=4):
    """
    Visualize dataset statistics and sample images
    
    Args:
        data_path: Path to dataset directory
        num_samples: Number of sample images to show per class
    """
    if not os.path.exists(data_path):
        st.error(f"Path does not exist: {data_path}")
        return
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_path) 
                 if os.path.isdir(os.path.join(data_path, d))]
    
    if not class_dirs:
        st.error(f"No class directories found in {data_path}")
        return
    
    # Display dataset statistics
    stats = []
    for class_name in class_dirs:
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        stats.append({
            "Class": class_name,
            "Sample Count": len(image_files)
        })
    
    st.write("### Dataset Statistics")
    
    # Convert list of dictionaries to DataFrame before processing
    stats_df = pd.DataFrame(stats)
    safe_display_dataframe(stats_df)
    
    # Show sample images for each class
    st.write("### Sample Images")
    cols = st.columns(len(class_dirs))
    
    for i, class_name in enumerate(class_dirs):
        cols[i].write(f"**{class_name}**")
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            cols[i].write("No images found")
            continue
        
        # Select random samples
        import random
        samples = random.sample(image_files, min(num_samples, len(image_files)))
        
        for sample in samples:
            try:
                img_path = os.path.join(class_path, sample)
                img = Image.open(img_path)
                cols[i].image(img, caption=sample, use_column_width=True)
            except Exception as e:
                cols[i].error(f"Error loading image: {e}")

# Function to display model architecture
def display_model_architecture(model_name, image_size):
    try:
        with st.spinner(f"Creating {model_name} architecture..."):
            model = ModelFactory.create_model(
                model_name=model_name,
                input_shape=(*image_size, 3),
                num_classes=1  # Binary classification
            )
            
            # Convert model summary to string
            from io import StringIO
            summary_io = StringIO()
            model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
            summary_str = summary_io.getvalue()
            
            st.code(summary_str, language="text")
            
            # Display total parameters
            total_params = model.count_params()
            st.info(f"Total parameters: {total_params:,}")
            
    except Exception as e:
        st.error(f"Error creating model: {e}")


# Function to launch training process
def launch_training(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Map config keys to argument names expected by main.py
        arg_mapping = {
            "model_name": "model",
            "data_path": "data-path",
            "image_size": "image-size",
            "batch_size": "batch-size",
            "epochs": "epochs",
            "k_folds": "k",
            "learning_rate": "learning-rate",
            "early_stopping_patience": "early-stopping-patience",
            "reduce_lr_patience": "reduce-lr-patience",
            "mixed_precision": "mixed-precision",
            "save_dir": "save-dir",
            "verbose": "verbose"
        }
        
        # Convert config JSON to command line arguments
        cmd = [sys.executable, "main.py"]
        
        # Add all config parameters as command line arguments
        for key, value in config.items():
            # Map the config key to the expected argument name
            if key in arg_mapping:
                arg_key = f"--{arg_mapping[key]}"
                
                # Handle list values (like image_size)
                if isinstance(value, list):
                    cmd.append(arg_key)
                    for item in value:
                        cmd.append(str(item))
                # Handle boolean values
                elif isinstance(value, bool):
                    if value:
                        cmd.append(arg_key)
                # Handle regular values
                else:
                    cmd.append(arg_key)
                    cmd.append(str(value))
        
        # Display the command being executed
        st.code(" ".join(cmd), language="bash")
        
        with st.spinner("Starting training process..."):
            # Start the process
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # Create an empty placeholder for the log
            log_output = st.empty()
            
            # Stream the output
            full_log = ""
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    full_log += output
                    log_output.code(full_log, language="bash")
            
            # Get return code
            return_code = process.poll()
            
            if return_code == 0:
                st.success("Training completed successfully!")
            else:
                error = process.stderr.read()
                st.error(f"Training failed with return code {return_code}. Error: {error}")
                
    except Exception as e:
        st.error(f"Error launching training: {e}")

def plot_training_history(history_data):
    """
    Create interactive training history plot with Plotly
    
    Args:
        history_data: Dictionary with training history data
    
    Returns:
        Plotly figure object
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots with 2 y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add accuracy traces
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history_data['accuracy'])+1)), 
                  y=history_data['accuracy'],
                  name="Training Accuracy",
                  line=dict(color="blue")),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history_data['val_accuracy'])+1)), 
                  y=history_data['val_accuracy'],
                  name="Validation Accuracy",
                  line=dict(color="blue", dash="dash")),
        secondary_y=False
    )
    
    # Add loss traces
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history_data['loss'])+1)), 
                  y=history_data['loss'],
                  name="Training Loss",
                  line=dict(color="red")),
        secondary_y=True
    )
    
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history_data['val_loss'])+1)), 
                  y=history_data['val_loss'],
                  name="Validation Loss",
                  line=dict(color="red", dash="dash")),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="Training History",
        xaxis_title="Epoch",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    # Configure y-axes
    fig.update_yaxes(title_text="Accuracy", secondary_y=False)
    fig.update_yaxes(title_text="Loss", secondary_y=True)
    
    return fig

def compare_experiments(experiment_names):
    """
    Compare multiple experiments with interactive visualizations
    
    Args:
        experiment_names: List of experiment names to compare
    """
    if not experiment_names:
        st.warning("No experiments selected for comparison")
        return
    
    # Load all experiment data
    experiments_data = []
    for exp_name in experiment_names:
        exp_data = load_experiment_results(exp_name)
        if exp_data:
            experiments_data.append({
                "name": exp_name,
                "data": exp_data
            })
    
    if not experiments_data:
        st.error("Could not load experiment data")
        return
    
    # Extract key metrics for comparison
    import plotly.graph_objects as go
    
    metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'auc_mean']
    metrics_pretty = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    # Create metrics comparison chart
    fig = go.Figure()
    
    for exp in experiments_data:
        metric_values = []
        for metric in metrics:
            if metric in exp['data'].get('results', {}):
                metric_values.append(exp['data']['results'][metric])
            else:
                metric_values.append(None)
        
        fig.add_trace(go.Scatterpolar(
            r=metric_values,
            theta=metrics_pretty,
            fill='toself',
            name=exp['name']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Metrics Comparison"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare hyperparameters
    st.subheader("Hyperparameter Comparison")
    
    # Extract hyperparameters
    params_to_compare = [
        'model_name', 'batch_size', 'epochs', 'learning_rate', 
        'early_stopping_patience', 'dropout_rate'
    ]
    
    params_data = {}
    for exp in experiments_data:
        config = exp['data'].get('config', {})
        for param in params_to_compare:
            if param in config:
                if param not in params_data:
                    params_data[param] = []
                params_data[param].append({
                    'Experiment': exp['name'],
                    'Value': config[param]
                })
    
    # Create a table for each parameter
    for param, data in params_data.items():
        st.write(f"**{param.replace('_', ' ').title()}**")
        param_df = pd.DataFrame(data)
        safe_display_dataframe(param_df)

def enhanced_dataset_visualization(data_path):
    """
    Enhanced visualization of dataset with class distribution and image analysis
    
    Args:
        data_path: Path to dataset directory
    """
    if not os.path.exists(data_path):
        st.error(f"Path does not exist: {data_path}")
        return
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_path) 
                 if os.path.isdir(os.path.join(data_path, d))]
    
    if not class_dirs:
        st.error(f"No class directories found in {data_path}")
        return
    
    # Collect dataset statistics
    stats = []
    total_samples = 0
    
    for class_name in class_dirs:
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_count = len(image_files)
        total_samples += class_count
        stats.append({
            "Class": class_name,
            "Sample Count": class_count
        })
    
    # Add percentage
    for stat in stats:
        stat["Percentage"] = f"{(stat['Sample Count'] / total_samples * 100):.2f}%"
    
    st.write("### Dataset Statistics")
    
    # Convert list of dictionaries to DataFrame before processing
    stats_df = pd.DataFrame(stats)
    safe_display_dataframe(stats_df)
    
    # Create a pie chart for class distribution
    import plotly.express as px
    
    fig = px.pie(
        values=[stat["Sample Count"] for stat in stats],
        names=[stat["Class"] for stat in stats],
        title="Class Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Image dimensions analysis
    st.write("### Image Dimensions Analysis")
    
    # Sample some images from each class to analyze dimensions
    dimension_data = []
    for class_name in class_dirs:
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample up to 20 images per class
        sample_size = min(20, len(image_files))
        if sample_size > 0:
            import random
            samples = random.sample(image_files, sample_size)
            
            for sample in samples:
                try:
                    img_path = os.path.join(class_path, sample)
                    img = Image.open(img_path)
                    width, height = img.size
                    dimension_data.append({
                        "Class": class_name,
                        "Width": width,
                        "Height": height,
                        "Aspect Ratio": width / height if height > 0 else 0
                    })
                except Exception as e:
                    pass
    
    if dimension_data:
        # Create scatter plot of image dimensions
        import plotly.express as px
        
        fig = px.scatter(
            dimension_data,
            x="Width",
            y="Height",
            color="Class",
            hover_name="Class",
            title="Image Dimensions Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show aspect ratio distribution
        fig = px.histogram(
            dimension_data,
            x="Aspect Ratio",
            color="Class",
            marginal="rug",
            title="Aspect Ratio Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Continue with sample image display as before
    st.write("### Sample Images")
    cols = st.columns(len(class_dirs))
    
    for i, class_name in enumerate(class_dirs):
        cols[i].write(f"**{class_name}**")
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            cols[i].write("No images found")
            continue
        
        # Select random samples
        import random
        samples = random.sample(image_files, min(4, len(image_files)))
        
        for sample in samples:
            try:
                img_path = os.path.join(class_path, sample)
                img = Image.open(img_path)
                cols[i].image(img, caption=sample, use_column_width=True)
            except Exception as e:
                cols[i].error(f"Error loading image: {e}")

def visualize_model_architecture_diagram(model_name, image_size):
    """
    Create a visual representation of the model architecture
    
    Args:
        model_name: Name of the model
        image_size: Tuple of (height, width)
    """
    try:
        with st.spinner(f"Creating {model_name} architecture visualization..."):
            model = ModelFactory.create_model(
                model_name=model_name,
                input_shape=(*image_size, 3),
                num_classes=1  # Binary classification
            )
            
            # Create a diagram of the model architecture
            import tensorflow as tf
            from tensorflow.keras.utils import plot_model
            import tempfile
            import os
            
            # Create a temporary file to save the model plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name
            
            # Generate the plot
            plot_model(
                model, 
                to_file=temp_path, 
                show_shapes=True, 
                show_layer_names=True, 
                rankdir='TB'  # Top to Bottom layout
            )
            
            # Display the image
            st.image(temp_path, use_column_width=True)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
    except Exception as e:
        st.error(f"Error creating model architecture visualization: {e}")
        st.write("Falling back to text representation:")
        display_model_architecture(model_name, image_size)

def plot_confusion_matrix(confusion_matrix_data):
    """
    Create an interactive confusion matrix visualization
    
    Args:
        confusion_matrix_data: Confusion matrix as a numpy array
    
    Returns:
        Plotly figure
    """
    import plotly.figure_factory as ff
    
    # Make sure we have the class labels
    labels = ["Class " + str(i) for i in range(confusion_matrix_data.shape[0])]
    
    # Create the confusion matrix figure
    fig = ff.create_annotated_heatmap(
        z=confusion_matrix_data,
        x=labels,
        y=labels,
        colorscale='Blues',
        showscale=True
    )
    
    # Update layout
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="True Label"),
        xaxis_side="bottom"
    )
    
    # Add a note on how to read the confusion matrix
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text="True labels on the y-axis, predicted labels on the x-axis",
        showarrow=False
    )
    
    return fig

def search_experiments(experiments, query):
    """
    Search experiments based on a query string
    
    Args:
        experiments: List of experiment names
        query: Search query string
        
    Returns:
        Filtered list of experiments
    """
    if not query:
        return experiments
    
    query = query.lower()
    return [exp for exp in experiments if query in exp.lower()]

def create_metrics_overview():
    """
    Create a metrics overview for the dashboard with recent experiment performance
    """
    experiments = get_available_experiments()
    if not experiments:
        st.info("No experiments found. Start training a model in the Training page.")
        return
    
    # Get data for 5 most recent experiments
    recent_exps = experiments[:5]
    metrics_data = []
    
    for exp in recent_exps:
        exp_data = load_experiment_results(exp)
        if exp_data:
            metrics = exp_data.get("results", {})
            if metrics:
                metrics_data.append({
                    "Experiment": exp,
                    "Accuracy": metrics.get("accuracy_mean", 0),
                    "Precision": metrics.get("precision_mean", 0),
                    "Recall": metrics.get("recall_mean", 0),
                    "F1": metrics.get("f1_mean", 0),
                    "AUC": metrics.get("auc_mean", 0)
                })
    
    if metrics_data:
        # Create a metrics trend visualization
        import plotly.express as px
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            metrics_df,
            color="Accuracy",
            labels={"Experiment": "Experiment", 
                    "Accuracy": "Accuracy", 
                    "Precision": "Precision",
                    "Recall": "Recall", 
                    "F1": "F1 Score", 
                    "AUC": "AUC"},
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Recent Experiments Performance Overview"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics data found for recent experiments.")

# Function to display experiment results
def display_experiment_results(experiment_name):
    experiment_dir = Path(f"./results/{experiment_name}")
    if not experiment_dir.exists():
        st.error(f"Experiment directory not found: {experiment_dir}")
        return
    
    # Load experiment data
    experiment_data = load_experiment_results(experiment_name)
    if not experiment_data:
        st.error("Could not load experiment results")
        return
    
    # Display configuration
    st.write("### Experiment Configuration")
    config_df = pd.DataFrame.from_dict(experiment_data.get("config", {}), orient="index", columns=["Value"])
    safe_display_dataframe(config_df)
    
    # Display metrics
    st.write("### Performance Metrics")
    
    results = experiment_data.get("results", {})
    metrics_df = pd.DataFrame.from_dict(results, orient="index", columns=["Value"])
    safe_display_dataframe(metrics_df)
    
    # Create metric gauges for key metrics
    if results:
        col1, col2, col3, col4 = st.columns(4)
        
        # Helper function to create a gauge
        def create_metric_gauge(value, title, column):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": title},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 0.5], "color": "lightgray"},
                        {"range": [0.5, 0.75], "color": "gray"},
                        {"range": [0.75, 1], "color": "lightblue"}
                    ]
                }
            ))
            fig.update_layout(height=250, width=250)
            column.plotly_chart(fig, use_container_width=True)
        
        # Create gauges for key metrics
        if "accuracy_mean" in results:
            create_metric_gauge(results["accuracy_mean"], "Accuracy", col1)
        if "precision_mean" in results:
            create_metric_gauge(results["precision_mean"], "Precision", col2)
        if "recall_mean" in results:
            create_metric_gauge(results["recall_mean"], "Recall", col3)
        if "auc_mean" in results:
            create_metric_gauge(results["auc_mean"], "AUC", col4)
    
    # Display confusion matrix interactively if data available
    confusion_matrix_path = next(experiment_dir.glob("**/confusion_matrix.npy"), None)
    if confusion_matrix_path:
        st.write("### Confusion Matrix")
        try:
            confusion_matrix_data = np.load(str(confusion_matrix_path))
            fig = plot_confusion_matrix(confusion_matrix_data)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading confusion matrix: {e}")
            # Fallback to image if available
            confusion_matrix_img = next(experiment_dir.glob("**/confusion_matrix.png"), None)
            if confusion_matrix_img:
                st.image(str(confusion_matrix_img), use_column_width=True)
    
    # Display ROC curve if available
    roc_curve_path = next(experiment_dir.glob("**/roc_curve.png"), None)
    if roc_curve_path:
        st.write("### ROC Curve")
        st.image(str(roc_curve_path), use_column_width=True)
    
    # Display precision-recall curve if available
    pr_curve_path = next(experiment_dir.glob("**/precision_recall_curve.png"), None)
    if pr_curve_path:
        st.write("### Precision-Recall Curve")
        st.image(str(pr_curve_path), use_column_width=True)
    
    # Display training history plots interactively if history data available
    history_path = next(experiment_dir.glob("**/training_history.json"), None)
    if history_path:
        st.write("### Training History")
        try:
            with open(history_path, 'r') as f:
                history_data = json.load(f)
            
            # Create interactive plot
            fig = plot_training_history(history_data)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating interactive training history plot: {e}")
            # Fallback to static images
            accuracy_plot_path = next(experiment_dir.glob("**/accuracy_plot.png"), None)
            loss_plot_path = next(experiment_dir.glob("**/loss_plot.png"), None)
            
            if accuracy_plot_path or loss_plot_path:
                cols = st.columns(2)
                
                if accuracy_plot_path:
                    cols[0].image(str(accuracy_plot_path), use_column_width=True)
                
                if loss_plot_path:
                    cols[1].image(str(loss_plot_path), use_column_width=True)
    else:
        # Fallback to static images
        accuracy_plot_path = next(experiment_dir.glob("**/accuracy_plot.png"), None)
        loss_plot_path = next(experiment_dir.glob("**/loss_plot.png"), None)
        
        if accuracy_plot_path or loss_plot_path:
            st.write("### Training History")
            cols = st.columns(2)
            
            if accuracy_plot_path:
                cols[0].image(str(accuracy_plot_path), use_column_width=True)
            
            if loss_plot_path:
                cols[1].image(str(loss_plot_path), use_column_width=True)
                
# Dashboard page
if page == "Dashboard":
    st.header("Dashboard")
    
    # System information
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"TensorFlow Version: {tf.__version__}")
        st.info(f"Python Version: {sys.version.split()[0]}")
    
    with col2:
        if gpus:
            for i, gpu in enumerate(gpus):
                st.info(f"GPU {i+1}: {gpu.name}")
        else:
            st.warning("No GPUs detected. Training will use CPU only.")
    
    # Add a metrics overview chart
    st.subheader("Performance Overview")
    create_metrics_overview()
    
    # Recent experiments
    st.subheader("Recent Experiments")
    
    experiments = get_available_experiments()
    if experiments:
        # Add experiment search functionality
        search_query = st.text_input("Search experiments", "")
        filtered_experiments = search_experiments(experiments, search_query)
        
        if not filtered_experiments and search_query:
            st.warning(f"No experiments found matching '{search_query}'")
            filtered_experiments = experiments[:5]  # Show 5 most recent as fallback
        elif not filtered_experiments:
            filtered_experiments = experiments[:5]  # Show 5 most recent
        
        for exp in filtered_experiments:
            with st.expander(exp):
                exp_data = load_experiment_results(exp)
                if exp_data:
                    # Display key metrics
                    metrics = exp_data.get("results", {})
                    cols = st.columns(4)
                    if "accuracy_mean" in metrics:
                        cols[0].metric("Accuracy", f"{metrics['accuracy_mean']:.4f}")
                    if "precision_mean" in metrics:
                        cols[1].metric("Precision", f"{metrics['precision_mean']:.4f}")
                    if "recall_mean" in metrics:
                        cols[2].metric("Recall", f"{metrics['recall_mean']:.4f}")
                    if "auc_mean" in metrics:
                        cols[3].metric("AUC", f"{metrics['auc_mean']:.4f}")
                    
                    # Link to full results
                    st.button("View Full Results", key=f"view_{exp}", 
                              on_click=lambda exp=exp: st.session_state.update(
                                  {"page": "Results Viewer", "selected_experiment": exp}
                              ))
    else:
        st.info("No experiments found. Start training a model in the Training page.")
    
    # Add experiment comparison
    st.subheader("Compare Experiments")
    if experiments:
        # Multi-select for experiments to compare
        selected_experiments = st.multiselect(
            "Select experiments to compare",
            options=experiments,
            default=experiments[:2] if len(experiments) >= 2 else experiments[:1]
        )
        
        if st.button("Compare Selected Experiments") and selected_experiments:
            compare_experiments(selected_experiments)
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Configure New Experiment", use_container_width=True):
            st.session_state["page"] = "Configure Experiment"
    
    with col2:
        if st.button("View Results", use_container_width=True):
            st.session_state["page"] = "Results Viewer"

# Configure Experiment page
elif page == "Configure Experiment":
    st.header("Configure Experiment")
    
    # Data path selection
    st.subheader("Dataset")
    data_path = st.text_input("Data Directory Path", value="./data")
    
    if st.button("Explore Dataset"):
        enhanced_dataset_visualization(data_path)

    
    # Model selection
    st.subheader("Model")
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox("Model Architecture", get_available_models())
    
    with col2:
        # Use a regular selectbox instead of select_slider to avoid tuple comparison issues
        image_size_options = ["224×224", "256×256", "299×299", "331×331", "384×384", "512×512"]
        image_size_map = {
            "224×224": (224, 224),
            "256×256": (256, 256),
            "299×299": (299, 299),
            "331×331": (331, 331),
            "384×384": (384, 384),
            "512×512": (512, 512)
        }
        image_size_selection = st.selectbox(
            "Image Size",
            options=image_size_options,
            index=0
        )
        image_size = image_size_map[image_size_selection]
    
    if st.button("Show Model Architecture"):
        visualize_model_architecture_diagram(model_name, image_size)
    
    # Training parameters
    st.subheader("Training Parameters")

    st.subheader("Validation Strategy")
    validation_strategy = st.radio(
        "Validation Strategy",
        options=["cross_validation", "holdout"],
        format_func=lambda x: "Cross-Validation" if x == "cross_validation" else "Holdout (Train/Val Split)"
    )

    if validation_strategy == "cross_validation":
        k_folds = st.slider("Cross-Validation Folds", min_value=2, max_value=10, value=5)
        holdout_split = 0.2  # Default value, not shown to user
    else:  # holdout
        holdout_split = st.slider(
            "Validation Split Proportion", 
            min_value=0.1, 
            max_value=0.4, 
            value=0.2, 
            step=0.05,
            help="Proportion of training data to use for validation"
        )
        k_folds = 5  # Default value, not shown to user

    test_split = st.slider(
        "Test Split Proportion", 
        min_value=0.1, 
        max_value=0.3, 
        value=0.2, 
        step=0.05,
        help="Proportion of data to use for final testing"
    )

    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.select_slider(
            "Batch Size",
            options=[8, 16, 32, 64, 128],
            value=32
        )
        
        epochs = st.slider("Maximum Epochs", min_value=10, max_value=300, value=100, step=10)
        
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.0001,
            format_func=lambda x: f"{x:.4f}"
        )
    
    with col2:
        k_folds = st.slider("Cross-Validation Folds", min_value=2, max_value=10, value=5)
        
        early_stopping_patience = st.slider(
            "Early Stopping Patience", 
            min_value=5, 
            max_value=30, 
            value=10
        )
        
        mixed_precision = st.checkbox("Use Mixed Precision", value=True)
    
    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.7, value=0.5, step=0.1)
            reduce_lr_patience = st.slider("Reduce LR Patience", min_value=3, max_value=20, value=5)
        
        with col2:
            save_dir = st.text_input("Results Directory", value="./results")
            verbose = st.radio("Verbosity Level", options=[0, 1, 2], index=1)
    
    # Save configuration
    if st.button("Save Configuration", type="primary"):
        # Create configuration object
        config = {
            "model_name": model_name,
            "data_path": data_path,
            "image_size": list(image_size),
            "batch_size": batch_size,
            "epochs": epochs,
            "validation_strategy": validation_strategy,
            "k_folds": k_folds,
            "holdout_split": holdout_split,
            "test_split": test_split,
            "learning_rate": learning_rate,
            "early_stopping_patience": early_stopping_patience,
            "reduce_lr_patience": reduce_lr_patience,
            "mixed_precision": mixed_precision,
            "dropout_rate": dropout_rate,
            "save_dir": save_dir,
            "verbose": verbose
        }
        
        # Save to file
        os.makedirs("configs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        config_path = f"configs/config_{timestamp}.json"
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        st.success(f"Configuration saved to {config_path}")
        
        # Save in session state and navigate to training page
        st.session_state["config_path"] = config_path
        
        # Provide option to start training immediately or go to training page
        st.success(f"Configuration saved to {config_path}")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Training Now", type="primary"):
                launch_training(config_path)
        
        with col2:
            if st.button("Go to Training Page"):
                st.session_state["page"] = "Training"
                st.experimental_rerun()

# Training page
elif page == "Training":
    st.header("Model Training")
    
    # Check if config path is in session state
    config_path = st.session_state.get("config_path", None)
    
    if not config_path:
        # Let user select a config file
        configs_dir = Path("./configs")
        if configs_dir.exists():
            config_files = list(configs_dir.glob("*.json"))
            if config_files:
                config_options = [f.name for f in config_files]
                selected_config = st.selectbox(
                    "Select Configuration File", 
                    options=config_options
                )
                config_path = str(configs_dir / selected_config)
            else:
                st.error("No configuration files found. Please create one in the Configure Experiment page.")
                if st.button("Go to Configure Experiment"):
                    st.session_state["page"] = "Configure Experiment"
                    st.experimental_rerun()
        else:
            st.error("Configs directory not found. Please create a configuration in the Configure Experiment page.")
            if st.button("Go to Configure Experiment"):
                st.session_state["page"] = "Configure Experiment"
                st.experimental_rerun()
    
    # Display selected configuration
    if config_path and os.path.exists(config_path):
        st.write("### Selected Configuration")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Format the configuration as a table
        config_df = pd.DataFrame.from_dict(config, orient="index", columns=["Value"])
        safe_display_dataframe(config_df)
        
        # Launch training button
        if st.button("Start Training", type="primary"):
            launch_training(config_path)

# Results Viewer page
elif page == "Results Viewer":
    st.header("Results Viewer")
    
    # Get list of experiments
    experiments = get_available_experiments()
    
    if not experiments:
        st.info("No experiments found. Train a model first.")
    else:
        # Select experiment
        selected_experiment = st.session_state.get(
            "selected_experiment", 
            experiments[0] if experiments else None
        )
        
        selected_experiment = st.selectbox(
            "Select Experiment", 
            options=experiments,
            index=experiments.index(selected_experiment) if selected_experiment in experiments else 0
        )
        
        # Display experiment results
        display_experiment_results(selected_experiment)

# Handle page navigation from session state
if "page" in st.session_state and st.session_state["page"] != page:
    page = st.session_state["page"]
    st.experimental_rerun()


# Model Evaluation page
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    
    # Create tabs for "Run Evaluation" and "Browse Results"
    eval_tab, results_tab = st.tabs(["Run Evaluation", "Browse Evaluation Results"])
    
    with eval_tab:
        # Get list of trained models
        model_files = []
        for exp in get_available_experiments():
            exp_dir = Path(f"./results/{exp}")
            # Look for .keras or .h5 model files
            for model_file in exp_dir.glob("**/*.keras"):
                model_files.append((exp, str(model_file)))
            for model_file in exp_dir.glob("**/*.h5"):
                model_files.append((exp, str(model_file)))
        
        if not model_files:
            st.info("No trained models found. Train a model first.")
        else:
            # Select model
            model_options = [f"{exp}: {os.path.basename(path)}" for exp, path in model_files]
            selected_model_idx = st.selectbox(
                "Select Model", 
                range(len(model_options)),
                format_func=lambda i: model_options[i]
            )
            selected_experiment, selected_model_path = model_files[selected_model_idx]
        
        # Display model information
        st.write(f"**Selected Model:** {os.path.basename(selected_model_path)}")
        st.write(f"**From Experiment:** {selected_experiment}")
        
        # Load experiment config if available
        exp_config_path = os.path.join(os.path.dirname(os.path.dirname(selected_model_path)), "config.json")
        if os.path.exists(exp_config_path):
            try:
                with open(exp_config_path, 'r') as f:
                    config = json.load(f)
                st.write("**Original Training Configuration:**")
                config_df = pd.DataFrame.from_dict(config, orient="index", columns=["Value"])
                safe_display_dataframe(config_df)
            except Exception as e:
                st.warning(f"Could not load experiment configuration: {e}")
        
        # Input for evaluation dataset
        st.subheader("Evaluation Dataset")
        eval_data_path = st.text_input("Evaluation Data Directory Path", value="./eval_data")
        
        if st.button("Explore Evaluation Dataset"):
            enhanced_dataset_visualization(eval_data_path)
        
        # Evaluation options
        st.subheader("Evaluation Options")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.select_slider(
                "Batch Size",
                options=[8, 16, 32, 64, 128],
                value=32
            )
        
        with col2:
            # Use same image size selection logic as in Configure Experiment
            image_size_options = ["224×224", "256×256", "299×299", "331×331", "384×384", "512×512"]
            image_size_map = {
                "224×224": (224, 224),
                "256×256": (256, 256),
                "299×299": (299, 299),
                "331×331": (331, 331),
                "384×384": (384, 384),
                "512×512": (512, 512)
            }
            image_size_selection = st.selectbox(
                "Image Size",
                options=image_size_options,
                index=0
            )
            image_size = image_size_map[image_size_selection]
        
        # Evaluation name and output directory
        eval_name = st.text_input(
            "Evaluation Name (Optional)",
            value="",
            help="Custom name for this evaluation run (leave empty for auto-generated name)"
        )
        
        output_base_dir = st.text_input(
            "Base Results Directory", 
            value="./evaluation_results"
        )
        
        # Run evaluation button
        if st.button("Run Evaluation", type="primary"):
            # Build command
            cmd = [
                sys.executable, 
                "evaluate.py", 
                "--model-path", selected_model_path,
                "--data-path", eval_data_path,
                "--output-dir", output_base_dir,
                "--batch-size", str(batch_size),
                "--image-size", str(image_size[0]), str(image_size[1])
            ]
            
            # Add name if provided
            if eval_name:
                cmd.extend(["--name", eval_name])
            
            # Check for class mapping file
            class_mapping_path = os.path.join(os.path.dirname(os.path.dirname(selected_model_path)), "class_mapping.txt")
            if os.path.exists(class_mapping_path):
                cmd.extend(["--class-mapping", class_mapping_path])
            
            # Add model config if available
            if os.path.exists(exp_config_path):
                cmd.extend(["--model-config", exp_config_path])
            
            # Display the command
            st.code(" ".join(cmd), language="bash")
            
            # Run evaluation
            with st.spinner("Evaluating model..."):
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Create an empty placeholder for the log
                log_output = st.empty()
                
                # Stream the output
                full_log = ""
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        full_log += output
                        log_output.code(full_log, language="bash")
                
                # Get return code
                return_code = process.poll()
                
                if return_code == 0:
                    st.success("Evaluation completed successfully!")
                    
                    # Find the evaluation directory
                    # Parse output to find the actual evaluation directory that was created
                    eval_output_dir = None
                    for line in full_log.split('\n'):
                        if "Evaluating model" in line:
                            # Look for the previous line which might have the output directory
                            log_lines = full_log.split('\n')
                            for i, log_line in enumerate(log_lines):
                                if "Evaluating model" in log_line and i > 0:
                                    # Check a few lines before for directory creation
                                    for j in range(max(0, i-5), i):
                                        if output_base_dir in log_lines[j]:
                                            # Extract the path - this is a rough heuristic
                                            parts = log_lines[j].split(output_base_dir)
                                            if len(parts) > 1:
                                                subdir = parts[1].strip()
                                                if subdir.startswith('/') or subdir.startswith('\\'):
                                                    subdir = subdir[1:]
                                                eval_output_dir = os.path.join(output_base_dir, subdir)
                                                break
                    
                    # If we couldn't extract from logs, look for the most recent directory
                    if not eval_output_dir:
                        try:
                            base_dir = Path(output_base_dir)
                            if base_dir.exists():
                                # Get all subdirectories and sort by creation time (newest first)
                                subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
                                if subdirs:
                                    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                                    eval_output_dir = str(subdirs[0])
                        except Exception as e:
                            st.warning(f"Could not determine evaluation directory: {e}")
                            eval_output_dir = output_base_dir
                    
                    if not eval_output_dir:
                        eval_output_dir = output_base_dir
                    
                    st.info(f"Evaluation results saved to: {eval_output_dir}")
                    
                    # Display results if available
                    results_path = os.path.join(eval_output_dir, "evaluation_metrics.json")
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                        
                        st.subheader("Evaluation Results")
                        
                        # Display metrics as gauges
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Helper function to create a gauge (same as in results viewer)
                        def create_metric_gauge(value, title, column):
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=value,
                                title={"text": title},
                                gauge={
                                    "axis": {"range": [0, 1]},
                                    "bar": {"color": "darkblue"},
                                    "steps": [
                                        {"range": [0, 0.5], "color": "lightgray"},
                                        {"range": [0.5, 0.75], "color": "gray"},
                                        {"range": [0.75, 1], "color": "lightblue"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=250, width=250)
                            column.plotly_chart(fig, use_container_width=True)
                        
                        # Create gauges for key metrics
                        if "accuracy" in results:
                            create_metric_gauge(results["accuracy"], "Accuracy", col1)
                        if "precision" in results:
                            create_metric_gauge(results["precision"], "Precision", col2)
                        if "recall" in results:
                            create_metric_gauge(results["recall"], "Recall", col3)
                        if "auc" in results:
                            create_metric_gauge(results["auc"], "AUC", col4)
                        
                        # Show full metrics
                        st.write("**Full Metrics:**")
                        metrics_df = pd.DataFrame.from_dict(results, orient="index", columns=["Value"])
                        safe_display_dataframe(metrics_df)
                        
                        # Show confusion matrix if available
                        confusion_matrix_img = os.path.join(eval_output_dir, "confusion_matrix.png")
                        if os.path.exists(confusion_matrix_img):
                            st.write("**Confusion Matrix:**")
                            st.image(confusion_matrix_img, use_column_width=True)
                            
                        # Show ROC curve if available
                        roc_curve_img = os.path.join(eval_output_dir, "roc_curve.png")
                        if os.path.exists(roc_curve_img):
                            st.write("**ROC Curve:**")
                            st.image(roc_curve_img, use_column_width=True)
                            
                        # Show precision-recall curve if available
                        pr_curve_img = os.path.join(eval_output_dir, "precision_recall_curve.png")
                        if os.path.exists(pr_curve_img):
                            st.write("**Precision-Recall Curve:**")
                            st.image(pr_curve_img, use_column_width=True)
                    else:
                        st.warning("Evaluation results file not found")
                else:
                    error = process.stderr.read()
                    st.error(f"Evaluation failed with return code {return_code}. Error: {error}")
    
    # Browse evaluation results tab
    with results_tab:
        st.subheader("Browse Evaluation Results")
        
        # Function to get all evaluation directories
        @st.cache_data(ttl=60)  # Cache for 1 minute
        def get_evaluation_results(base_dir="./evaluation_results"):
            if not os.path.exists(base_dir):
                return []
                
            results = []
            base_path = Path(base_dir)
            
            # Get all subdirectories
            for subdir in base_path.glob("*"):
                if subdir.is_dir():
                    # Check if it contains evaluation metrics
                    metrics_file = subdir / "evaluation_metrics.json"
                    if metrics_file.exists():
                        # Get creation time and format as string
                        created = time.strftime(
                            "%Y-%m-%d %H:%M:%S", 
                            time.localtime(os.path.getmtime(subdir))
                        )
                        
                        # Get model name if available
                        model_name = "Unknown"
                        params_file = subdir / "evaluation_params.json"
                        if params_file.exists():
                            try:
                                with open(params_file, 'r') as f:
                                    params = json.load(f)
                                    if "model_path" in params:
                                        model_name = os.path.basename(os.path.dirname(os.path.dirname(params["model_path"])))
                            except:
                                pass
                        
                        results.append({
                            "name": subdir.name,
                            "path": str(subdir),
                            "created": created,
                            "model": model_name
                        })
            
            # Sort by creation time (newest first)
            results.sort(key=lambda x: x["created"], reverse=True)
            return results
        
        # Get list of evaluation results
        eval_base_dir = st.text_input("Evaluation Results Directory", value="./evaluation_results")
        if st.button("Refresh Evaluation Results"):
            st.cache_data.clear()
        
        evaluation_results = get_evaluation_results(eval_base_dir)
        
        if not evaluation_results:
            st.info(f"No evaluation results found in {eval_base_dir}. Run an evaluation first.")
        else:
            # Show evaluation results in a searchable table
            st.write(f"Found {len(evaluation_results)} evaluation results:")
            
            # Create a DataFrame for display
            results_df = pd.DataFrame(evaluation_results)
            
            # Add search functionality
            search_term = st.text_input("Search Evaluations", value="")
            if search_term:
                results_df = results_df[
                    results_df["name"].str.contains(search_term, case=False) |
                    results_df["model"].str.contains(search_term, case=False)
                ]
            
            # Display results table
            st.dataframe(results_df[["name", "model", "created"]], use_container_width=True)
            
            # Select evaluation to view
            selected_eval = st.selectbox(
                "Select Evaluation to View",
                options=range(len(results_df)),
                format_func=lambda i: f"{results_df.iloc[i]['name']} ({results_df.iloc[i]['created']})"
            )
            
            # Display selected evaluation results
            if st.button("View Selected Evaluation"):
                eval_path = results_df.iloc[selected_eval]["path"]
                st.write(f"**Viewing evaluation: {results_df.iloc[selected_eval]['name']}**")
                
                # Load and display metrics
                metrics_file = os.path.join(eval_path, "evaluation_metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Display metrics as gauges
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Helper function to create a gauge
                    def create_metric_gauge(value, title, column):
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=value,
                            title={"text": title},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": "darkblue"},
                                "steps": [
                                    {"range": [0, 0.5], "color": "lightgray"},
                                    {"range": [0.5, 0.75], "color": "gray"},
                                    {"range": [0.75, 1], "color": "lightblue"}
                                ]
                            }
                        ))
                        fig.update_layout(height=250, width=250)
                        column.plotly_chart(fig, use_container_width=True)
                    
                    # Create gauges for key metrics
                    if "accuracy" in metrics:
                        create_metric_gauge(metrics["accuracy"], "Accuracy", col1)
                    if "precision" in metrics:
                        create_metric_gauge(metrics["precision"], "Precision", col2)
                    if "recall" in metrics:
                        create_metric_gauge(metrics["recall"], "Recall", col3)
                    if "auc" in metrics:
                        create_metric_gauge(metrics["auc"], "AUC", col4)
                    
                    # Show full metrics
                    st.write("**Full Metrics:**")
                    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
                    safe_display_dataframe(metrics_df)
                
                # Display visualizations
                st.write("**Visualizations:**")
                viz_col1, viz_col2 = st.columns(2)
                
                # Confusion matrix
                confusion_matrix_img = os.path.join(eval_path, "confusion_matrix.png")
                if os.path.exists(confusion_matrix_img):
                    viz_col1.write("**Confusion Matrix:**")
                    viz_col1.image(confusion_matrix_img, use_column_width=True)
                
                # ROC curve
                roc_curve_img = os.path.join(eval_path, "roc_curve.png")
                if os.path.exists(roc_curve_img):
                    viz_col2.write("**ROC Curve:**")
                    viz_col2.image(roc_curve_img, use_column_width=True)
                
                # Precision-recall curve
                pr_curve_img = os.path.join(eval_path, "precision_recall_curve.png")
                if os.path.exists(pr_curve_img):
                    viz_col1.write("**Precision-Recall Curve:**")
                    viz_col1.image(pr_curve_img, use_column_width=True)
                
                # Display parameters used for this evaluation
                params_file = os.path.join(eval_path, "evaluation_params.json")
                if os.path.exists(params_file):
                    st.write("**Evaluation Parameters:**")
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                    params_df = pd.DataFrame.from_dict(params, orient="index", columns=["Value"])
                    safe_display_dataframe(params_df)