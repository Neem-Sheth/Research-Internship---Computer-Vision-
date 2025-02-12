import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import seaborn as sns

def evaluate_model(model, test_dataset, device):
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    
    model.eval()
    test_loss = 0.0
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item() * features.size(0)
            y_pred.extend(outputs.squeeze().cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Calculate various metrics
    abs_errors = np.abs(y_pred - y_true)
    rel_errors = abs_errors / y_true * 100  # percentage
    
    metrics = {
        'mae': np.mean(abs_errors),
        'min_error': np.min(abs_errors),
        'max_error': np.max(abs_errors),
        'mean_rel_error': np.mean(rel_errors),
        'rmse': np.sqrt(np.mean((y_pred - y_true) ** 2))
    }
    
    print("\nModel Performance Metrics:")
    print(f"Mean Absolute Error: {metrics['mae']:.2f} meters")
    print(f"Minimum Error: {metrics['min_error']:.2f} meters")
    print(f"Maximum Error: {metrics['max_error']:.2f} meters")
    print(f"Mean Relative Error: {metrics['mean_rel_error']:.2f}%")
    print(f"Root Mean Square Error: {metrics['rmse']:.2f} meters")

    return y_pred, metrics

def plot_results(y_true, y_pred):
    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # 1. Scatter plot with perfect prediction line
    ax1.scatter(y_true, y_pred, alpha=0.5, c='blue', label='Predictions')
    ax1.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Perfect Prediction')
    ax1.set_xlabel('Actual Distance (meters)')
    ax1.set_ylabel('Predicted Distance (meters)')
    ax1.set_title('Actual vs Predicted Distance')
    ax1.legend()
    
    # 2. Error Distribution
    errors = y_pred - y_true
    sns.histplot(errors, kde=True, ax=ax2)
    ax2.set_xlabel('Prediction Error (meters)')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution')
    
    # 3. Relative Error vs Actual Distance
    rel_errors = np.abs(errors) / y_true * 100
    ax3.scatter(y_true, rel_errors, alpha=0.5)
    ax3.set_xlabel('Actual Distance (meters)')
    ax3.set_ylabel('Relative Error (%)')
    ax3.set_title('Relative Error vs Actual Distance')
    
    # 4. Error vs Predicted Distance
    ax4.scatter(y_pred, np.abs(errors), alpha=0.5)
    ax4.set_xlabel('Predicted Distance (meters)')
    ax4.set_ylabel('Absolute Error (meters)')
    ax4.set_title('Error vs Predicted Distance')
    
    plt.tight_layout()
    plt.show()
    
    # Create training progress plot if data is available
    if hasattr(plt, 'training_history'):
        plt.figure(figsize=(10, 5))
        plt.plot(plt.training_history['train_loss'], label='Training Loss')
        plt.plot(plt.training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.show()
