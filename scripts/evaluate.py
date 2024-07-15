import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

def evaluate_model(model, test_dataset, device):
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    
    model.eval()
    test_loss = 0.0
    y_pred = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item() * features.size(0)
            y_pred.extend(outputs.squeeze().tolist())
    
    test_loss /= len(test_loader.dataset)
    print(f'Test MAE: {test_loss:.2f} meters')

    return y_pred, test_loss

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Distance (meters)')
    plt.ylabel('Predicted Distance (meters)')
    plt.title('Actual vs Predicted Distance')
    plt.show()
