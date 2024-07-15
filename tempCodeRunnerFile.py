device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# Initialize and train model
model = DistancePredictor().to(device)
# train_model(model, train_dataset, val_dataset, device)

# Load the best model
model.load_state_dict(torch.load('best_model_1.pth'))

# Evaluate model
y_pred, test_loss = evaluate_model(model, test_dataset, device)

# Plot results
plot_results(y_test, y_pred)