# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch import nn
# import torch

# def train_model(model, train_dataset, val_dataset, device, num_epochs=1000, learning_rate=0.0001):
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     best_val_loss = float('inf')
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for features, labels in train_loader:
#             features, labels = features.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(features)
#             loss = criterion(outputs.squeeze(), labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * features.size(0)
        
#         epoch_loss = running_loss / len(train_loader.dataset)
        
#         # Validation
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for features, labels in val_loader:
#                 features, labels = features.to(device), labels.to(device)
#                 outputs = model(features)
#                 loss = criterion(outputs.squeeze(), labels)
#                 val_loss += loss.item() * features.size(0)
        
#         val_loss /= len(val_loader.dataset)
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), 'best_model_5.pth')
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # from torch import nn
# # import torch

# # def train_model(model, train_dataset, val_dataset, device, num_epochs=1000, learning_rate=0.0001, batch_size=16):
# #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
# #     criterion = nn.MSELoss()
# #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
# #     best_val_loss = float('inf')
# #     patience = 10
# #     patience_counter = 0
    
# #     for epoch in range(num_epochs):
# #         model.train()
# #         running_loss = 0.0
# #         for features, labels in train_loader:
# #             features, labels = features.to(device), labels.to(device)
# #             optimizer.zero_grad()
# #             outputs = model(features)
# #             loss = criterion(outputs.squeeze(), labels)
# #             loss.backward()
# #             optimizer.step()
# #             running_loss += loss.item() * features.size(0)
        
# #         epoch_loss = running_loss / len(train_loader.dataset)
        
# #         # Validation
# #         model.eval()
# #         val_loss = 0.0
# #         with torch.no_grad():
# #             for features, labels in val_loader:
# #                 features, labels = features.to(device), labels.to(device)
# #                 outputs = model(features)
# #                 loss = criterion(outputs.squeeze(), labels)
# #                 val_loss += loss.item() * features.size(0)
        
# #         val_loss /= len(val_loader.dataset)
        
# #         if val_loss < best_val_loss:
# #             best_val_loss = val_loss
# #             patience_counter = 0
# #             torch.save(model.state_dict(), 'best_model_3.pth')
# #         else:
# #             patience_counter += 1
            
# #         if patience_counter >= patience:
# #             print(f"Early stopping at epoch {epoch}")
# #             break
            
# #         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
    
# #     return best_val_loss


import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import torch

def train_model(model, train_dataset, val_dataset, device, num_epochs=1000, learning_rate=0.0001):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    # Add history tracking
    training_history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * features.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        training_history['train_loss'].append(epoch_loss)
        training_history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'training_history': training_history
            }, 'best_model_5.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Store history for plotting
    import matplotlib.pyplot as plt
    plt.training_history = training_history
    
    return best_val_loss

