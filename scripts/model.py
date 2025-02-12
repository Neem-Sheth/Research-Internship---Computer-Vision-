import torch.nn as nn
import torch

# class DistancePredictor(nn.Module): # 0.91
#     def __init__(self):
#         super(DistancePredictor, self).__init__()
#         self.fc1 = nn.Linear(6, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 1)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

class DistancePredictor(nn.Module): # 0.67
    def __init__(self):
        super(DistancePredictor, self).__init__()
        self.fc1 = nn.Linear(6, 64, True)
        self.fc2 = nn.Linear(64, 64, True)
        self.fc3 = nn.Linear(64, 1, True)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x): 
        x = torch.selu(self.fc1(x))
        x = torch.selu(self.fc2(x))
        x = self.fc3(x)
        return x

# import torch.nn as nn
# import torch

# class DistancePredictor(nn.Module):
#     def __init__(self, hidden_sizes=[64, 64], activation='relu', dropout_rate=0.2):
#         super(DistancePredictor, self).__init__()
        
#         self.activation_dict = {
#             'relu': nn.ReLU(),
#             'leaky_relu': nn.LeakyReLU(),
#             'selu': nn.SELU(),
#             'elu': nn.ELU(),
#             'silu': nn.SiLU(),
#             'tanh': nn.Tanh()
#         }
        
#         self.activation = self.activation_dict[activation]
#         layers = []
        
#         # Input layer
#         layers.append(nn.Linear(6, hidden_sizes[0]))
#         layers.append(self.activation)
#         layers.append(nn.Dropout(dropout_rate))
        
#         # Hidden layers
#         for i in range(len(hidden_sizes)-1):
#             layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
#             layers.append(self.activation)
#             layers.append(nn.Dropout(dropout_rate))
        
#         # Output layer(x))
#         layers.append(nn.Linear(hidden_sizes[-1], 1))
        
#         self.model = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.model(x)
