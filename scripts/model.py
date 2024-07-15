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

class DistancePredictor(nn.Module): # 0.71
    def __init__(self):
        super(DistancePredictor, self).__init__()
        self.fc1 = nn.Linear(6, 64, True)
        self.fc2 = nn.Linear(64, 64, True)
        self.fc3 = nn.Linear(64, 1, True)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x): 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
