import torch
import torch.nn as nn


class SingleTaskMLP(nn.Module):
    def __init__(self, hidden_unit = 128, class_type = 1):
        '''
        type : 1 (classify 0 & 1), 2 (classify 0 & 2)
        '''
        super().__init__()
        
        self.in_dim = 16 # Feature dimension
        self.out_dim = 1
        
        self.fc1 = nn.Linear(self.in_dim, hidden_unit)
        self.fc2 = nn.Linear(hidden_unit, self.out_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = x.reshape(-1, self.in_dim)
        a = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(a))
        return out