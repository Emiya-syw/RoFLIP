import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingLayer(nn.Module):
    def __init__(self, in_channel=768, out_channel=1152):
        super(MappingLayer, self).__init__()
        self.fc = nn.Linear(in_features=in_channel, out_features=out_channel, bias=False)
    
    def forward(self, x):
        return self.fc(x)
    
class RobustProjector(nn.Module):
    def __init__(self, in_channel=768, out_channel=1152, hidden_dim=1152):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channel),
        )

    def forward(self, x):
        x = self.net(x)
        return x