import torch
import torch.nn as nn
import torch.nn.functional as F

class router(nn.Module):
    def __init__(self, dim, channel_num, t):
        super().__init__()
        self.l1 = nn.Linear(dim, int(dim/8))
        self.l2 = nn.Linear(int(dim/8), channel_num)
        self.t = t
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.l2(F.relu(F.normalize(self.l1(x), p=2, dim=1)))/self.t
        output = torch.softmax(x, dim=1)
        return output