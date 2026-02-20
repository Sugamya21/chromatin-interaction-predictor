
import torch
import torch.nn as nn

class ChromatinCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        with torch.no_grad():
            dummy = torch.zeros(1,4,4000)
            dummy_out = self.conv(dummy)
            self.flat_dim = dummy_out.view(1,-1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim*2,128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,1)
        )

    def forward(self,x1,x2):
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        x1 = torch.flatten(x1,1)
        x2 = torch.flatten(x2,1)
        x = torch.cat((x1,x2),dim=1)
        return self.fc(x)
