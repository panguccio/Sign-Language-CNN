from torch import nn
import torch

class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # first block
        # input = [b, 3, 64, 64]
        # conv1 = [b, 32, 64, 64]
        # pool1 = [b, 32, 32, 32]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # we try padding = 1, but could be 0
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # second block
        # conv2 = [b, 32, 32, 32]
        # pool2 = [b, 32, 16, 16]
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # third block
        # conv3 = [b, 64, 16, 16]
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        # flatten [b, 64*16*16]
        self.fc1 = nn.Linear(in_features=64*16*16, out_features=256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
        
        
    def forward(self, x, return_activations=False):
        activations = {}
        
        x = self.conv1(x)
        activations['conv1'] = x
        
        x = self.relu1(x)
        activations['relu1'] = x
        
        x = self.pool1(x)
        activations['pool1'] = x
        
        x = self.conv2(x)
        activations['conv2'] = x
        
        x = self.relu2(x)
        activations['relu2'] = x
        
        x = self.pool2(x)
        activations['pool2'] = x
        
        x = self.conv3(x)
        activations['conv3'] = x
        
        x = self.relu3(x)
        activations['relu3'] = x
        
        x = torch.flatten(x, start_dim=1)
        activations['flatten'] = x
        
        x = self.fc1(x)
        activations['fc1'] = x
        
        x = self.relu4(x)
        activations['relu4'] = x
        
        symbols = self.fc2(x)
        activations['symbols'] = symbols
        
        if return_activations:
            return symbols, activations
        return symbols