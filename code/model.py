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
    
    def evaluate(self, loader, criterion, device):
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = self(images)
                loss = criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def train_sgd(self, train_loader, test_loader, epochs, lr, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        
        history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }
        
        for epoch in range(1, epochs + 1):
            self.train()

            running_loss = 0.0
            running_correct = 0
            running_samples = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = self(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_samples += images.size(0)

            train_loss = running_loss / running_samples
            train_acc = running_correct / running_samples
            test_loss, test_acc = self.evaluate(test_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
            )

        return history