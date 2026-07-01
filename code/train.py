from dataset_build import get_dataset
from torch import nn
from model import SignLanguageCNN
import os
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from ResNet import Bottleneck, ResNet, ResNet50

output_dir = "models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

def training(model, train_loader, test_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    
    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
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
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

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


def train_and_save(dataset, model_class, config, device, output_path=None):
    
    img_size = (64,64)
    
    if model_class == ResNet50:
        img_size = (224, 224)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    target_transform=transforms.Lambda(
        lambda y: torch.tensor(y)
    )
    
    train, test = get_dataset(dataset, config.train_threshold, transform, target_transform)
    num_classes = len(train.label_to_idx)

    train_loader = DataLoader(
            train,
            config.batch_size,
            shuffle=True,
    )

    test_loader = DataLoader(
            test,
            config.batch_size,
            shuffle=False,
    )

    model = model_class(num_classes).to(device)

    history = training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=config.epochs,
        lr=config.learning_rate,
        device=device
    )
    
    if output_path:
        output_path = os.path.join(output_dir, output_path)
        torch.save(model, output_path)
        print(f"Model saved at {output_path}")
    
    return history


