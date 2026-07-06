import os
import torch
from representation import show_feature_maps, show_representation_summaries
import torchvision.transforms as transforms
from dataset_build import get_dataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, vgg16, mobilenet_v2


# dir for the models
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


def train_and_save(dataset, model_class, config, device, output_path=None, save_repr=False):
    
    # default image resize for our custom nn
    img_size = (64,64)
    
    # if resnet, change the value of the image resize
    if model_class == resnet50 or model_class == vgg16 or model_class == mobilenet_v2:
        img_size = (224, 224)
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    target_transform = transforms.Lambda(
        lambda y: torch.tensor(y)
    )
    
    train, test = get_dataset(dataset, config.train_threshold, train_transform, test_transform, target_transform)
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

    if model_class == resnet50:
        model = resnet50(weights='DEFAULT')
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_class == vgg16:
        model = vgg16(weights='DEFAULT')
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_class == mobilenet_v2:
        model = mobilenet_v2(weights='DEFAULT')
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        model = model_class(num_classes)

    model = model.to(device)

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
        
    if save_repr:
        show_feature_maps(model, test_loader, train.idx_to_label, device)
    
    return history


