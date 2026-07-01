from main.dataset_build import get_dataset
from code.model import SignLanguageCNN
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch

output_dir = "models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def train_and_save(dataset, train_threshold, batch_size, learning_rate, epochs, device, output_path):
    
    train, test = get_dataset(dataset, train_threshold)
    num_classes = len(train.label_to_idx)

    train_loader = DataLoader(
            train,
            batch_size,
            shuffle=True,
    )

    test_loader = DataLoader(
            test,
            batch_size,
            shuffle=False,
    )

    model = SignLanguageCNN(num_classes).to(device)

    history = model.train_sgd(
        train_loader,
        test_loader,
        epochs,
        learning_rate,
        device,
    )
    
    output_path = os.path.join(output_dir, output_path)
    torch.save(model, output_path)
    print(f"Model saved at {output_path}")
    
    return history




















# # pytorch
# import torch
# import torch.nn as nn
# 
# from torchvision import transforms, datasets

# # math utils
# import numpy as np
# import matplotlib.pyplot as plt

# # kaggle (for downloading datasets) and dataset manage
# import kagglehub as kh
# import os


# # dataset download
# lis_path = kh.dataset_download("nicholasnisopoli/lisdataset")
# asl_path = kh.dataset_download("grassknoted/asl-alphabet")
# print("Italian Sing Language dataset path:", lis_path)
# print("American Sign Language dataset path:", asl_path)
# lis_folder = "LIS-fingerspelling-dataset"
# asl_folder = "asl_alphabet_train/asl_alphabet_train" 
# dataset = os.path.join(lis_path, lis_folder)
# asl_dataset = os.path.join(asl_path, asl_folder)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# def evaluate(model, loader, criterion, device):
#         model.eval()
#         total_loss = 0.0
#         total_correct = 0
#         total_samples = 0

#         with torch.no_grad():
#             for images, labels in loader:
#                 images = images.to(device)
#                 labels = labels.to(device)

#                 logits = model(images)
#                 loss = criterion(logits, labels)

#                 total_loss += loss.item() * images.size(0)
#                 preds = logits.argmax(dim=1)
#                 total_correct += (preds == labels).sum().item()
#                 total_samples += images.size(0)

#         avg_loss = total_loss / total_samples
#         accuracy = total_correct / total_samples
#         return avg_loss, accuracy

# def train(model, train_loader, test_loader, epochs, lr, device):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
#     history = {
#         "train_loss": [],
#         "train_acc": [],
#         "test_loss": [],
#         "test_acc": [],
#     }
    
#     for epoch in range(1, epochs + 1):
#         model.train()

#         running_loss = 0.0
#         running_correct = 0
#         running_samples = 0

#         for images, labels in train_loader:
#             images = images.to(device)
#             labels = labels.to(device)

#             logits = model(images)
#             loss = criterion(logits, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * images.size(0)
#             preds = logits.argmax(dim=1)
#             running_correct += (preds == labels).sum().item()
#             running_samples += images.size(0)

#         train_loss = running_loss / running_samples
#         train_acc = running_correct / running_samples
#         test_loss, test_acc = evaluate(model, test_loader, criterion, device)

#         history["train_loss"].append(train_loss)
#         history["train_acc"].append(train_acc)
#         history["test_loss"].append(test_loss)
#         history["test_acc"].append(test_acc)

#         print(
#             f"Epoch {epoch:02d}/{epochs} | "
#             f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
#             f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
#         )

#     return history

# from dataset_build import get_dataset

# THRESH = 0.85
# EPOCHS = 25
# BATCH_SIZE = 32
# LEARNING_RATE = 0.01

# asl_train, asl_test = get_dataset(asl_dataset, THRESH)
# asl_classes = asl_train.label_to_idx
# # print(lis_test)


# train_loader = DataLoader(
#         asl_train,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
# )

# test_loader = DataLoader(
#         asl_test,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
# )

# asl_model = SignLanguageCNN(len(asl_classes)).to(device)



# 
# output_dir = Path("outputs_cnn_tutorial")

# history = train(
#     model=asl_model,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     epochs=EPOCHS,
#     lr=LEARNING_RATE,
#     device=device,
# )


# torch.save(asl_model, 'asl_model.pth')
