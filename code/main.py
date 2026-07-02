from testing import test_camera
from ResNet import ResNet50
lis_idx_to_string = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'h', 7: 'i', 8: 'k', 9: 'l', 10: 'm', 11: 'n', 12: 'o', 13: 'p', 14: 'q', 15: 'r', 16: 't', 17: 'u', 18: 'v', 19: 'w', 20: 'x', 21: 'y'}
test_camera(ResNet50, "resnet_lis_model", lis_idx_to_string)


# # pytorch
# import torch

# # kaggle (for downloading datasets) and dataset manage
# import kagglehub as kh
# import os

# # dataset download
# lis_path = kh.dataset_download("nicholasnisopoli/lisdataset")
# asl_path = kh.dataset_download("grassknoted/asl-alphabet")
# print("Italian Sing Language dataset path:", lis_path)
# print("American Sign Language dataset path:", asl_path)
# lis_dataset = os.path.join(lis_path, "LIS-fingerspelling-dataset")
# asl_dataset = os.path.join(asl_path, "asl_alphabet_train/asl_alphabet_train")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# from train import train_and_save
# from plotter import plot_loss, plot_acc

# from dataclasses import dataclass

# @dataclass
# class TrainConfig:
#     train_threshold: float = 0.85
#     epochs: int = 25
#     batch_size: int = 32
#     learning_rate: float = 0.01
    
    
# config = TrainConfig()


# from model import SignLanguageCNN
# lis_history = train_and_save(dataset=lis_dataset, model_class=SignLanguageCNN, config=config, device=device, output_path="lis_model.pth")