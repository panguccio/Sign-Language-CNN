from model import SignLanguageCNN
import torch
import cv2
import torchvision.transforms as transforms
import os
from camera import Camera
import torch.nn.functional as F

model_dir = "models"


def to_evaluate(model_class, model_path: str, idx_to_class: int):
    model = model_class(len(idx_to_class))
    model = torch.load(os.path.join(model_dir, model_path), weights_only=False)
    model.eval()
    return model

def test_picture(model_class, model_path: str, idx_to_class: int, img_path: str):
    model = to_evaluate(model_class, model_path, idx_to_class)
    cropped = cv2.imread(img_path)
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped = cv2.resize(cropped, (64, 64))
    transform = transforms.ToTensor()
    input = transform(cropped)
    input = input.unsqueeze(0)

    with torch.no_grad():
        logits = model(input)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    output = int(predicted_class)
    print(output)
    print(idx_to_class[output])
    

def test_camera(model_class, model_path: str, idx_to_class: int):
    model = to_evaluate(model_class, model_path, idx_to_class)
    cam = Camera(model, idx_to_class)
    cam.start()
    pass
