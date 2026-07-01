from model import SignLanguageCNN
import torch
import cv2
import torchvision.transforms as transforms
import os

path_a = "/Users/anna/.cache/kagglehub/datasets/nicholasnisopoli/lisdataset/versions/1/LIS-fingerspelling-dataset/a/a_FATIMA_9_front.JPG"
path_michi = "img.jpg"

model_dir = "code/models"

import torch.nn.functional as F

lis_classes = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'h', 7: 'i', 8: 'k', 9: 'l', 10: 'm', 11: 'n', 12: 'o', 13: 'p', 14: 'q', 15: 'r', 16: 't', 17: 'u', 18: 'v', 19: 'w', 20: 'x', 21: 'y'}

lis_model = SignLanguageCNN(len(lis_classes))
asl_model = torch.load(os.path.join(model_dir, "asl_model.pth"), weights_only=False)
lis_model.eval()


cropped = cv2.imread(path_michi)
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
cropped = cv2.resize(cropped, (64, 64))
transform = transforms.ToTensor()
input = transform(cropped)
input = input.unsqueeze(0)

with torch.no_grad():
    logits = lis_model(input)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

output = int(predicted_class)
print(output)
print(lis_classes[output])

with torch.no_grad():
    logits = lis_model(input)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

output = int(predicted_class)
print(output)
print(lis_classes[output])

from camera import Camera

cam = Camera(lis_model, lis_classes)
cam.start()