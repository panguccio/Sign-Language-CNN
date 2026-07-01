from model import SignLanguageCNN
import torch
import cv2
import torchvision.transforms as transforms
import os

model_dir = "code/models"

path_a = "/Users/anna/.cache/kagglehub/datasets/nicholasnisopoli/lisdataset/versions/1/LIS-fingerspelling-dataset/a/a_FATIMA_9_front.JPG"
path_michi = "img.jpg"

import torch.nn.functional as F

asl_classes = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

asl_model = SignLanguageCNN(len(asl_classes))
asl_model = torch.load(os.path.join(model_dir, "asl_model.pth"), weights_only=False)
asl_model.eval()

cropped = cv2.imread(path_michi)
cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
cropped = cv2.resize(cropped, (64, 64))
transform = transforms.ToTensor()
input = transform(cropped)
input = input.unsqueeze(0)


with torch.no_grad():
    logits = asl_model(input)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

output = int(predicted_class)
print(output)
print(asl_classes[output])

from camera import Camera

cam = Camera(asl_model, asl_classes)
cam.start()