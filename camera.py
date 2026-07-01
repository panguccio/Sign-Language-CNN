import cv2
from model import SignLanguageCNN
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch

import torchvision.transforms as transforms
import torch.nn.functional as F


class Camera():
    
    def __init__(self, model, idx_to_label):
        self.model = model
        self.idx_to_label = idx_to_label
        
        
    def start(self):
        cam = cv2.VideoCapture(0)

        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)
        # set save=True to save the first frame
        save = False
        while True:
            success, img = cam.read()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            detection_result = detector.detect(mp_image)

            # Debug: controlla se sono state trovate mani
            num_hands = len(detection_result.hand_landmarks)
            
            max_x, max_y = 0, 0
            min_x, min_y = frame_width, frame_height
            
            # Disegna i landmark sull'immagine
            for hand_landmarks in detection_result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y

            padding = 80
            
            if not (max_x == 0 or max_y == 0 or min_x == frame_width or min_y == frame_height):    
                
                # find the center of the hand
                mean_x = (max_x + min_x)/2
                mean_y = (max_y + min_y)/2
                
                hand_width = max_x - min_x
                hand_height = max_y - min_y
                side_length = max(hand_width, hand_height) + padding
                square_padding = side_length/2
                
                max_x, max_y = int(mean_x + square_padding), int(mean_y + square_padding)
                min_x, min_y = int(mean_x - square_padding), int(mean_y - square_padding)
                
                if min_x < 0:
                    min_x = 0
                if min_y < 0:
                    min_y = 0
                if max_x > frame_width:
                    max_x = frame_width
                if max_y > frame_height:
                    max_y = frame_height
                
                
                cropped = img[min_y:max_y, min_x:max_x]
                if save:
                    cv2.imwrite("img.jpg", cropped)
                    save = False
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                cropped = cv2.resize(cropped, (64, 64))
                transform = transforms.ToTensor()
                input = transform(cropped)
                input = input.unsqueeze(0)
                
                with torch.no_grad():
                    logits = self.model(input)
                    probabilities = F.softmax(logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1)
                
                output = int(predicted_class)
                
                cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
                
                label = self.idx_to_label[output]
                cv2.putText(img, label.upper(), (min_x, min_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3, cv2.LINE_AA)
            
            cv2.imshow("Hand Tracker", img)
                
            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()