import time
import torch
import os
from Imageprocessing import Singular_image_process
import cv2
import numpy as np
from Cnn_emtion import Cnn_emtion
import torch.nn as nn
from dotenv import load_dotenv
load_dotenv()  

Directory_test = os.getenv("Directory_test")
model_path = os.getenv("best_model_path")
imgs_per_class = 100
num_classes = 7
batch_size = 64 

Emotions_paths = []
just_names = []
emtion_length = []
imgs = []
labels = []
for Emotion in os.listdir(Directory_test):
    Path = os.path.join(Directory_test,Emotion)
    count = 0
    in_path = os.listdir(Path)
    for img in in_path:
        if count < imgs_per_class:
            print(f"Processing {img} for emotion {Emotion}")
            imgs.append(Singular_image_process(os.path.join(Path,img)))
            labels.append(os.listdir(Directory_test).index(Emotion))
            count += 1
    Emotions_paths.append(Path)
    just_names.append(Emotion)
    emtion_length.append(len(os.listdir(Path)))
imgs = torch.stack(imgs,dim=0)
labels = torch.tensor(labels, dtype = torch.long)

print(f"loaded images: {imgs.shape}")
print(f"loaded labels: {labels}")
print(f"emotion lengths: {emtion_length}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Cnn_emtion(num_classes = num_classes, color_channel = 1, sp = False)
model.to(device)
print(f"Using device: {device}")

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.eval()

all_preds = []
all_labels = []



with torch.no_grad():
    for i in range(0, len(imgs), batch_size):
        batch_imgs = imgs[i:i+batch_size].to(device)
        batch_labels = labels[i:i+batch_size].to(device)

        outputs = model(batch_imgs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

# Acc
correct = sum(p == l for p, l in zip(all_preds, all_labels))
accuracy = correct / len(all_labels) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# ex 
cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
for t, p in zip(all_labels, all_preds):
    cm[t][p] += 1

print("\n rows = true labels, cols = predicted")
header = "       " + " ".join(f"{name[:5]:>5}" for name in just_names)
print(header)
for i, row in enumerate(cm):
    row_str = " ".join(f"{val:>5}" for val in row)
    print(f"{just_names[i][:5]:>5} {row_str}")
