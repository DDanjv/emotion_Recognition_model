import time
import torch
import os
from trainfunc import train_model
from Imageprocessing import Singular_image_process, mult_image_process, mult_tensor_augmention
import cv2
import numpy as np
from Cnn_emtion import Cnn_emtion
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
load_dotenv()  

print("_____________________________________________________________: Training started")
Directory_train = os.getenv("Directory_train")
print("Training data directory:", Directory_train)
Existing_model_path = os.getenv("best_model_path")
print("Existing model path:", Existing_model_path)
imgs_per_class = 8000
sp = False
num_classes = 7
learn_rate = 0.0005
num_of_cycles = 50
batch_size = 128


# path for each emtion and names 
Emotions_paths = []
just_names = []
emtion_length = []
imgs = []
labels = []
for Emotion in os.listdir(Directory_train):
    Path = os.path.join(Directory_train,Emotion)
    count = 0
    in_path = os.listdir(Path)
    for img in in_path:
        if count < imgs_per_class:
            print(f"Processing {img} for emotion {Emotion}")
            imgs.append(Singular_image_process(os.path.join(Path,img)))
            labels.append(os.listdir(Directory_train).index(Emotion))
            count += 1
    Emotions_paths.append(Path)
    just_names.append(Emotion)
    emtion_length.append(len(os.listdir(Path)))
imgs = torch.stack(imgs,dim=0)
labels = torch.tensor(labels, dtype = torch.long)
print(imgs.shape)
print(labels)
print(emtion_length)

model = Cnn_emtion(num_classes = 7, color_channel = 1, sp = False)
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

if Existing_model_path and os.path.exists(Existing_model_path):
    try:
        # Load the model state dictionary
        model.load_state_dict(torch.load(Existing_model_path, map_location=device))
        print(f"Model loaded successfully from: {Existing_model_path}")
    except Exception as e:
        print(f"Error loading model from {Existing_model_path}: {e}")
        print("Starting training with a newly initialized model instead.")
else:
    print("No existing model specified or found at the given path. Starting training with a new model.")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learn_rate)

# Start training
train_model(model, imgs, labels, criterion, optimizer, num_classes, batch_size, num_of_cycles)

print("_____________________________________________________________: Training finished")

# Save model
os.makedirs(os.path.dirname(Existing_model_path), exist_ok=True)
model_path = os.path.join(
    os.path.dirname(Existing_model_path),
    f"facial_expression_model_{time.strftime('%Y%m%d-%H%M%S')}{'spaeffect' if sp else ''}.pth"
)
torch.save(model.state_dict(), model_path)
print("_____________________________________________________________: Model saved to " + model_path)



