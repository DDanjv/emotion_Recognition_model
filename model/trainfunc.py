import time
import torch
import os
import cv2
import torchvision.transforms as transforms
import torch.nn as nn
from Imageprocessing import Singular_image_process, mult_image_process, mult_tensor_augmention
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.data import Subset
import torchvision.transforms as transforms
from dotenv import load_dotenv
load_dotenv()

best_model_dir = os.getenv("best_model_path")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),              
    transforms.RandomRotation(10),                  
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  
    transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
    transforms.RandomAdjustSharpness(2.0, p=0.5)
])

def to_tensorload( imgs, labels, num_classes, batch_size):
    dataset = TensorDataset(imgs, labels)

    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()  

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    dataset_loaded = DataLoader(dataset, batch_size = batch_size, sampler = sampler)
    return dataset_loaded

def to_tensorload_datset_spec(dataset, num_classes, batch_size):

    labels = torch.tensor([dataset.dataset[i][1] for i in dataset.indices])

    # need to rework to give futher bias 
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()  

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    dataset_loaded = DataLoader(dataset, batch_size = batch_size, sampler = sampler)
    return dataset_loaded


def loop_helper(model, tensordata, device, optimizer, criterion, train_mode=True):
    correct = 0
    total = 0
    start = time.time()
    running_loss = 0.0

    for batches_of_images, labels_in_loader in tensordata:
            batches_of_images = batches_of_images.to(device)
            labels_in_loader = labels_in_loader.to(device)

            if train_mode:
                for i in range(batches_of_images.size(0)):
                    batches_of_images[i] = train_transform(batches_of_images[i])
            if train_mode:
                optimizer.zero_grad()

            outputs = model(batches_of_images)
            loss = criterion(outputs,labels_in_loader.long())

            if train_mode:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels_in_loader.size(0)
            correct += predicted.eq(labels_in_loader).sum().item()
    avg_loss = running_loss / len(tensordata)
    accuracy = 100 * correct/ total
    return avg_loss, accuracy

def train_val_dataset(imgs, labels, val_split=0.25):
    dataset = TensorDataset(imgs, labels)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


def train_model(model, imgs, labels, criterion, optimizer, num_classes = 7, batch_size = 32, num_of_cycles = 12):

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_dataset, val_dataset = train_val_dataset(imgs, labels, val_split=0.25)

    train_dataset = to_tensorload_datset_spec(train_dataset, num_classes, batch_size)
    
    val_dataset = DataLoader(val_dataset, batch_size= batch_size, shuffle= False)
    
    best_val_acc = 0.0  # track best accuracy
    best_model_path = "best_model.pth"

    for cycle in range(num_of_cycles):
        print(f"Cycle {cycle+1}/{num_of_cycles}")
        model.train()

        train_loss,train_accuracy = loop_helper(model, train_dataset, device, optimizer, criterion, train_mode=True)

        model.eval()
        with torch.no_grad():
            Validation_loss, Validation_accuracy = loop_helper(model, val_dataset, device, optimizer, criterion, train_mode=False)

        print(f"Training loss:{train_loss:.4f}, train accuary{train_accuracy}")
        print(f"Validation loss:{Validation_loss:.4f}, Validation accuary{Validation_accuracy}")

        if Validation_accuracy > best_val_acc:
            best_val_acc = Validation_accuracy
            print("Validation accuracy is higher than training accuracy, saving model...")
            os.makedirs(os.path.dirname(best_model_dir), exist_ok=True)
            model_path = os.path.join(
                os.path.dirname(best_model_dir),
                f"facial_expression_model_best_{time.strftime('%Y%m%d-%H%M%S')}.pth"
            )
            torch.save(model.state_dict(),best_model_dir)
            print(f"Model saved to {model_path}")
            

    
    print("finshed training")



