import numpy as np
import torch
import torchvision.transforms as transforms
import os
import cv2


def Singular_image_process(path_to_image):
    Image = cv2.imread(path_to_image)
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    tensor = transform(Image)
    return tensor

def mult_image_process(path_to_image_dir):
    array_of_tensor = []
    print(f"file: {path_to_image_dir}")
    for image in os.listdir(path_to_image_dir):
        path_to_image = os.path.join(path_to_image_dir,image)
        if os.path.isfile(path_to_image):
            array_of_tensor.append(Singular_image_process(path_to_image))
    return torch.stack(array_of_tensor)

def mult_tensor_augmention(array_of_tensor):
    fixed_arr = []
    print(f"tenosr: {array_of_tensor}")
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(43, 43), antialias=False),
        transforms.Resize(size=(43, 43)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    for tensor in array_of_tensor:
        fixed_arr.append(transform(tensor))
    print("check")
    return fixed_arr

def effects_image_(X):
    X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(X, 100, 200)
    sobelx = cv2.Sobel(X, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(X, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edge = np.sqrt(sobelx**2 + sobely**2)
    blurred = cv2.GaussianBlur(X, (5, 5), 0)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    hog_features, _ = hog.compute(X, winStride=(8, 8), padding=(32, 32))
    return edges, sobel_edge, blurred, hog_features
    



