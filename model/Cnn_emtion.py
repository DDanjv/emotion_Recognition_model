import torch
import torchvision.transforms as transforms
import os
import cv2
import torch.nn as nn




class Cnn_emtion(nn.Module):
  def __init__(self, num_classes = 7, color_channel = 1, sp = False):

    super(Cnn_emtion, self).__init__()
    self.input_channel = color_channel
    self.num_classes = num_classes

    self.Main_block = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3),
      nn.MaxPool2d(2, 2),
      nn.ReLU(True),
      nn.Conv2d(16, 32, kernel_size=3, padding=1),
      nn.MaxPool2d(2, 2),
      nn.ReLU(True),
      nn.Conv2d(32, 64, kernel_size=5, padding=2),  # more padding to keep size
      nn.MaxPool2d(2, 2),
      nn.ReLU(True),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),  # smaller kernel
      nn.MaxPool2d(2, 2),
      nn.ReLU(True)
    )
    with torch.no_grad():
            dummy = torch.zeros(1, color_channel, 48, 48)
            out = self.Main_block(dummy)
            self.flattened_size = out.numel()

    self.Fc_block = nn.Sequential(
      nn.Linear(self.flattened_size, 64),
      nn.ReLU(True),  
      nn.Dropout(0.2),
      nn.Linear(64, 32),
      nn.ReLU(True), 
      nn.Dropout(0.2),
      nn.Linear(32, num_classes),
    )

    self.Initialize_weights()

  def Initialize_weights(self):
      for layer in self.modules():
        if isinstance(layer, nn.Linear):
          torch.nn.init.kaiming_uniform_(layer.weight, mode = 'fan_out', nonlinearity='leaky_relu')
          layer.bias.data.fill_(0.01)
          if layer.bias is not None:
              nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Conv2d):
          torch.nn.init.kaiming_uniform_(layer.weight, mode = 'fan_out', nonlinearity='leaky_relu')
          layer.bias.data.fill_(0.01)
          if layer.bias is not None:
              nn.init.constant_(layer.bias, 0)

  def forward(self, tensor_input):
    tensor_input = self.Main_block(tensor_input)
    tensor_input = tensor_input.view(tensor_input.size(0), -1)
    tensor_input = self.Fc_block(tensor_input)
    return tensor_input
       

    

        

