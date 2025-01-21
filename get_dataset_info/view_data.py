'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

img_transform = transforms.Compose([
    transforms.Resize((256, 256), # UNet size
                      interpolation=transforms.InterpolationMode.BICUBIC),  # to have higher quality than bilinear
    transforms.ToTensor()])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), # UNet size
                      interpolation=transforms.InterpolationMode.NEAREST), # other interpolations may lead to incorrect labels
    transforms.ToTensor()])

img = Image.open('C:/Users/MATTEO/Desktop/DL/project/oxford-iiit-pet/images/Abyssinian_1.jpg')
figure = plt.figure(figsize=(4, 4))
cols, rows = 2, 1
figure.add_subplot(rows, cols, 1)
plt.imshow(img)
plt.axis("off")

mask = Image.open('C:/Users/MATTEO/Desktop/DL/project/oxford-iiit-pet/annotations/trimaps/Abyssinian_1.png')
figure.add_subplot(rows, cols, 2)
plt.imshow(mask)
plt.axis("off")
#plt.show()

t = img_transform(img)
print(t)
print(t.size())
t2 = mask_transform(mask) 
#torch.set_printoptions(threshold=torch.inf)
print(t2)
print(t2.size())