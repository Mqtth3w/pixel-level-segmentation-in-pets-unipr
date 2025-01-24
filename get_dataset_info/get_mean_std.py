'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# define a transform to only convert to Tensor (no normalization yet)
transform = transforms.Compose([
    transforms.Resize((256, 256), # UNet size
                      interpolation=transforms.InterpolationMode.BICUBIC),  # to have higher quality than bilinear
    transforms.ToTensor()])

# load the dataset
dataset = datasets.OxfordIIITPet(root='.', split='test', download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# compute mean and std
mean = torch.zeros(3)
std = torch.zeros(3)
for images, _ in loader:
    images = images.permute(0, 2, 3, 1).reshape(-1, 3)  # Reshape to (pixels, channels)
    mean += images.mean(0)
    std += images.std(0)

# normalize by the number of images
mean /= len(dataset)
std /= len(dataset)

print(f"Mean: {mean}")
print(f"Std: {std}")

# IMPORTANT
# I tried to train the net on these normalization values, but they are too small so the imgs became really compressed and not understandable for the model
# So I decided to use the mean and norm of ImageNet bc they are calculated on millions of images (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# I alwasy used normalized images for the model because we saw in lab that the model perform a lot better with it than without
'''
# BILINEAR
#trainval
Mean: tensor([0.0075, 0.0070, 0.0062])
Std: tensor([0.0041, 0.0040, 0.0042])
#test
Mean: tensor([0.0077, 0.0072, 0.0063])
Std: tensor([0.0042, 0.0041, 0.0043])

# BICUBIC
Mean: tensor([0.0075, 0.0070, 0.0062])
Std: tensor([0.0042, 0.0041, 0.0042])
#test
Mean: tensor([0.0077, 0.0072, 0.0063])
Std: tensor([0.0042, 0.0041, 0.0043])
'''