'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image


import random
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF

class OxfordIIITPetTrainDataset(OxfordIIITPet):
    """A custom dataset to apply the same random horizontal flip to img and mask."""
    def __init__(self, root, split='trainval', target_types='segmentation', 
                 download=False):
        super().__init__(root, split=split, target_types=target_types, 
                         download=download)

    def custom_transforms(self, image, mask):
        # resize
        img_resize = transforms.Resize((256, 256), # UNet size
                                       interpolation=transforms.InterpolationMode.BICUBIC) # to have higher quality than bilinear
        mask_resize = transforms.Resize((256, 256), # UNet size
                                       interpolation=transforms.InterpolationMode.NEAREST) # other interpolations may lead to incorrect labels
        image = img_resize(image)
        mask = mask_resize(mask)

        # random horizontal flipping (applied to both image and mask)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # normalize the img only
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # custom normalization, the values were calculated with the "get_mean_std.py" script
        image = normalize(image)

        return image, mask

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        # apply custom transforms to both image and mask
        image, mask = self.custom_transforms(image, mask)

        return image, mask



img_transform = transforms.Compose([
    transforms.Resize((256, 256), # UNet size
                      interpolation=transforms.InterpolationMode.BICUBIC),  # to have higher quality than bilinear
    transforms.ToTensor()])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), # UNet size
                      interpolation=transforms.InterpolationMode.NEAREST), # other interpolations may lead to incorrect labels
    transforms.ToTensor()])

'''
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
'''

root = "."
trainset = OxfordIIITPetTrainDataset(root=root, 
                                    split="trainval",
                                    target_types="segmentation", 
                                    download=True)
trainloader = torch.utils.data.DataLoader(trainset, 
                                        batch_size=1,
                                        shuffle=True, 
                                        num_workers=2)
# load test ds
testset = OxfordIIITPet(root=root, 
                        split="test", 
                        target_types="segmentation",
                        transform=img_transform, 
                        target_transform=mask_transform, 
                        download=True)
testloader = torch.utils.data.DataLoader(testset, 
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=2)

def visualize_dataset(data_loader, num_samples=5):
    figure = plt.figure(figsize=(10, 5 * num_samples))
    for i, (img, mask) in enumerate(data_loader):
        if i >= num_samples:
            break

        # tensor to numpy array
        img_np = img.squeeze(0).permute(1, 2, 0).numpy() # CxHxW to HxWxC
        mask_np = mask.squeeze(0).squeeze(0).numpy() # CxHxW to HxW

        # img
        ax = figure.add_subplot(num_samples, 2, i * 2 + 1)
        ax.imshow(img_np)
        ax.set_title(f"Image {i + 1}")
        ax.axis("off")

        # mask
        ax = figure.add_subplot(num_samples, 2, i * 2 + 2)
        ax.imshow(mask_np, cmap="gray")
        ax.set_title(f"Mask {i + 1}")
        ax.axis("off")

    plt.show()

if __name__ == "__main__":
    # view few imgs and corresponding masks from train
    visualize_dataset(trainloader, num_samples=5)

    # view few imgs and corresponding from test
    visualize_dataset(testloader, num_samples=5)