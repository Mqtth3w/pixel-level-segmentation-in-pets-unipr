'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


img_transform = transforms.Compose([
    transforms.Resize((256, 256),
                      interpolation=transforms.InterpolationMode.BICUBIC),  # to have higher quality than bilinear
    transforms.ToTensor()])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256),
                      interpolation=transforms.InterpolationMode.NEAREST), # other interpolations may lead to incorrect labels
    transforms.Lambda(lambda mask: torch.as_tensor(np.array(mask), dtype=torch.long))])

'''
# see one example as img and tensor
img = Image.open('C:/Users/MATTEO/Desktop/DL/project/oxford-iiit-pet/images/Abyssinian_10.jpg')
figure = plt.figure(figsize=(4, 4))
cols, rows = 2, 1
figure.add_subplot(rows, cols, 1)
plt.imshow(img)
plt.axis("off")

mask = Image.open('C:/Users/MATTEO/Desktop/DL/project/oxford-iiit-pet/annotations/trimaps/Abyssinian_10.png')
figure.add_subplot(rows, cols, 2)
plt.imshow(mask)
plt.axis("off")
plt.show()

print(torch.unique(mask)) # see lables [1, 2, 3]
t = img_transform(img)
print(t)
print("pure img (no batch)", t.size())
t2 = mask_transform(mask) 
#torch.set_printoptions(threshold=torch.inf)
print(t2)
print("pure mask (no batch)", t2.size())
print(torch.unique(t2))
'''

import numpy as np
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

root = "."
'''
transform = transforms.Compose([
    transforms.Lambda(lambda mask: torch.as_tensor(np.array(mask), dtype=torch.long))
])
dataset = OxfordIIITPet(root=root, split="trainval", target_types="segmentation", 
                                download=True, transform=transform, target_transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
# Initialize a set to store unique labels across the entire dataset
all_unique_labels = set()
for idx, (_, target) in enumerate(data_loader):
    unique_labels = torch.unique(target).tolist()
    all_unique_labels.update(unique_labels)
    print(f"Image {idx + 1}: Unique labels: {unique_labels}")
print("\nOverall unique labels in the dataset:", sorted(all_unique_labels))
'''


import random
import torchvision.transforms.functional as TF

class OxfordIIITPetTrainDataset(OxfordIIITPet):
    """A custom dataset to apply the same random horizontal flip to img and mask."""
    def __init__(self, root, split='trainval', target_types='segmentation', 
                 download=False, resize_dim=(256, 256)):
        super().__init__(root, split=split, target_types=target_types, 
                         download=download)
        
        self.resize_dim = resize_dim

    def custom_transforms(self, image, mask):
        # resize
        img_resize = transforms.Resize(self.resize_dim,
                                       interpolation=transforms.InterpolationMode.BICUBIC) # to have higher quality than bilinear
        mask_resize = transforms.Resize(self.resize_dim,
                                       interpolation=transforms.InterpolationMode.NEAREST) # other interpolations may lead to incorrect labels
        image = img_resize(image)
        mask = mask_resize(mask)

        # random horizontal flipping (applied to both image and mask)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # random color jitter keeping a realistic image (little variation)
        if random.random() > 0.5:
            image = transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                                           saturation=0.1, hue=0.05)(image)

        # transform to tensor
        image = TF.to_tensor(image)
        mask = torch.tensor(np.array(mask)-1, dtype=torch.long) # tensor without [0, 1] normalization
        # dataset classes [1, 2, 3], so the -1 is necessary to satisfy the constraint >= 0 and < num_classes
        # normalize the image only
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # check the "get_dataset_info/get_mean_std.py" script for more details about the normalization
        image = normalize(image)
        
        return image, mask

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        # apply custom transforms to both image and mask
        image, mask = self.custom_transforms(image, mask)

        return image, mask

trainset = OxfordIIITPetTrainDataset(root=root, 
                                    split="trainval",
                                    target_types="segmentation", 
                                    download=True)
trainloader = torch.utils.data.DataLoader(trainset, 
                                        batch_size=1,
                                        shuffle=False, 
                                        num_workers=2)
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
    visualize_dataset(trainloader, num_samples=5)
    visualize_dataset(testloader, num_samples=5)