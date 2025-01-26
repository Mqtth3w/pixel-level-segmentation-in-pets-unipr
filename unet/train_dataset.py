'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
import numpy as np
import random
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF
import torch.nn.functional as F

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

        # transform to tensor
        image = TF.to_tensor(image)
        mask = torch.tensor(np.array(mask)-1, dtype=torch.long) # tensor without [0, 1] normalization
        # dataset classes [1, 2, 3], so the -1 is necessary to satisfy the constraint >= 0 and <= num_classes
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