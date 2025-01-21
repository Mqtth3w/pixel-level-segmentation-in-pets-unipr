'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import random
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms.functional as TF

class OxfordIIITPetTrainDataset(OxfordIIITPet):
    """A custom dataset to apply the same random horizontal flip to img and mask."""
    def __init__(self, root, split='trainval', target_types='segmentation', 
                 download=False):
        super().__init__(root, split=split, target_types=target_types, 
                         download=download)

    def transform(self, image, mask):
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
        normalize = transforms.Normalize(mean=(0.0075, 0.0070, 0.0062), 
                                         std=(0.0042, 0.0041, 0.0042))
        # custom normalization, the values were calculated with the "get_mean_std.py" script
        image = normalize(image)

        return image, mask

    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        # apply custom transformations to both image and mask
        image, mask = self.transform(image, mask)

        return image, mask