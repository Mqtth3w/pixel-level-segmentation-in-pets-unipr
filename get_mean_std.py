import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a transform to only convert to Tensor (no normalization yet)
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Load the dataset
dataset = datasets.OxfordIIITPet(root='.', split='test', download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Compute mean and std
mean = torch.zeros(3)
std = torch.zeros(3)
for images, _ in loader:
    images = images.permute(0, 2, 3, 1).reshape(-1, 3)  # Reshape to (pixels, channels)
    mean += images.mean(0)
    std += images.std(0)

# Normalize by the number of images
mean /= len(dataset)
std /= len(dataset)

print(f"Mean: {mean}")
print(f"Std: {std}")
'''
#trainval
Mean: tensor([0.0075, 0.0070, 0.0062])
Std: tensor([0.0041, 0.0040, 0.0042])

#test
Mean: tensor([0.0077, 0.0072, 0.0063])
Std: tensor([0.0042, 0.0041, 0.0043])

# norm -> mean 0, std 1
'''