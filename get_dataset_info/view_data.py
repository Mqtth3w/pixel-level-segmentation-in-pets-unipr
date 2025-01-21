'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((256, 256)), # UNet size
    transforms.ToTensor()])

img = Image.open('Desktop/DL/project/oxford-iiit-pet/images/Abyssinian_1.jpg')
figure = plt.figure(figsize=(4, 4))
cols, rows = 2, 1
figure.add_subplot(rows, cols, 1)
plt.imshow(img)
plt.axis("off")

mask = Image.open('Desktop/DL/project/oxford-iiit-pet/annotations/trimaps/Abyssinian_1.png')
figure.add_subplot(rows, cols, 2)
plt.imshow(mask)
plt.axis("off")
#plt.show()

t = transform(img)
#print(t)
print(t.size())
t2 = transform(mask)
#torch.set_printoptions(threshold=torch.inf)
#print(t2)
print(t2.size())


'''
if torch.any(t2 != 0.0078):
    print("diff.")
else:
    print("all eq.")
'''
