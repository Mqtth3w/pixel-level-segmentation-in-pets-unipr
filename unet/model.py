'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

# Define the UNet architecture

import torch
import torch.nn as nn

def conv_layer(input_channels, output_channels, dropout_probability): # This is a helper function to create the convolutional blocks
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.Dropout(p=dropout_probability)
    )
    return conv

class Net(nn.Module):
    def __init__(self, args = None):
        super(Net, self).__init__()

        self.dropout = getattr(args, 'dropout', 0.1)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # input dim 256x256
        self.down_1 = conv_layer(3, 64, self.dropout) #128x128
        self.down_2 = conv_layer(64, 128, self.dropout) #64x64
        self.down_3 = conv_layer(128, 256, self.dropout) #32x32
        self.down_4 = conv_layer(256, 512, self.dropout) #16x16
        self.down_5 = conv_layer(512, 1024, self.dropout) #8x8
        
        self.up_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = conv_layer(1024, 512, self.dropout)
        self.up_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = conv_layer(512, 256, self.dropout)
        self.up_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = conv_layer(256, 128, self.dropout)
        self.up_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = conv_layer(128, 64, self.dropout)
        # three classes as the Oxford IIIT Pet dataset (each mask should "have" the background, the pet edge and the pet itself)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, padding=0)
        self.output_activation = nn.Softmax(dim=1) # [B, C, H, W] 
           
    def forward(self, img): # The print statements can be used to visualize the input and output sizes for debugging
        x1 = self.down_1(img) #256x256
        #print(x1.size())
        x2 = self.max_pool(x1)
        #print(x2.size())
        x3 = self.down_2(x2)
        #print(x3.size())
        x4 = self.max_pool(x3)
        #print(x4.size())
        x5 = self.down_3(x4)
        #print(x5.size())
        x6 = self.max_pool(x5)
        #print(x6.size())
        x7 = self.down_4(x6)
        #print(x7.size())
        x8 = self.max_pool(x7) #8x8
        #print(x8.size())
        x9 = self.down_5(x8)
        #print(x9.size())
        
        x = self.up_1(x9) # x.size() = bs, 512, h, w x7.size() = bs, 512, h, w #16x16
        #print(x.size())
        x = self.up_conv_1(torch.cat([x, x7], 1)) # torch.cat([x, x7], 1).size() = bs, 1024, h, w
        #print(x.size())
        x = self.up_2(x)
        #print(x.size())
        x = self.up_conv_2(torch.cat([x, x5], 1)) 
        #print(x.size())
        x = self.up_3(x)
        #print(x.size())
        x = self.up_conv_3(torch.cat([x, x3], 1))
        #print(x.size())
        x = self.up_4(x)
        #print(x.size())
        x = self.up_conv_4(torch.cat([x, x1], 1)) #256x256
        #print(x.size())
        
        x = self.output(x)
        x = self.output_activation(x)
        #print(x.size())
        
        return x