'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

# Define the DeepLabV3 model architecture with a ResNet-101 backbone

import torchvision
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        # num_classes=3 (backgroud, pet edge, pet)
        self.net = torchvision.models.segmentation.deeplabv3_resnet101(weights=torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
                                                                       #num_classes=3,
                                                                       )
        # only these pretrained weights are available, with the constraint of 21 classes so I manually change the last layer in the classifier to have 3 classes
        self.net.classifier[4] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
        self.net.aux_classifier[4] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))

        # map into probabilities
        self.output_activation = nn.Softmax(dim=1)

        # freze the backbone except the last part (layer4 and classifier)
        # this to have enough degree of fredom and make this finetune comparable with the UNet complete train
        # also because is necessary to pass from 21 classes weights to 3
        for name, param in self.net.backbone.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False
        
    def forward(self, img):
        x = self.net(img)
        x = x['out'] # extract the output tensor
        x = self.output_activation(x)
        return x