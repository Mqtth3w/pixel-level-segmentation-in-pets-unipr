'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
import torch.optim as optim
import torch.nn as nn
import os

from tqdm import tqdm
from model import Net

# Define loss/help functions
def iou(pred, target):
    smooth = 1e-6 # avoid zero division
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def iou_loss(pred, target):
    return 1 - iou(pred, target)

def dc_loss(pred, target):
    smooth = 1e-6 # avoid zero division
    predf = pred.view(-1)
    targetf = target.view(-1)
    intersection = (predf * targetf).sum()
    return 1 - ((2. * intersection + smooth) /
              (predf.sum() + targetf.sum() + smooth))


class Solver(object):
    """Solver for training and testing."""

    def __init__(self, train_loader, test_loader, device, writer, args):
        """Initialize configurations."""

        self.args = args
        self.model_name = 'OxfordIIITPet_UNet_{}.pth'.format(self.args.model_name)

        # Define the model
        self.net = Net(self.args).to(device)

        # load a pretrained model
        if self.args.resume_train == True:
            self.load_model()
        
        # Define Loss function
        if self.args.loss == "BCE": # not the bank, lol
            self.criterion = nn.BCELoss()
        elif self.args.loss == "dice":
            self.criterion = dc_loss()
        elif self.args.loss == "iou":
            self.criterion == iou_loss()

        # Choose optimizer 
        if self.args.opt == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, 
                                        foreach=True)
        elif self.args.opt == "RSMprop":
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, 
                                           momentum=self.args.momentum, foreach=True)
        elif self.args.opt == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, foreach=True)
        
        self.epochs = self.args.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.device = device

        self.writer = writer

    def save_model(self):
        # if you want to save the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        torch.save(self.net.state_dict(), check_path)
        print("Model saved!")

    def load_model(self):
        # function to load the model
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path))
        print("Model loaded!")
    
    def train(self):
        self.net.train()
        for epoch in range(self.epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            loop = tqdm(enumerate(self.train_loader), total = len(self.train_loader), leave = False)
            for batch, (imgs, masks) in loop:
                # put data on correct device
                imgs = imgs.to(self.device)
                masks = masks.to(self.device) # the ground truth mask

                # zero the parameter gradients
                # forward + backward + optimize
                self.net.zero_grad()
                preds = self.net(imgs)
                loss = self.criterion(preds, masks)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()



        self.writer.flush()
        self.writer.close()
        print('Finished Training')   
    
    def test(self, epoch, batch):
        # now lets evaluate the model on the test set
        # Intersection over Union, L1 distance are good metrics to evaluate the results
        # init loss and iou, l1 metrics
        test_loss = 0.0
        tot_iou = 0.0
        tot_l1_distance = 0.0

        # put net into evaluation mode
        self.net.eval()

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for (imgs, masks) in self.test_loader:
                
                # put data on correct device
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                # calculate outputs by running images through the network
                preds = self.net(imgs)
                
                # test loss
                loss = self.criterion(preds, masks)
                test_loss += loss.item()

                # metrics!
                tot_iou += iou(preds, masks)

                l1_distance = torch.abs(preds - masks).mean()
                tot_l1_distance += l1_distance.item()

        # print and log statistics
        num_batches = len(self.test_loader)
        avg_test_loss = test_loss / num_batches
        avg_iou = tot_iou / num_batches
        avg_l1_distance = tot_l1_distance / num_batches

        self.writer.add_scalar('Test Loss (avg on batches)', 
                               avg_test_loss, epoch * num_batches + batch)
        self.writer.add_scalar('IoU (avg on batches)', 
                               avg_iou, epoch * num_batches + batch)
        self.writer.add_scalar('L1 Distance (avg on batches)', 
                               avg_l1_distance, epoch * num_batches + batch)

        print(f"Epoch {epoch}, Batch {batch}:")
        print(f"Avg test Loss: {avg_test_loss:.4f}.")
        print(f"Avg IoU: {avg_iou:.4f}.")
        print(f"Avg L1 Distance: {avg_l1_distance:.4f}.\n")
        self.net.train()