'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np

from tqdm import tqdm
from model import Net

# Define a dice loss functions
def iou_loss(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

def dc_loss(pred, target):
    smooth = 1e-6
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
            self.criterion = nn.BCEWithLogitsLoss()
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

        # Keep track of average training and test losses for each epoch
        avg_train_losses = []
        avg_test_losses = []
        
        # Trigger for earlystopping
        earlystopping = False 

        for epoch in range(self.epochs):  # loop over the dataset multiple times

            # Record the training and test losses for each batch in this epoch
            train_losses = []
            test_losses = []
            running_loss = 0.0

            self.net.train()

            loop = tqdm(enumerate(self.train_loader), total = len(self.train_loader), leave = False)
            for batch, (images, targets) in loop:

                images = images.to(self.device)
                targets = targets.to(self.device) # the ground truth mask
                
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                pred = self.net(images)
                loss = self.criterion(pred, targets)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
                running_loss += loss.item()

                # print statistics
                if batch % self.args.print_every == self.args.print_every - 1:  
                    
                    print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / self.args.print_every:.3f}')

                    self.writer.add_scalar('training loss',
                        running_loss / self.args.print_every,
                        epoch * len(self.train_loader) + batch)
                    
                    running_loss = 0.0

                    # Test model ####################??????????
                    self.test(epoch, batch)

            self.save_model()
        
            ##############
            self.net.eval()
            
            with torch.no_grad():     # Record and print average validation loss for each epoch 
                for test_batch, (test_images, test_targets) in enumerate(self.test_loader):
                    test_images = test_images.to(self.device)
                    test_targets = test_targets.to(self.device)
                    test_pred = self.net(test_images.detach())

                    test_loss = dc_loss(test_pred, test_targets).item()

                    test_losses.append(test_loss)

                epoch_avg_train_loss = np.mean(train_losses)
                epoch_avg_test_loss = np.mean(test_losses)
                avg_train_losses.append(epoch_avg_train_loss)
                avg_test_losses.append(epoch_avg_test_loss)

                print_msg = (f'train_loss: {epoch_avg_train_loss:.5f} ' + f'valid_loss: {epoch_avg_test_loss:.5f}')

                print(print_msg)
            
            if epoch > 10:     #Early stopping with a patience of 1 and a minimum of 10 epochs 
                if avg_test_losses[-1]<=avg_test_losses[-2]:
                    print("Early Stopping Triggered With Patience 1")
                    torch.save(self.net.state_dict(), self.args.checkpoint_path)
                    earlystopping = True 
            if earlystopping:
                break

        #return  avg_train_losses, avg_test_losses



        self.writer.flush()
        self.writer.close()
        print('Finished Training')   
    
    def test(self, epoch, i):
        # now lets evaluate the model on the test set
        correct = 0
        total = 0

        # put net into evaluation mode
        self.net.eval()

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                # put data on correct device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # calculate outputs by running images through the network
                outputs = self.net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.writer.add_scalar('test accuracy',
            100 * correct / total,
            epoch * len(self.train_loader) + i)

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
        self.net.train()