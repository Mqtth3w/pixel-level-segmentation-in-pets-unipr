'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from tqdm import tqdm
from model import Net

# Define loss/help functions
def iou(pred, target):
    smooth = 1e-4 # avoid zero division
    # pred is [B, 3, H, W] and target is [B, H, W]
    # so one hot encoding fo the mask is necessary
    B, H, W = target.size()
    onehot = torch.zeros(B, 3, H, W, device=target.device, dtype=torch.float32)
    target = target.unsqueeze(1) - 1
    # dataset labels [1, 2, 3], indexes needed [0, 1, 2]
    target = onehot.scatter(1, target, 1)
    # with the sum dim=(1, 2, 3) it will consider each img separately 
    # so each img will equally contribute
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def dc_loss(pred, target):
    smooth = 1e-4 # avoid zero division
    # the same idea used in iou is applied here
    B, H, W = target.size()
    onehot = torch.zeros(B, 3, H, W, device=target.device, dtype=torch.float32)
    target = target.unsqueeze(1) - 1
    target = onehot.scatter(1, target, 1)
    #predf = pred.view(pred.size(0), -1)
    #targetf = target.view(target.size(0), -1)
    #intersection = (predf * targetf).sum(dim=1)
    #dice = ((2. * intersection + smooth) /
              #(predf.sum(dim=1) + targetf.sum(dim=1) + smooth))
    intersection = (pred * target).sum(dim=(1, 2, 3)) # this should use less memory
    dice = ((2. * intersection + smooth) /
              (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth))
    return 1 - dice.mean() 

def dc_ce_loss(pred, target):
    ce_loss = nn.CrossEntropyLoss()
    B, H, W = target.size()
    onehot = torch.zeros(B, 3, H, W, device=target.device, dtype=torch.long)
    target2 = target.unsqueeze(1) - 1
    target2 = onehot.scatter(1, target2, 1)
    return ce_loss(pred, target2) + dc_loss(pred, target)

class Solver(object):
    """Solver for training and testing."""

    def __init__(self, train_loader, test_loader, device, writer, args):
        """Initialize configurations."""

        self.args = args
        self.model_name = 'OxfordIIITPet_{}.pth'.format(self.args.model_name)

        # define the model
        self.net = Net(self.args).to(device)

        # load a pretrained model
        if self.args.resume_train == True:
            self.load_model()
        
        # define Loss function
        if self.args.loss == "dice": # it should be similar to IoU but faster in convergence and more stable
            self.criterion = dc_loss # focus on overlap btween pred mask and ground truth
        elif self.args.loss == "CE": # multi-class loss
            self.criterion = nn.CrossEntropyLoss() 
        elif self.args.loss == "combo": 
            self.criterion = dc_ce_loss

        # choose optimizer 
        if self.args.opt == "Adam": # more adaptive (faster convergence) and robust (e.g., bad initial lr)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, 
                                        foreach=True) # Adam is also reccomended by google developers forum
        elif self.args.opt == "RSMprop": # used by milesial/Pytorch-UNet but I think Adam is better as said before
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, 
                                           momentum=self.args.momentum, foreach=True)
        elif self.args.opt == "SGD": # static lr, not very good I think I will not use it here
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay,
                                       momentum=self.args.momentum, foreach=True)
        
        # scheduler to change lr during the trainig for better convergence, and to maximize the IoU
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=self.args.patience2) 

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
        # to save only the best model
        best_iou = 0.0 
        best_l1_distance = 1000
        # for early stopping 
        bad_epochs_ctr = 0

        # save training/validation time
        tot_time = 0.0

        self.net.train()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            # epoch time
            start = time.time()
            # epoch loss
            running_loss = 0.0
            loop = tqdm(enumerate(self.train_loader), total = len(self.train_loader), leave = False)
            for batch, (imgs, masks) in loop:
                # add loop info
                loop.set_description(f"Epoch [{epoch+1}/{self.epochs}]")

                # put data on correct device
                imgs = imgs.to(device=self.device, dtype=torch.float32)
                masks = masks.to(device=self.device, dtype=torch.long) # the ground truth mask

                # zero the parameter gradients
                self.net.zero_grad()
                # forward + backward + optimize
                preds = self.net(imgs)
                loss = self.criterion(preds, masks)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # show batch loss
                loop.set_postfix(loss=loss.item())

            # epoch finished, print statistics
            self.writer.add_scalar('Training loss',
                    running_loss / len(self.train_loader),
                    epoch * len(self.train_loader))
            print(f"Epoch {epoch+1}, training loss {running_loss / len(self.train_loader)}")

            # test the model (for each epoch it's more regular and standard than with print_every)
            iou, l1_distance = self.test(epoch+1)

            # maximize goal with lr variations
            self.scheduler.step(iou)

            # time statistics
            end = time.time()
            print(f"Took {((end - start) / 60):.4f} minutes for epoch {epoch}")
            tot_time += end - start
            self.writer.add_scalar('Epoch time',
                    (end - start) / 60,
                    epoch * len(self.train_loader))

            #self.save_model()
            # save only the best model 
            if iou > best_iou or (iou == best_iou and l1_distance < best_l1_distance):
                best_iou = iou
                best_l1_distance = l1_distance
                self.save_model()
                self.writer.add_text('Info best model', f"New best model saved with IoU {best_iou:.4f}, L1 distance {best_l1_distance:.4f}")
                print(f"New best model saved with IoU {best_iou:.4f}, L1 distance {best_l1_distance:.4f}")
            else:
                bad_epochs_ctr += 1

            # early stopping
            if epoch > 10:
                if bad_epochs_ctr >= self.args.patience:
                    self.writer.add_text('Info early stopping ', f"Early stopping triggered with patience {self.args.patience} at epoch {epoch + 1}")
                    print(f"Early stopping triggered with patience {self.args.patience} at epoch {epoch + 1}")
                    break
        
        print(f"Took {(tot_time / 60):.4f} minutes for trainvaltest")
        self.writer.add_text('Total time', f"{tot_time / 60}")

        self.writer.flush()
        self.writer.close()
        print("Finished Training")   
    
    def test(self, epoch):
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

        self.writer.add_scalar('Test loss (avg on test)', 
                               avg_test_loss, epoch * num_batches)
        self.writer.add_scalar('IoU (avg on test)', 
                               avg_iou, epoch * num_batches)
        self.writer.add_scalar('L1 Distance (avg on test)', 
                               avg_l1_distance, epoch * num_batches)

        print(f"Epoch {epoch}, test data:")
        print(f"Avg test Loss: {avg_test_loss:.4f}")
        print(f"Avg IoU: {avg_iou:.4f}")
        print(f"Avg L1 Distance: {avg_l1_distance:.4f}\n")
        self.net.train()
        return avg_iou, avg_l1_distance