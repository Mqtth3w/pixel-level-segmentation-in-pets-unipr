'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from solver import Solver
from train_dataset import OxfordIIITPetTrainDataset

def get_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument('--run_name', type=str, default="run_1", help='Name of current run')
    parser.add_argument('--model_name', type=str, default="first_train", help='Name of the model to be saved/loaded')

    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, choices=[8, 16, 32], help='Number of elements in batch size')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers in data loader')

    parser.add_argument('--img_resize', type=int, default=256, choices=[128, 256, 512], help='Image resize dimesions')

    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--opt', type=str, default='Adam', choices=['Adam', 'RSMprop', 'SGD'], help = 'Optimizer used for training')
    parser.add_argument('--loss', type=str, default='dice', choices=['dice', 'CE', 'combo'], help = 'Loss function used for training')
    parser.add_argument('--patience', type=float, default=6, help='Patience for early stopping')
    parser.add_argument('--patience2', type=float, default=5, help='Patience used by the scheduler to reduce the lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for bad gradient cases (e.g., flat zone)')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay for optimizer (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability for the convolutional layers in the model')

    parser.add_argument('--dataset_path', type=str, default='./', help='Path were to save/get the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='Path were to save the trained model')

    parser.add_argument('--resume_train', action='store_true', help='Load the model from checkpoint before training')

    return parser.parse_args()

def main(args):
    writer = SummaryWriter('./runs/' + args.run_name)

    # define transforms
    # train transforms are already defined inside the custom train dataset
    # I decide to use the size 256x256 for UNet to have a manageable training time
    img_test_transform = transforms.Compose([
        transforms.Resize((args.img_resize, args.img_resize),
                          interpolation=transforms.InterpolationMode.BICUBIC),  # to have higher quality than bilinear
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    # check the "get_dataset_info/get_mean_std.py" script for more details about the normalization
    mask_test_transform = transforms.Compose([
        transforms.Resize((args.img_resize, args.img_resize),
                          interpolation=transforms.InterpolationMode.NEAREST), # other interpolations may lead to incorrect labels
        transforms.Lambda(lambda mask: torch.as_tensor(np.array(mask)-1, dtype=torch.long))])
        # it is like ToTensor() but without [0, 1] normalization
        # dataset classes [1, 2, 3], so the -1 is necessary to satisfy the constraint >= 0 and < num_classes
    transforms.RandomChoice([transforms.RandomEqualize(), transforms.ColorJitter()])

    # load train ds
    trainset = OxfordIIITPetTrainDataset(root=args.dataset_path, 
                                         split="trainval",
                                         target_types="segmentation", 
                                         download=True,
                                         resize_dim=(args.img_resize, args.img_resize))
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size,
                                              shuffle=True, 
                                              num_workers=args.workers)
    # load test ds
    testset = torchvision.datasets.OxfordIIITPet(root=args.dataset_path, 
                                                 split="test", 
                                                 target_types="segmentation",
                                                 transform=img_test_transform, 
                                                 target_transform=mask_test_transform, 
                                                 download=True)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # define solver class
    solver = Solver(train_loader=trainloader,
            test_loader=testloader,
            device=device,
            writer=writer,
            args=args)

    # TRAIN model
    solver.train()
    
    
if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)