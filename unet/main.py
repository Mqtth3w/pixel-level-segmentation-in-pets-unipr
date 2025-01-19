'''
@author Matteo Gianvenuti https://GitHub.com/Mqtth3w
@license GPL-3.0
'''

import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.tensorboard import SummaryWriter

from solver import Solver

def get_args():
    parser = argparse.ArgumentParser()   

    parser.add_argument('--run_name', type=str, default="run_1", help='Name of current run.')
    parser.add_argument('--model_name', type=str, default="first_train", help='Name of the model to be saved/loaded.')

    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of elements in batch size.')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers in data loader.')
    parser.add_argument('--print_every', type=int, default=500, help='Print losses every N iteration.')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--opt', type=str, default='Adam', choices=['Adam', 'RSMprop', 'SGD'], help = 'Optimizer used for training.')
    parser.add_argument('--loss', type=str, default='BCE', choices=['BCE', 'dice'], help = 'Loss function used for training.')
    parser.add_argument('--patience', type=float, default=5, help='Patience for early stopping.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--weight_deacy', type=float, default=1e-5, help='Weight decay for regularization.')
    #parser.add_argument('--use_norm', action='store_true', help='Use normalization layers in model.')
    #parser.add_argument('--feat', type=int, default=16, help='Number of features in model.')

    parser.add_argument('--dataset_path', type=str, default='../', help='Path were to save/get the dataset.')
    parser.add_argument('--checkpoint_path', type=str, default='./', help='Path were to save the trained model.')

    parser.add_argument('--resume_train', action='store_true', help='Load the model from checkpoint before training.')

    return parser.parse_args()

def main(args):
    writer = SummaryWriter('./runs/' + args.run_name)

    # define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), # UNet size
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10), # out of size
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0075, 0.0070, 0.0062), std=(0.0041, 0.0040, 0.0042))])
    # custom normalization, the values were calculated with the "get_mean_std.py" script

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)), # UNet size
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.0075, 0.0070, 0.0062), std=(0.0041, 0.0040, 0.0042))])
    # custom normalization, the values were calculated with the "get_mean_std.py" script

    # load train ds 
    trainset = torchvision.datasets.OxfordIIITPet(root=args.dataset_path, split="trainval",
                                                  target_types="segmentation", transform=train_transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.workers)
    # load test ds
    testset = torchvision.datasets.OxfordIIITPet(root=args.dataset_path, split="test",
                                                  target_types="segmentation", transform=test_transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.workers)

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