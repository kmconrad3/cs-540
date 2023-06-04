# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for comp visiual
import torchvision
import torchvision.transforms as transforms



class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        self.shape = input_shape #
        self.num = num_classes #
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, bias = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )
        self. conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        ) 
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=5*5*16, out_features=256, bias = True),
            nn.ReLU()
        )
        self.fc2 = nn. Sequential(
            nn.Linear(in_features=256, out_features=128, bias = True),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=100, bias = True)
        )

        
        
    def forward(self, x):
        shape_dict = {}
        
        conv1_output = self.conv1(x) 
        shape_dict[1] = list(conv1_output.shape)
        
        conv2_output = self.conv2(conv1_output) 
        shape_dict[2] = list(conv2_output.shape)
        
        flat_output = self.flat(conv2_output)
        shape_dict[3] = list(flat_output.shape)
        
        fc1_output = self.fc1(flat_output) 
        shape_dict[4] = list(fc1_output.shape)
        
        fc2_output=self.fc2(fc1_output)  
        shape_dict[5] = list(fc2_output.shape)
        
        out = self.fc3(fc2_output)
        shape_dict[6] = list(out.shape)

        return out, shape_dict

    
    
def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter. numel()
        model_params+=param
    return model_params/1e6



def train_model(model, train_loader, optimizer, criterion, epoch):
    '''
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    '''
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))
    return train_loss



def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
