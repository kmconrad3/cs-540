import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    INPUT: An optional boolean argument (default value is True for training dataset)
    RETURNS: Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if training == True:
        training_set=datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
        return torch.utils.data.DataLoader(training_set, batch_size = 64)
    else:
        testing_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)
        return torch.utils.data.DataLoader(testing_set, batch_size = 64)



def build_model():
    """
    INPUT: None
    RETURNS: An untrained neural network model
    """
    model = nn.Sequential(
                nn.Flatten(),           
                nn.Linear(784, 128), 
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
                )
    return model



def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training
    RETURNS: None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    for epoch in range(T):
        running_loss = 0.0
        
        correct = 0
        total = 0  
        
        for i, data in enumerate(train_loader, 0):
            images, labels = data

            opt.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
       
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()    
        print(f"Train Epoch: {epoch}, Accuracy: {running_loss:.0f}/{len(train_loader.dataset)}({(correct / total)*100:.2f}%), Loss: {running_loss/i:.3f}")    

        

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 
    RETURNS: None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.eval()
    
    running_loss = 0.0

    correct = 0
    total = 0  
        
    for i, data in enumerate(test_loader, 0):
        images, labels = data

        opt.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
      
        
    if show_loss == True:
        print(f"Average loss: {running_loss/i:.4f}")    
        print(f"Accuracy: {(correct / total)*100:.2f}%")
    else:
        print(f"Accuracy: {(correct / total)*100:.2f}%")

        

def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1
    RETURNS: None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt' ,'Sneaker','Bag','Ankle Boot']
    inputs = test_images[index]
    outputs = model(inputs)
    
    prob = F.softmax(outputs, dim = 1).detach().numpy()[0]
    #print(prob)
    #indices = sorted(range(len(prob)), key=lambda i: prob[i], reverse=True)[:3]
    #print(indices)
    top_idx = np.argsort(prob)[-3:]
    #print(top_2_idx)
    for i in range(2,-1,-1):
        #print(class_names[top_idx[i]]+": "+str(prob[top_idx[i]]*100)+"%")
        print(f"{class_names[top_idx[i]]}: {prob[top_idx[i]]*100:.2f}%")



if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    
    #train_loader = get_data_loader()
    #model = build_model()
    #train_model(model,train_loader,criterion,5)
    #test_loader = get_data_loader(True)
    #for i, data in enumerate(test_loader, 0):
        ## get the inputs; data is a list of [inputs, labels]
        #images, labels = data
        #break
    #predict_label(model, images, 1)
    #print(type(train_loader))
    #print(train_loader.dataset)
