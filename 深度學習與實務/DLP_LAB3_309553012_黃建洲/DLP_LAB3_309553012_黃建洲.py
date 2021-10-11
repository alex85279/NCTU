import dataloader
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def gen_dataset(train_x, train_y, test_x, test_y):
    datasets = []
    for x, y in [(train_x, train_y), (test_x, test_y)]:
        x = torch.stack(
            [torch.Tensor(x[i]) for i in range(x.shape[0])]
        )
        y = torch.stack(
            [torch.Tensor(y[i:i+1]) for i in range(y.shape[0])]
        )
        datasets += [TensorDataset(x, y)]
        
    return datasets

train_dataset, test_dataset = gen_dataset(*dataloader.read_bci_data())
class EEGNet(nn.Module):
    def __init__(self, activation, dropout=0.25, ):
        super(EEGNet, self).__init__()
        
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(dropout)
        
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(dropout)
        
        
        )
        self.flatten_size = 736
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, self.flatten_size)
        x = self.classify(x)
        return x
        
    
        

class deepConvNet(nn.Module):
    def __init__(self, activation, dropout=0.5):
        super(deepConvNet, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1,5), stride=(1,1), bias=False)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(2,1), stride=(1,1), bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1,5), stride=(1,1), bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1,5), stride=(1,1), bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            activation(),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.Dropout(dropout)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1,5), stride=(1,1), bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            activation(),
            nn.MaxPool2d(kernel_size=(1,2), stride = (1,2)),
            nn.Dropout(dropout)
        )
        #flatten size = total input size / batch size = 550400 / 64 = 8600
        self.flatten_size = 8600
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.flatten_size, 100),
            activation()

        )
        self.fc2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(100, 50),
            activation()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(50, 2)
        )
    def forward(self, x):
        
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.flatten_size)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def run(model, train_data, test_data, optimizer, epochs=300, batch_size=64, learning_rate=1e-3  , criterion=nn.CrossEntropyLoss()):
    trainDataLoader = DataLoader(train_data, batch_size=batch_size)
    testDataLoader = DataLoader(test_data, batch_size=batch_size)
    train_loss = 0.0    
    max_test_accuracy = 0
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    test_accuracy = []
    train_accuracy = []
    for epoch in range(epochs):
        print(f"now is epoch: {epoch}")
        #train
        model.train()
        train_correct = 0
        loss = 0

        for x,y in trainDataLoader:
            
            data = x.to(device)
            #target = y.to(device).long().view(-1)
            target = y.to(device).long().view(-1)
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            output = model.forward(data)
            loss = criterion(output, target)

            
            
            #print(f"before: {y.to(device).size()}")
            #print(f"after: {target.size()}")
            #print(f"output: {output.size()}")
            loss.backward()
            optimizer.step()
            
            train_correct += (torch.max(output, 1)[1] == target).sum().item()
        #test
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for x,y in testDataLoader:
                data = x.to(device)
                target = y.to(device).long().view(-1)
                #optimizer.zero_grad()
                output = model.forward(data)
                #v_loss = criterion(output, target)
                #scheduler.step(v_loss)
                test_correct += (torch.max(output, 1)[1] == target).sum().item()
                
                #print(f"output: {torch.max(output, 1)[1]}, target: {target.long().view(-1)}")  
        if test_correct * 100 / len(test_data) > max_test_accuracy:
        
            max_test_accuracy = test_correct * 100 / len(test_data)
        train_accuracy.append(train_correct * 100 / len(train_data))
        test_accuracy.append(test_correct * 100 / len(test_data))
        print(f"train accuracy: {train_correct * 100 / len(train_data)}, epochs = {epoch}, Loss: {loss}")
        print(f"test accuracy: {test_correct * 100 / len(test_data)}, epochs = {epoch}, max_accuracy = {max_test_accuracy}")
    return train_accuracy, test_accuracy
def draw_pic(train_ReLU, test_ReLU, train_LeakyReLU, test_LeakyReLU, train_ELU, test_ELU, epochs):
    plt.plot(range(epochs), train_ReLU, label='Train ReLU')
    plt.plot(range(epochs), test_ReLU , label='Test ReLU')
    plt.plot(range(epochs), train_LeakyReLU, label='Train LeakyReLU')
    plt.plot(range(epochs), test_LeakyReLU, label='Test LeakyReLU')
    plt.plot(range(epochs), train_ELU, label='Train ReLU')
    plt.plot(range(epochs), test_ELU, label='Test ELU')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('pytorch device : ', device)
    
    EEGmodel_ReLU = EEGNet(nn.ReLU).to(device)
    EEGmodel_LeakyReLU = EEGNet(nn.LeakyReLU).to(device)
    EEGmodel_ELU = EEGNet(nn.ELU).to(device)
    optimizer_EEG_ReLU = optim.Adam(EEGmodel_ReLU.parameters(), lr = 1e-3, betas=(0.9, 0.999), eps=1e-08)
    optimizer_EEG_LeakyReLU = optim.Adam(EEGmodel_LeakyReLU.parameters(), lr = 1e-3, betas=(0.9, 0.999), eps=1e-08)
    optimizer_EEG_ELU = optim.Adam(EEGmodel_ELU.parameters(), lr = 1e-3, betas=(0.9, 0.999), eps=1e-08)
    train_eeg_ReLU, test_eeg_ReLU = run(EEGmodel_ReLU,train_dataset, test_dataset, optimizer_EEG_ReLU)
    train_eeg_LeakyReLU, test_eeg_LeakyReLU = run(EEGmodel_LeakyReLU,train_dataset, test_dataset, optimizer_EEG_LeakyReLU)
    train_eeg_ELU, test_eeg_ELU = run(EEGmodel_ELU,train_dataset, test_dataset, optimizer_EEG_ELU)
    draw_pic(train_eeg_ReLU, test_eeg_ReLU, train_eeg_LeakyReLU, test_eeg_LeakyReLU, train_eeg_ELU, test_eeg_ELU, 300)
    
    
    '''deepConvModel_ReLU = deepConvNet(nn.ReLU).to(device)
    deepConvModel_LeakyReLU = deepConvNet(nn.LeakyReLU).to(device)
    deepConvModel_ELU = deepConvNet(nn.ELU).to(device)
    optimizer_dcv_ReLU = optim.Adam(deepConvModel_ReLU.parameters(), lr = 1e-3, betas=(0.9, 0.999), eps=1e-08)
    optimizer_dcv_LeakyReLU = optim.Adam(deepConvModel_LeakyReLU.parameters(), lr = 1e-3, betas=(0.9, 0.999), eps=1e-08)
    optimizer_dcv_ELU = optim.Adam(deepConvModel_ELU.parameters(), lr = 1e-3, betas=(0.9, 0.999), eps=1e-08)
    train_dcv_ReLU, test_dcv_ReLU = run(deepConvModel_ReLU,train_dataset, test_dataset, optimizer_dcv_ReLU)
    train_dcv_LeakyReLU, test_dcv_LeakyReLU = run(deepConvModel_LeakyReLU,train_dataset, test_dataset, optimizer_dcv_LeakyReLU)
    train_dcv_ELU, test_dcv_ELU = run(deepConvModel_ELU,train_dataset, test_dataset, optimizer_dcv_ELU)
    draw_pic(train_dcv_ReLU, test_dcv_ReLU, train_dcv_LeakyReLU, test_dcv_LeakyReLU, train_dcv_ELU, test_dcv_ELU, 300)'''
    #run(deepConvModel,train_dataset, test_dataset, optimizer_dcv)
    