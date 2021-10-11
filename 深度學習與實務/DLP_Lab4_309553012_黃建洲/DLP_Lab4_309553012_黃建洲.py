import dataloader
from torchvision import transforms
from torch import nn
from torch import optim
import numpy as np
import pandas as pd
import os
import torchvision
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch import cuda
aug = [
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomVerticalFlip(p = 0.5)
    ]
test_data = dataloader.RetinopathyLoader('./data', 'test')
train_data = dataloader.RetinopathyLoader('./data','train')
train_data_aug = dataloader.RetinopathyLoader('./data','train',augmentation = aug)

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_size, out_size, stride = 1, kernel_size = 3, padding = 1, activation = nn.ReLU(inplace=True), dropout = 0.25, downsample=None):
        super(Block, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.blk1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size = kernel_size, stride = stride, padding = padding, bias = False),
            nn.BatchNorm2d(out_size),
            self.activation
        )
        self.blk2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, kernel_size = kernel_size, padding = padding, bias = False),
            nn.BatchNorm2d(out_size)
        )
        self.downsample = downsample
        
    def forward(self, x):
        res = x
        if self.downsample:
            res = self.downsample(x)
        y = self.blk1(x)
        y = self.blk2(y)
        
            
        y = y + res
        y = self.activation(y)
        
        return y
class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_size, out_size, stride = 1, activation = nn.ReLU(inplace=True), downsample=None):
        super(BottleneckBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.blk1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_size),
            self.activation
        )
        self.blk2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, stride = stride, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_size),
            self.activation
        )
        self.blk3 = nn.Sequential(
            nn.Conv2d(out_size, out_size * self.expansion, kernel_size = 1, bias = False),
            nn.BatchNorm2d(out_size * self.expansion)
            
        )
        self.downsample = downsample
        
    def forward(self, x):
        res = x
        if self.downsample != None:
            res = self.downsample(x)
        #print(res.size())
        #print(x.size())
        y = self.blk1(x)
        #print(y.size())
        y = self.blk2(y)
        #print(y.size())
        y = self.blk3(y)
        #print(y.size())
        y = y + res
        y = self.activation(y)
        
        return y   
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_size = 64
        self.init_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * block.expansion, 100),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(50, num_classes)
        )
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.init_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        
    def _make_layer(self, block, out_size, num_block, stride):
        
        downsample = None
        strides = [stride] + [1] * (num_block-1)
        layers = []

        if stride != 1 or self.in_size != out_size * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_size, out_size * block.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_size * block.expansion)
            )
            

        
        layers.append(block(self.in_size, out_size, stride=stride, downsample = downsample))
        self.in_size = out_size * block.expansion
        for i in range(num_block):
            layers.append(block(self.in_size, out_size))
        '''for i , s in enumerate(strides):
            layers.append(block(self.in_size, out_size, stride = s, downsample = downsample))
            if i == 0:
                self.in_size = out_size * block.expansion
            downsample = None'''
        
        return nn.Sequential(*layers)
        
class PretrainedResNet(nn.Module):
    def __init__(self, num_classes, version):
        super(PretrainedResNet, self).__init__()
        resnet = None
        if version == 18:
            resnet = torchvision.models.resnet18(pretrained=True)
        elif version == 50:
            resnet = torchvision.models.resnet50(pretrained=True)
        else:
            print("Wrong layer number for resnet")
            exit()
        self.conv1 = resnet._modules['conv1']
        self.bn1 = resnet._modules['bn1']
        self.act1 = resnet._modules['relu']
        self.maxpooling = resnet._modules['maxpool']
        self.layer1 = resnet._modules['layer1']
        self.layer2 = resnet._modules['layer2']
        self.layer3 = resnet._modules['layer3']
        self.layer4 = resnet._modules['layer4']
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(resnet._modules['fc'].in_features, num_classes)
        #print(resnet._modules['fc'].in_features)
        del resnet
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpooling(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpooling(x)
        
        x = x.view(x.size(0), -1)
        #print(x.size)
        
        x = self.fc(x)
        return x

def ResNet18():
    return ResNet(Block, num_blocks = [2,2,2,2], num_classes=5)
def ResNet18_pretrained():
    return PretrainedResNet(num_classes = 5, version = 18)

def ResNet50():
    return ResNet(BottleneckBlock, num_blocks = [3,4,6,3], num_classes=5)

def ResNet50_pretrained():
    return PretrainedResNet(num_classes = 5, version = 50)

def run(model, train_dataset, test_dataset, optimizer, model_name, epochs=20, batch_size=8, learning_rate=1e-3  , criterion=nn.CrossEntropyLoss()):
    trainDataLoader = DataLoader(train_dataset, batch_size=batch_size)
    testDataLoader = DataLoader(test_dataset, batch_size=batch_size)
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

            output = model(data)
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
        y_predict = np.array([])
        y_gt = np.array([])
        print(y_predict)
        print(y_gt)
        with torch.no_grad():
            for x,y in testDataLoader:
                data = x.to(device)
                target = y.to(device)
                #optimizer.zero_grad()
                output = model(data)
                #v_loss = criterion(output, target)
                #scheduler.step(v_loss)
                test_correct += (torch.max(output, 1)[1] == target.long().view(-1)).sum().item()
                y_predict = np.concatenate((y_predict, torch.max(output, 1)[1].cpu().numpy()), axis = 0)
                y_gt = np.concatenate((y_gt, target.long().view(-1).cpu().numpy()), axis = 0)
                
                '''print("testing")
                
                print(torch.max(output, 1)[1].cpu().numpy())
                print(target.long().view(-1).cpu().numpy())'''
                
                
                
                #print(f"output: {torch.max(output, 1)[1]}, target: {target.long().view(-1)}")  
        if test_correct * 100 / len(test_data) > max_test_accuracy:
            max_test_accuracy = test_correct * 100 / len(test_dataset)
        train_accuracy.append(train_correct * 100 / len(train_dataset))
        test_accuracy.append(test_correct * 100 / len(test_data))
        print(f"train accuracy: {train_correct * 100 / len(train_dataset)}, epochs = {epoch}, Loss: {loss}")
        print(f"test accuracy: {test_correct * 100 / len(test_dataset)}, epochs = {epoch}, max_accuracy = {max_test_accuracy}")
        #confusion matrix
        cm = confusion_matrix(y_gt, y_predict)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(y_predict)
        print(y_gt)
        print(cm)
        
          
    return train_accuracy, test_accuracy
def draw_pic(train_ResNet18, test_ResNet18, train_ResNet50, test_ResNet50, train_P_ResNet18, test_P_ResNet50, train_P_ResNet50, test_P_ResNet18, epochs):
    plt.plot(range(epochs), train_ResNet18, label='Train ResNet18 w/o pretrain')
    plt.plot(range(epochs), test_ResNet18 , label='Test ResNet18 w/o pretrain')
    plt.plot(range(epochs), train_ResNet50, label='Train ResNet50 w/o pretrain')
    plt.plot(range(epochs), test_ResNet50, label='Test ResNet50 w/o pretrain')
    plt.plot(range(epochs), train_P_ResNet18, label='Train ResNet18 with pretrain')
    plt.plot(range(epochs), test_P_ResNet18, label='Test ResNet18 with pretrain')
    plt.plot(range(epochs), train_P_ResNet50, label='Train ResNet50 with pretrain')
    plt.plot(range(epochs), test_P_ResNet18, label='Test ResNet50 with pretrain')
    plt.legend()
    plt.show()
    plt.savefig('result.png')
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('pytorch device : ', device)
    print("test message")
    #ResNet18_model = ResNet18().to(device)
    #ResNet50_model = ResNet50().to(device)
    #ResNet18_pretrained_model = ResNet18_pretrained().to(device)
    ResNet50_pretrained_model = ResNet50_pretrained().to(device)
    #optimizer_ResNet18 = optim.SGD(ResNet18_model.parameters(), lr = 1e-4)
    #optimizer_ResNet50 = optim.SGD(ResNet50_model.parameters(), lr = 1e-4)
    #optimizer_ResNet18_pretrained = optim.SGD(ResNet18_pretrained_model.parameters(), lr = 1e-4)
    optimizer_ResNet50_pretrained = optim.SGD(ResNet50_pretrained_model.parameters(), lr = 1e-3)
    
    train_ResNet18, test_ResNet18 = run(ResNet18_model,train_data_aug, test_data, optimizer_ResNet18, "ResNet18")
    cuda.empty_cache()
    
    train_ResNet50, test_ResNet50 = run(ResNet50_model,train_data_aug, test_data, optimizer_ResNet50, "ResNet50")
    cuda.empty_cache()
    
    train_ResNet18_pretrained, test_ResNet18_pretrained = run(ResNet18_pretrained_model, train_data_aug, test_data, optimizer_ResNet18_pretrained, "ResNet18_pretrained")
    cuda.empty_cache()
    
    train_ResNet50_pretrained, test_ResNet50_pretrained = run(ResNet50_pretrained_model, train_data_aug, test_data, optimizer_ResNet50_pretrained, "ResNet50_pretrained")
    #torch.save(ResNet50_pretrained_model.state_dict(), "ResNet50_pretrain_model.pt")
    
    draw_pic(train_ResNet18, test_ResNet18, train_ResNet50, test_ResNet50, train_ResNet18_pretrained, test_ResNet18_pretrained, train_ResNet50_pretrained, test_ResNet50_pretrained, 10)