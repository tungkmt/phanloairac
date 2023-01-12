import copy
import time
import random
import numpy as np
import torch.nn.functional as f

from PIL import Image
from torch import nn, utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from torch.utils.data import DataLoader, Dataset
resize = 224
num_epochs = 10
savepath = '.\\models.pth'
data_transforms = {
    'train': transforms.Compose([

        transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ]),
    'test': transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

}

path = '.\\dataset_rac\\'
train_dataset = torchvision.datasets.ImageFolder(path, transform=data_transforms['train'])
test_dataset = torchvision.datasets.ImageFolder(path, transform=data_transforms['test'])
LABELS = train_dataset.classes


indices = np.random.permutation(len(train_dataset)).tolist()
test_ratio = 0.2
test_border = len(train_dataset) - int(len(train_dataset) * (test_ratio))

train_data = torch.utils.data.Subset(train_dataset, indices[:test_border])
test_data = torch.utils.data.Subset(test_dataset, indices[test_border:])

train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size

train_data, val_data = utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

dataloader_dict = {"train":train_loader, "val":val_loader}

net = torchvision.models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 6)

criterior = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataloader_dict, criterion, optimizer, num_epochs):
    begin = time.time()
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        model.to(device)
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy =  running_corrects.double() / len(dataloader_dict[phase].dataset)
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))


    end = time.time()
    print("Training time: {:.4f}s".format(end - begin))
    torch.save(model.state_dict(), savepath)

if __name__ == '__main__':
    train(net,dataloader_dict, criterior, optimizer, num_epochs)

