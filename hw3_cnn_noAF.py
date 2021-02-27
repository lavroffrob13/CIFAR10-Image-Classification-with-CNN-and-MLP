import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random, time

# Define hyperparameters
batch_size = 64
test_batch_size = 64
epochs = 40
size_input = 3072
eta = 0.01
momentum = 0.9


# Import data
def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, 4),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    test = datasets.CIFAR10('./', train=False,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

class CNN_ReLu(nn.Module):
    def __init__(self):
        super(CNN_ReLu, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(64,256,kernel_size=4,stride=2,padding=0)
        self.conv3 = nn.Conv2d(256,512, kernel_size=3,stride=2,padding=0)
        self.conv4 = nn.Conv2d(512,1024, kernel_size=5,stride=2,padding=0)
        self.fc1 = nn.Linear(1024*2*2,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1,1024*2*2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = CNN_ReLu()
use_cuda = True

if use_cuda and torch.cuda.is_available():
    model.cuda()

crit = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr = eta, momentum=momentum)

for epoch in range(epochs):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        optim.zero_grad()
        train_output = model(images)
        loss = crit(train_output,labels)
        loss.backward()
        optim.step()
        total_loss=total_loss+loss
        num_batches = i+1
    epoch_loss = total_loss / num_batches
    print('Epoch: (%d/%d), Loss = %.4f' % (epoch + 1, epochs, epoch_loss))

accurate = 0.
tot = 0.
for images, labels in test_loader:
    if use_cuda and torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    test_output = model(images)
    _, prediction=torch.max(test_output,1)
    accurate += (prediction == labels).sum()
    tot += images.shape[0]
print('Accuracy of the model on the test images: %f %%' % (100 * (accurate.float() / tot)))