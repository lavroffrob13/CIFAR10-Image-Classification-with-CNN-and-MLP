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
epochs = 10
size_input = 3072
classes = 10
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


# Build fully connected network with ReLu activation
class FC_ReLu(nn.Module):
    def __init__(self, size_input, classes):
        super(FC_ReLu, self).__init__()
        self.fc1 = nn.Linear(size_input, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


# Set up loss function and optimizer. Use GPU via CUDA if available
model = FC_ReLu(size_input, classes)
use_CUDA = True

if use_CUDA and torch.cuda.is_available():
    model.cuda()

optim = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum)
crit = nn.CrossEntropyLoss()

# Train the model
print('Beginning to train the model')
for epoch in range(epochs):
    loss_tot = 0.
    num_batches = 0.
    for i, (images, labels) in enumerate(train_loader):
        batch_count = images.shape[0]
        images = Variable(images.view(-1, 3 * 32 * 32))
        if use_CUDA and torch.cuda.is_available():
            images = Variable(images.view(-1, 3 * 32 * 32)).cuda()
            labels = labels.cuda()
        optim.zero_grad()
        tr_output = model(images)
        loss = crit(tr_output, labels)
        loss.backward() 
        optim.step()
        loss_tot = loss_tot + loss
        num_batches = i + 1
    loss_epoch = loss_tot / num_batches
    print('Epoch %d of %d, Loss = %.4f' % (epoch + 1, epochs, loss_epoch))

# Test the model
print('Beginning to test the model')
accurate = 0.
tot = 0.
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images.view(-1, 3*32*32))
        _, predicted = torch.max(outputs.data, 1)
        tot += labels.size(0)
        accurate += (predicted == labels).sum().item()
print('Accuracy of the network on the test images: %d %%' % (
    100 * accurate / tot))
