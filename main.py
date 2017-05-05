'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar
from torch.autograd import Variable
from loaders import create_loaders

def run_model(dataset, lr, momentum, weight_decay, verbose=False):
    use_cuda = torch.cuda.is_available()

    train_loader, val_loader, test_loader = create_loaders(dataset=dataset,
                                                           cuda=True,
                                                           augment=True,
                                                           batch_size=128,
                                                           test_batch_size=100)

    # Data
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = ConvNet(dataset=dataset)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    # Training
    def train(epoch):
        if verbose:
            print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if verbose and batch_idx % 20 == 0:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test():
        if verbose:
            print('Test')
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if verbose and batch_idx % 20 == 0:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return correct / total

    def validate():
        if verbose:
            print('Validate')
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if verbose and batch_idx % 20 == 0:
                print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return correct / total

    optimizer = optim.SGD(net.parameters(), lr=lr[0], momentum=momentum,
                          weight_decay=weight_decay)
    for epoch in range(0, 25):
        train(epoch)
        validate()
        test()

    optimizer = optim.SGD(net.parameters(), lr=lr[1], momentum=momentum,
                          weight_decay=weight_decay)
    for epoch in range(26, 50):
        train(epoch)
        validate()
        test()

    optimizer = optim.SGD(net.parameters(), lr=lr[2], momentum=momentum,
                          weight_decay=weight_decay)
    for epoch in range(51, 75):
        train(epoch)
        validate()
        test()

    val_accuracy = validate()
    test_accuracy = test()

    return val_accuracy, test_accuracy

if __name__ == '__main__':
    run_model([2], [0.1, 0.01, 0.001], 0.9, 5e-4, verbose=True)
