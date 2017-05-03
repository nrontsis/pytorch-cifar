import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os

from torchnet.dataset import SplitDataset

def create_loaders(dataset, cuda, augment, batch_size, test_batch_size,
                   use_indexing=False):

    kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}

    dataset = datasets.CIFAR10 if dataset == 'cifar10' \
        else datasets.CIFAR100
    root = './data'

    # Data loading code
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    normalize = transforms.Normalize(mean=[x / 255.0 for x in mean],
                                     std=[x / 255.0 for x in std])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    # define two datasets in order to have different transforms
    # on training and validation
    dataset_train = dataset(root=root, train=True,
                            transform=transform_train)
    dataset_val = dataset(root=root, train=True,
                          transform=transform_test)

    # partition data accordingly to split
    partition = {'train': 0.9, 'val': 0.1}
    dataset_train = SplitDataset(dataset_train, partition,
                                 initial_partition='train')
    dataset_val = SplitDataset(dataset_val, partition,
                               initial_partition='val')

    train_loader = data.DataLoader(dataset_train,
                                   batch_size=batch_size,
                                   shuffle=True, **kwargs)

    val_loader = data.DataLoader(dataset_val,
                                 batch_size=test_batch_size,
                                 shuffle=True, **kwargs)

    test_loader = data.DataLoader(dataset(root=root, train=False,
                                          transform=transform_test),
                                  batch_size=test_batch_size,
                                  shuffle=False, **kwargs)

    train_loader.tag = 'train'
    val_loader.tag = 'val'
    test_loader.tag = 'test'

    return train_loader, val_loader, test_loader
