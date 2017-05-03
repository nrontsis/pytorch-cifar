import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os

from torchnet.dataset import SplitDataset


def create_loaders(args):

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}

    dataset = datasets.CIFAR10 if args.dataset == 'cifar10' \
        else datasets.CIFAR100
    root = './data'

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    dataset_no_split = dataset(root=root, train=True, transform=transform_train)

    split_dataset = SplitDataset(dataset_no_split, partitions={'train': 0.9, 'val': 0.1})

    split_dataset.select('train')
    train_loader = data.DataLoader(split_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True, **kwargs)

    split_dataset.select('val')
    val_loader = data.DataLoader(split_dataset,
                                 batch_size=args.test_batch_size,
                                 shuffle=False, **kwargs)

    test_loader = data.DataLoader(dataset(root=root, train=False,
                                          transform=transform_test),
                                  batch_size=args.test_batch_size,
                                  shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
