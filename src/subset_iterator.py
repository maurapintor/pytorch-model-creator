import torchvision
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
import torch
import numpy as np


class SubLoaderMNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, include_list=range(10), nb_samples=None, **kwargs):
        super(SubLoaderMNIST, self).__init__(*args, **kwargs)

        if include_list == []:
            raise ValueError("Parameter 'include_list' is an empty list. Select at least one class.")

        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = np.where((labels.reshape(-1, 1) == include).any(axis=1))
        self.data = self.data[mask]
        self.targets = self.targets[mask]
        # Re-enumerate the labels so that they are a range starting from
        # zero and stopping at len(include_list)
        for new_idx, cl in enumerate(include_list):
            self.targets[self.targets == cl] = new_idx
        if nb_samples is not None:
            indices = list(range(len(self.data)))
            np.random.shuffle(indices)
            indices = indices[:nb_samples]
            self.data = self.data[indices]
            self.targets = self.targets[indices]

        self.targets = self.targets.tolist()

        self.classes = [self.classes[i] for i in include_list]



class SubLoaderCIFAR(torchvision.datasets.CIFAR10):
    def __init__(self, *args, include_list=range(10), nb_samples=None, **kwargs):
        super(SubLoaderCIFAR, self).__init__(*args, **kwargs)

        if include_list == []:
            raise ValueError("Parameter 'include_list' is an empty list. Select at least one class.")

        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = np.where((labels.reshape(-1, 1) == include).any(axis=1))[0]
        self.data = np.array(self.data)[mask]
        self.targets = np.array(self.targets)[mask]
        # Re-enumerate the labels so that they are a range starting from
        # zero and stopping at len(include_list)
        for new_idx, cl in enumerate(include_list):
            self.targets[self.targets == cl] = new_idx
        if nb_samples is not None:
            indices = list(range(len(self.data)))
            np.random.shuffle(indices)
            indices = indices[:nb_samples]
            self.data = self.data[indices]
            self.targets = self.targets[indices]

        self.targets = self.targets.tolist()

        self.classes = [self.classes[i] for i in include_list]


class SubsetIterator:

    def __init__(self, nb_train, nb_test, include_list=range(10), batch_size=50, transform_train=None,
                 transform_test=None):
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.nb_train = nb_train
        self.nb_test = nb_test

        if self.transform_train is None:
            self.transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

        if self.transform_test is None:
            self.transform_test = transforms.Compose([
                transforms.ToTensor()
            ])

        # Use only the selected classes
        self.include_list = include_list
        self.batch_size = batch_size

    def get_train_valid_test(self, valid_size=0.2, dataset='mnist'):

        # Download and load the training data
        if dataset == 'mnist':
            trainset = SubLoaderMNIST('~/.pytorch/MNIST/',
                                      download=True,
                                      train=True,
                                      nb_samples=self.nb_train,
                                      transform=self.transform_train,
                                      include_list=self.include_list)
        elif dataset == 'cifar10':
            trainset = SubLoaderCIFAR('~/.pytorch/CIFAR/',
                                      download=True,
                                      train=True,
                                      nb_samples=self.nb_train,
                                      transform=self.transform_train,
                                      include_list=self.include_list)
        else:
            raise NotImplementedError

        trainset_size = len(trainset)
        indices = list(range(trainset_size))
        split = int(np.floor(valid_size * trainset_size))
        np.random.shuffle(indices)
        train_indices, valid_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.batch_size,
                                                  sampler=train_sampler)
        validloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.batch_size,
                                                  sampler=valid_sampler)

        # Download and load the test data
        if dataset == 'mnist':
            testset = SubLoaderMNIST('~/.pytorch/MNIST/',
                                     download=True,
                                     train=False,
                                     nb_samples=self.nb_train,
                                     transform=self.transform_train,
                                     include_list=self.include_list)
        elif dataset == 'cifar10':
            testset = SubLoaderCIFAR('~/.pytorch/CIFAR/',
                                     download=True,
                                     train=False,
                                     nb_samples=self.nb_train,
                                     transform=self.transform_train,
                                     include_list=self.include_list)
        else:
            raise NotImplementedError

        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True)

        return trainloader, validloader, testloader
