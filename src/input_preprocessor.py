import torch
import torchvision.datasets 
import torchvision.transforms
import torch.utils.data

from PIL import Image 
import numpy as np

import pickle
import os
import sys

class Preproc(object):
    
    def import_and_preprocess_dataset(self, params) : 
        dataset = params.dataset 
        train_batch = params.train_batch 
        test_batch = params.test_batch 
        workers = params.workers
        data_loc = params.data_location
        
        assert dataset == 'cifar10' or dataset == 'cifar100'or dataset == 'imagenet', 'Dataset has to be one of cifar10, cifar100, imagenet'
        
        print('Preparing dataset %s' % dataset)
    
        # CIFAR10
        if dataset == 'cifar10': 
            (train_loader, valLoader, test_loader) = self.cifar(data_loc, workers, params, 10)
    
        # CIFAR100
        elif dataset == 'cifar100': 
            (train_loader, valLoader, test_loader) = self.cifar(data_loc, workers, params, 100)
        
        # ImageNet
        else : 
            (train_loader, test_loader) = self.imageNet(data_loc, workers, params)
    
        return (train_loader, valLoader, test_loader)
    
    def extract_subclasses(self, subclasses, Y_coarse) : 
        indices = []
        
        for sc in subclasses :
            indices += [i for i in range(len(Y_coarse)) if Y_coarse[i] == sc]
        
        return indices
    
    def create_subclass_dataset(self, dataset, coarseClasses=[]): 
        assert dataset != 'cifar10', 'No subclasses for cifar10'
        if dataset == 'cifar100' : 
            # extract the images for those classes
            YLabels = [self.coarseYNames.index(y) for y in coarseClasses] 
            trainIndices = self.extract_subclasses(YLabels, self.trainCoarseY)
            testIndices = self.extract_subclasses(YLabels, self.testCoarseY)
            
            return (trainIndices, testIndices)

    def extract_cifar_imgs(self, dataLoc, dataSetType):
        # dataLoc : path to directory which has train, test and meta files
        # dataSetType : string of train / test 
        with open(os.path.join(dataLoc, dataSetType), mode='rb') as data : 
            imgs = pickle.load(data, encoding='latin1')        
        
        X = imgs['data']
        fineY = imgs['fine_labels']
        coarseY = imgs['coarse_labels']
        fileNames = imgs['filenames']

        return imgs, X, fineY, coarseY, fileNames
    
    def extract_cifar_meta(self, dataLoc):
        with open(os.path.join(dataLoc, 'meta'), mode = 'rb') as metaData : 
            data = pickle.load(metaData, encoding='latin1')
        
        fineYNames = data['fine_label_names']
        coarseYNames = data['coarse_label_names']

        return data, fineYNames, coarseYNames

    def extract_cifar_data(self, dataset, data_loc):
        if dataset == 'cifar100':
            data_loc = os.path.join(data_loc, 'cifar-100-python')
    
            # training data
            self.trainingImgs, self.trainX, self.trainFineY, self.trainCoarseY, self.trainFilenames = self.extract_cifar_imgs(data_loc, 'train')
            
            # test data
            self.testImgs, self.testX, self.testFineY, self.testCoarseY, self.testFilenames = self.extract_cifar_imgs(data_loc, 'test')
            
            # meta data
            self.metaData, self.fineYNames, self.coarseYNames = self.extract_cifar_meta(data_loc)

    def get_validation_set(self, params, trainIndices):
        fineY = [self.trainFineY[i] for i in trainIndices]
        uniqueClasses = set(fineY)
        dist = {k:[trainIndices[i] for i,y in enumerate(fineY) if y == k] for k in uniqueClasses}
        
        numTrain = int(params.trainValSplit * len(next(iter(dist.values()))))
        trainIndices = []
        valIndices = []
        [trainIndices.extend(v[:numTrain]) for v in dist.values()]
        [valIndices.extend(v[numTrain:]) for v in dist.values()]

        return trainIndices, valIndices

    def get_loaders(self, params, trainSet, testSet, trainIndices=None, testIndices=None):
        if trainIndices is None:
           trainIndices = list(range(len(trainSet)))
        if testIndices is None:
            testIndices = list(range(len(testSet)))

        trainIndices, valIndices = self.get_validation_set(params, trainIndices)

        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=params.train_batch, num_workers=params.workers, sampler = torch.utils.data.sampler.SubsetRandomSampler(trainIndices))
        valLoader = torch.utils.data.DataLoader(trainSet, batch_size=params.test_batch, num_workers=params.workers, sampler = torch.utils.data.sampler.SubsetRandomSampler(valIndices))
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=params.test_batch, num_workers=params.workers, sampler = torch.utils.data.sampler.SubsetRandomSampler(testIndices))

        return trainLoader, valLoader, testLoader        

    def imageNet(self, data_loc, workers, params):
        # data_loc = '/mnt/storage/imagenet_original/data'
        if params.sub_classes != [] : 
            data_loc = self.create_subclass_dataset(params.dataset, data_loc, params.sub_classes) 
        # train_dir = os.path.join('/mnt/storage/imagenet_original/data', 'train')
        # test_dir = os.path.join('/mnt/storage/imagenet_original/data', 'validation')
        train_dir = os.path.join(data_loc, 'train')
        test_dir = os.path.join(data_loc, 'validation')
        num_classes = 1000
            
        train_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
            
        test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
        
        train_set = torchvision.datasets.ImageFolder(train_dir, train_transform)
        test_set = torchvision.datasets.ImageFolder(test_dir, test_transform)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.train_batch, shuffle=True, num_workers=workers)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=params.test_batch, shuffle=False, num_workers=workers)
        
        return (train_loader, test_loader)

    def cifar(self, data_loc, workers, params, cifarIndex):
        # data_loc = '/home/ar4414/multipres_training/organised/data'
        if cifarIndex == 100:
            data_loader = torchvision.datasets.CIFAR100
        elif cifarIndex == 10:
            data_loader = torchvision.datasets.CIFAR10

        self.extract_cifar_data(params.dataset, data_loc)

        if params.sub_classes != []: 
            print('Generating subset of dataset with classes %s' % params.sub_classes)
            train_indices, test_indices = self.create_subclass_dataset(params.dataset, params.sub_classes) 
        else:
            train_indices = None
            test_indices = None
        
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_set = data_loader(root=data_loc, train=True, download=False, transform=train_transform)
        if cifarIndex == 10:
            self.trainFineY = [targets for index, (inputs, targets) in enumerate(train_set)]
        test_set = data_loader(root=data_loc, train=False, download=False, transform=test_transform)

        train_loader, val_loader, test_loader = self.get_loaders(params, train_set, test_set, train_indices, test_indices)
        
        return train_loader, val_loader, test_loader
