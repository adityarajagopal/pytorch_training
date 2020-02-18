import torch
import torchvision.datasets 
import torchvision.transforms
import torch.utils.data

from PIL import Image 
import numpy as np

import pickle
import os
import sys
import csv
import json

class Preproc(object):
    
    def import_and_preprocess_dataset(self, params) : 
        #{{{
        dataset = params.dataset 
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
            (train_loader, valLoader, test_loader) = self.imageNet(data_loc, workers, params)
    
        return (train_loader, valLoader, test_loader)
        #}}}
    
    def extract_subclasses(self, subclasses, Y_coarse) : 
        #{{{
        indices = []
        
        for sc in subclasses :
            indices += [i for i in range(len(Y_coarse)) if Y_coarse[i] == sc]
        
        return indices
        #}}}
    
    def create_subclass_dataset(self, dataset, coarseClasses=[]): 
        #{{{
        if dataset == 'cifar100' : 
            # extract the images for those classes
            YLabels = [self.coarseYNames.index(y) for y in coarseClasses] 
            trainIndices = self.extract_subclasses(YLabels, self.trainCoarseY)
            testIndices = self.extract_subclasses(YLabels, self.testCoarseY)

        elif dataset == 'cifar10':
            YLabels = [self.coarseYNames.index(y) for y in coarseClasses] 
            trainIndices = self.extract_subclasses(YLabels, self.trainCoarseY)
            testIndices = self.extract_subclasses(YLabels, self.testCoarseY)
        
        return (trainIndices, testIndices)
        #}}}

    def extract_cifar_imgs(self, dataLoc, dataSetType, classes):
        #{{{
        # dataLoc : path to directory which has train, test and meta files
        # dataSetType : string of train / test 
        if classes == 100:
            with open(os.path.join(dataLoc, dataSetType), mode='rb') as data : 
                imgs = pickle.load(data, encoding='latin1')        
            
            X = imgs['data']
            fineY = imgs['fine_labels']
            coarseY = imgs['coarse_labels']
            fileNames = imgs['filenames']

        elif classes == 10:
            if dataSetType == 'train':
                for i in range(1,6):
                    with open(os.path.join(dataLoc, "data_batch_{}".format(i)), mode='rb') as data : 
                        imgs = pickle.load(data, encoding='latin1')        
                    
                    if i == 1:
                        X = imgs['data']
                        fileNames = imgs['filenames']
                        fineY = imgs['labels']
                    else:
                        np.append(X, imgs['data'])
                        np.append(fileNames, imgs['filenames'])
                        np.append(fineY, imgs['labels'])
            else:
                with open(os.path.join(dataLoc, "test_batch"), mode='rb') as data : 
                    imgs = pickle.load(data, encoding='latin1')        
                
                X = imgs['data']
                fileNames = imgs['filenames']
                fineY = imgs['labels']

            coarseY = fineY

        else:
            raise Exception('Invalid number of classes in "input_preprocessor.py - extract_cifar_imgs')
        
        return imgs, X, fineY, coarseY, fileNames
        #}}}
    
    def extract_cifar_meta(self, dataLoc, classes):
        #{{{
        if classes == 100:
            with open(os.path.join(dataLoc, 'meta'), mode = 'rb') as metaData : 
                data = pickle.load(metaData, encoding='latin1')
            
            fineYNames = data['fine_label_names']
            coarseYNames = data['coarse_label_names']
        elif classes == 10:
            with open(os.path.join(dataLoc, 'batches.meta'), mode = 'rb') as metaData : 
                data = pickle.load(metaData, encoding='latin1')
            
            coarseYNames = data['label_names']
            fineYNames = coarseYNames            

        else:
            raise Exception('Invalid number of classes in "input_preprocessor.py - extract_cifar_meta')

        return data, fineYNames, coarseYNames
        #}}}

    def extract_cifar_data(self, dataset, data_loc):
        #{{{
        if dataset == 'cifar100':
            data_loc = os.path.join(data_loc, 'cifar-100-python')
    
            # training data
            self.trainingImgs, self.trainX, self.trainFineY, self.trainCoarseY, self.trainFilenames = self.extract_cifar_imgs(data_loc, 'train', 100)
            
            # test data
            self.testImgs, self.testX, self.testFineY, self.testCoarseY, self.testFilenames = self.extract_cifar_imgs(data_loc, 'test', 100)
            
            # meta data
            self.metaData, self.fineYNames, self.coarseYNames = self.extract_cifar_meta(data_loc, 100)

        elif dataset == 'cifar10':
            data_loc = os.path.join(data_loc, 'cifar-10-batches-py')
    
            # training data
            self.trainingImgs, self.trainX, self.trainFineY, self.trainCoarseY, self.trainFilenames = self.extract_cifar_imgs(data_loc, 'train', 10)
            
            # test data
            self.testImgs, self.testX, self.testFineY, self.testCoarseY, self.testFilenames = self.extract_cifar_imgs(data_loc, 'test', 10)
           
            # meta data
            self.metaData, self.fineYNames, self.coarseYNames = self.extract_cifar_meta(data_loc, 10)
        #}}}

    def get_validation_set(self, params, trainIndices):
        #{{{
        fineY = [self.trainFineY[i] for i in trainIndices]
        uniqueClasses = set(fineY)
        dist = {k:[trainIndices[i] for i,y in enumerate(fineY) if y == k] for k in uniqueClasses}
        numTrain = int(params.trainValSplit * len(next(iter(dist.values()))))
        trainIndices = []
        valIndices = []
        [trainIndices.extend(v[:numTrain]) for v in dist.values()]
        [valIndices.extend(v[numTrain:]) for v in dist.values()]

        return trainIndices, valIndices
        #}}}

    def get_loaders(self, params, trainSet, testSet, trainIndices=None, testIndices=None):
    #{{{
        if trainIndices is None:
            trainIndices = list(range(len(trainSet)))
        if testIndices is None:
            testIndices = list(range(len(testSet)))
        
        if params.sub_classes != []: 
        #{{{
            print('Generating subset of dataset with classes %s' % params.sub_classes)
            trainIndFileName = 'coarseTrainIndices_' + str(params.trainValSplit).replace('.','_') + '.json'
            valIndFileName = 'coarseValIndices_' + str(params.trainValSplit).replace('.','_') + '.json'

            if params.dataset == 'cifar100':
                #{{{
                trainIndFile = os.path.join(params.data_location, 'cifar-100-python', trainIndFileName)
                valIndFile = os.path.join(params.data_location, 'cifar-100-python', valIndFileName)
                
                if not os.path.isfile(trainIndFile) or not os.path.isfile(valIndFile):
                    #{{{
                    print('Creating JSON with per coarse class train and val indices')
                    ccTrain = {cc:[] for cc in self.coarseYNames}
                    ccVal = {cc:[] for cc in self.coarseYNames}
                    for sc in self.coarseYNames:
                        train_indices, test_indices = self.create_subclass_dataset(params.dataset, [sc]) 
                        train, val = self.get_validation_set(params, train_indices)
                        ccTrain[sc] = train
                        ccVal[sc] = val
                    with open(trainIndFile, 'w') as tFile: 
                        json.dump(ccTrain, tFile)
                    with open(valIndFile, 'w') as vFile:
                        json.dump(ccVal, vFile)
                    #}}}
                
                else:
                    with open(trainIndFile, 'r') as tF:
                        ccTrainIndices = json.load(tF)
                    with open(valIndFile, 'r') as vF:
                        ccValIndices = json.load(vF)
                    
                    trainIndices = []
                    valIndices = []
                    for cc in params.sub_classes:
                        trainIndices += ccTrainIndices[cc]
                        valIndices += ccValIndices[cc]
                    self.trainIndices = trainIndices 
                    self.valIndices = valIndices
                        
                    _, testIndices = self.create_subclass_dataset(params.dataset, params.sub_classes) 
                #}}}
            else:
                raise ValueError('Sub Class Extraction not implemented for {}'.format(params.dataset))
        #}}}
            
        elif ('cifar' in params.dataset) or ('imagenet' in params.dataset):
        #{{{
            trainIndFileName = 'trainIndices_' + str(params.trainValSplit).replace('.','_') + '.csv'
            valIndFileName = 'valIndices_' + str(params.trainValSplit).replace('.','_') + '.csv'

            # if 'cifar10' in params.dataset:
            if params.dataset == 'cifar10':
                trainIndFile = os.path.join(params.data_location, 'cifar-10-batches-py', trainIndFileName)
                valIndFile = os.path.join(params.data_location, 'cifar-10-batches-py', valIndFileName)
            elif params.dataset == 'cifar100':
                trainIndFile = os.path.join(params.data_location, 'cifar-100-python', trainIndFileName)
                valIndFile = os.path.join(params.data_location, 'cifar-100-python', valIndFileName)
            else:
                # case of imagenet
                trainIndFile = os.path.join(params.data_location, trainIndFileName)
                valIndFile = os.path.join(params.data_location, valIndFileName)

            with open(trainIndFile, 'r') as csvFile:
                csvReader = csv.reader(csvFile, delimiter=',')
                trainIndices = next(csvReader)
                trainIndices = [int(x) for x in trainIndices]
                self.trainIndices = trainIndices
            
            with open(valIndFile, 'r') as csvFile:
                csvReader = csv.reader(csvFile, delimiter=',')
                valIndices = next(csvReader)
                valIndices = [int(x) for x in valIndices]
                self.valIndices = valIndices
        #}}}
        
        else:
            # never taken at the moment
            self.trainIndices, self.valIndices = self.get_validation_set(params, trainIndices)
        
        self.testIndices = testIndices

        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=params.train_batch, num_workers=params.workers, sampler = torch.utils.data.sampler.SubsetRandomSampler(trainIndices))
        valLoader = torch.utils.data.DataLoader(trainSet, batch_size=params.test_batch, num_workers=params.workers, sampler = torch.utils.data.sampler.SubsetRandomSampler(valIndices))
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=params.test_batch, num_workers=params.workers, sampler = torch.utils.data.sampler.SubsetRandomSampler(testIndices))

        return trainLoader, valLoader, testLoader        
    #}}}

    def imageNet(self, dataLoc, workers, params):
    #{{{
        if params.sub_classes != []: 
            print('Generating subset of dataset with classes %s' % params.sub_classes)
            trainIndices, testIndices = self.create_subclass_dataset(params.dataset, params.sub_classes) 
        else:
            trainIndices = None
            testIndices = None
        
        trainDir = os.path.join(dataLoc, 'train')
        testDir = os.path.join(dataLoc, 'validation')
        numClasses = 1000
            
        trainTransform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
            
        testTransform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
        
        trainSet = torchvision.datasets.ImageFolder(trainDir, trainTransform)
        testSet = torchvision.datasets.ImageFolder(testDir, testTransform)

        trainLoader, valLoader, testLoader = self.get_loaders(params, trainSet, testSet, trainIndices, testIndices)

        return (trainLoader, valLoader, testLoader)
    #}}}

    def cifar(self, data_loc, workers, params, cifarIndex):
    #{{{
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
        
        # train_loader, val_loader, test_loader = self.get_loaders(params, train_set, test_set, train_indices, test_indices)
        train_loader, val_loader, test_loader = self.get_loaders(params, train_set, test_set)
        
        return train_loader, val_loader, test_loader
    #}}}
