import models 
import torch.nn
import torch.backends
import torchvision 

class ModelCreator(object):
    
    def setup_model(self, params) : 
        model = self.read_model(params)
        model = self.transfer_to_gpu(params, model)
        model = self.load_pretrained(params, model)
        criterion = self.setup_criterion()
        optimiser = self.setup_optimiser(params, model)
    
        return (model, criterion, optimiser)
    
    def read_model(self, params):
        if params.dataset == 'cifar10' : 
            import models.cifar as models 
            num_classes = 10
    
        elif params.dataset == 'cifar100' : 
            import models.cifar as models 
            num_classes = 100
    
        else : 
            import models.imagenet as models 
            num_classes = 1000
    
        print("Creating Model %s" % params.arch)
        
        if params.arch.endswith('resnet'):
            model = models.__dict__[params.arch](
                        num_classes=num_classes,
                        depth=params.depth
                    )
        else:
            model = models.__dict__[params.arch](num_classes=num_classes)

        return model

    def transfer_to_gpu(self, params, model):
        gpu_list = [int(x) for x in params.gpu_id.split(',')]
        model = torch.nn.DataParallel(model, gpu_list)
        model = model.cuda()

        return model
    
    def load_pretrained(self, params, model):
        if params.resume == True or params.branch == True : 
            checkpoint = torch.load(params.pretrained)
            model.load_state_dict(checkpoint)
    
        if params.evaluate == True : 
            checkpoint = torch.load(params.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
            
        torch.backends.cudnn.benchmark = True
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

        return model
    
    def setup_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def setup_optimiser(self, params, model):
        return torch.optim.SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)

    



