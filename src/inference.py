import torch.autograd
from tqdm import tqdm
import sys
import math

import src.utils as utils

class Inferer(object):
    def test_network(self, params, test_loader, model, criterion, optimiser) :  
        model.eval()
            
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
    
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader), total=len(test_loader)-1, desc='inference', leave=False) : 
            # move inputs and targets to GPU
            with torch.no_grad():
                device = 'cuda:' + str(params.gpuList[0])
                if params.use_cuda : 
                    inputs, targets = inputs.cuda(device, non_blocking=True), targets.cuda(device, non_blocking=True)
                
                # perform inference 
                outputs = model(inputs) 
                loss = criterion(outputs, targets)
            
            prec1, prec5 = utils.accuracy(outputs.data, targets.data)
    
            losses.update(loss.item()) 
            top1.update(prec1.item()) 
            top5.update(prec5.item())
    
        if params.evaluate == True : 
            tqdm.write('Loss: {}, Top1: {}, Top5: {}'.format(losses.avg, top1.avg, top5.avg))
        
        return (losses.avg, top1.avg, top5.avg)
