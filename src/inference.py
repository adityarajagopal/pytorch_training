import torch.autograd
from tqdm import tqdm
import sys
import math

import src.utils as utils

class Inferer(object):
    def test_network(self, params, test_loader, model, criterion, optimiser, verbose=True) :  
        model.eval()
            
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
    
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc='Inference', leave=verbose) as t:
                for batch_idx, (inputs, targets) in enumerate(test_loader): 
                    device = 'cuda:' + str(params.gpuList[0])
                    inputs, targets = inputs.cuda(device, non_blocking=True),\
                            targets.cuda(device, non_blocking=True)
                    
                    outputs = model(inputs) 
                    loss = criterion(outputs, targets)
                    
                    prec1, prec5 = utils.accuracy(outputs.data, targets.data)
                    losses.update(loss.item()) 
                    top1.update(prec1.item()) 
                    top5.update(prec5.item())

                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg
                    })
                    t.update(1)
        
        return (losses.avg, top1.avg, top5.avg)
