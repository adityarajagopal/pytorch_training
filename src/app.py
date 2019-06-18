import src.checkpointing as checkpointingSrc
import src.param_parser as ppSrc
import src.input_preprocessor as preprocSrc
import src.model_creator as mcSrc
import src.training as trainingSrc
import src.inference as inferenceSrc

import os
import random
import sys

import configparser as cp

import tensorboardX as tbx

import torch
import torch.cuda

class Application(object):
    
    def __init__(self, configFile):
        self.setup_param_checkpoint(configFile)
        self.setup_others()

    def main(self):
        self.setup_dataset()
        self.setup_model()
        self.setup_tee_printing()
        
        # setup tensorboardX and checkpointer  
        self.tbx_writer = tbx.SummaryWriter(comment='-test-1')

        if self.params.evaluate == False : 
            self.run_training()
        else : 
            self.run_inference()

        self.tbx_writer.close()

    def run_training(self):
        # train model 
        print('==> Performing Training')
        self.trainer.train_network(self.params, self.tbx_writer, self.checkpointer, self.train_loader, self.valLoader, self.test_loader, self.model, self.criterion, self.optimiser, self.inferer) 

    def run_inference(self):
        # perform inference only
        print('==> Performing Inference')
        self.inferer.test_network(self.params, self.test_loader, self.model, self.criterion, self.optimiser)

    def setup_param_checkpoint(self, configFile):
        config = cp.ConfigParser() 
        config.read(configFile)
        self.params = ppSrc.Params(config)
        self.checkpointer = checkpointingSrc.Checkpointer(self.params, configFile)
        self.setup_params()

    def setup_others(self):
        self.preproc = preprocSrc.Preproc()
        self.mc = mcSrc.ModelCreator()
        self.trainer = trainingSrc.Trainer()
        self.inferer = inferenceSrc.Inferer()

    def setup_dataset(self):
        # setup dataset
        print('==> Setting up Input Dataset')
        self.train_loader, self.valLoader, self.test_loader = self.preproc.import_and_preprocess_dataset(self.params) 
    
    def setup_model(self):
        # setup model 
        print('==> Setting up Model')
        self.model, self.criterion, self.optimiser = self.mc.setup_model(self.params)
        
    def setup_tee_printing(self):
        # setup tee printing so some prints can be written to a log file
        if (self.params.tee_printing != 'None') : 
            print('==> Tee Printing Enabled to logfile {}'.format(self.params.tee_printing))
            sys.stdout = utils.TeePrinting(self.params.tee_printing)

    def setup_params(self):
        self.params = self.checkpointer.restore_state(self.params)
        self.params.use_cuda = torch.cuda.is_available()

        if self.params.use_cuda: 
            print('==> Using GPU %s' % self.params.gpu_id)
            os.environ['CUDA_VISIBLE_DEVICES'] = self.params.gpu_id
            self.params.gpu_list = [int(x) for x in self.params.gpu_id.split(',')]
            if len(self.params.gpu_list) == 1:
                torch.cuda.set_device(self.params.gpu_list[0])
        else : 
            print('==> No CUDA GPUs found --> Using CPU')

        # setup random number generator
        print('==> Setting up random number seed')
        if self.params.manual_seed is None or self.params.manual_seed < 0 : 
            self.params.manual_seed = random.randint(1, 10000)
        random.seed(self.params.manual_seed) 
        torch.manual_seed(self.params.manual_seed)
        if self.params.use_cuda : 
            torch.cuda.manual_seed_all(self.params.manual_seed)
