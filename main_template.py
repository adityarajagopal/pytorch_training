import os
import random
import sys

import src.app as applic

import src.param_parser as pp
import src.input_preprocessor as preproc
import src.model_creator as mc
import src.training as training
import src.inference as inference
import src.utils as utils
from src.checkpointing import Checkpointer

import tensorboardX as tbx

import torch
import torch.cuda

import argparse

import getpass

def parse_command_line_args() : 
    parser = argparse.ArgumentParser(description='PyTorch Pruning')
    parser.add_argument('--config-file', default='None', type=str, help='config file with training parameters')
    args = parser.parse_args()
    return args

def main() : 
    # parse config
    print('==> Parsing Config File')
    args = parse_command_line_args()
    
    username = getpass.getuser()

    if args.config_file != 'None' : 
        app = applic.Application(args.config_file)
    else : 
        raise ValueError('Need to specify config file with parameters')

    app.main()

        
main()
         
