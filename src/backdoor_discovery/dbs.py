from asyncio.log import logger
import torch 
import numpy as np 
import os 
import random 
import warnings 
import time 
import yaml 
import argparse 
import sys 
import logging 
import json

from src.backdoor_discovery.utils.logger import Logger
from src.backdoor_discovery.utils.utils import load_models, enumerate_trigger_options, load_data
from src.backdoor_discovery.scanner import DBS_Scanner 


warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def dbs(model_filepath, tokenizer_filepath, examples_dirpath, config_name='config', scratch_dirpath='./scratch'):

    start_time = time.time()

    logging_filepath = os.path.join(scratch_dirpath + '.log')
    logger = Logger(logging_filepath, logging.DEBUG, logging.DEBUG)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load config 
    config_filepath = f'./src/backdoor_discovery/configs/{config_name}.yaml'
    with open(config_filepath) as f: 
        config = yaml.safe_load(f)

    # fix seed
    seed_torch(config['seed'])

    target_model,benign_model,tokenizer = load_models(model_filepath, tokenizer_filepath, device)
    
    target_model.eval() 
    benign_model.eval()

    trigger_options = enumerate_trigger_options()

    scanning_result_list = [] 

    best_loss = 1e+10

    for trigger_opt in trigger_options:
        position = trigger_opt['position']
        special_token = trigger_opt['special_token']

        prompts, targets = load_data(examples_dirpath)

        scanner = DBS_Scanner(target_model,benign_model,tokenizer,device,logger,config)

        trigger,loss = scanner.generate(prompts,targets,position)
        
        scanning_result = {'position': position, 'trigger':trigger, 'loss': loss}
        scanning_result_list.append(scanning_result)

        if loss <= best_loss:
            best_loss = loss 
            best_estimation = scanning_result
    
    for scanning_result in scanning_result_list:
        logger.result_collection('position: {}  trigger: {}  loss: {:.6f}'.format(scanning_result['position'], scanning_result['trigger'],scanning_result['loss']))
    
    end_time = time.time()
    scanning_time = end_time - start_time
    logger.result_collection('Scanning Time: {:.2f}s'.format(scanning_time))

    logger.best_result('position: {}  trigger: {}  loss: {:.6f}'.format(best_estimation['position'], best_estimation['trigger'],best_estimation['loss']))
    
    return best_estimation, scanning_time
        







