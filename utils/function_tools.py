import os
import json
import logging
import torch
import numpy as np
import torch.nn as nn
from argparse import Namespace

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    
def save_config(cfg,save_dir,fname=None):
    if isinstance(cfg,Namespace):
        cfg_dict = vars(cfg)
    elif isinstance(cfg,object):
        cfg_dict = {}
        cfg_dict.update(cfg.__dict__)
    else:
        assert isinstance(cfg,dict)
        cfg_dict = cfg

    if fname is None:
        fpath = os.path.join(save_dir,'config.json')
    else:
        assert fname.endswith(".json")
        fpath = os.path.join(save_dir,fname)
    
    with open(fpath,"w") as output:
        json.dump(cfg_dict,output,indent=4)

def load_config(load_dir,to_Namespace=True):
    files = os.listdir(load_dir)
    flist = list(filter(lambda x:x.endswith(".json"),files))
    try:
        assert len(flist) == 1
    except:
        print(f"Existing mmultiple cfg files:{flist}",flush=True)

    fname = flist[0]
    with open(os.path.join(load_dir,fname),"r") as input:
        cfg = json.load(input)
    
    assert isinstance(cfg,dict)

    if to_Namespace:
        cfg = Namespace(**cfg)
    
    return cfg

def get_logger(logger_name,logger_dir=None,log_name=None,is_mute_logger=False):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear() 

    if is_mute_logger:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    hterm = logging.StreamHandler()
    hterm.setFormatter(formatter)
    hterm.setLevel(logging.INFO)
    logger.addHandler(hterm)

    if logger_dir is not None:
        if log_name is None:
            logger_path = os.path.join(logger_dir,f"{logger_name}.log")
        else:
            logger_path = os.path.join(logger_dir,log_name)
        hfile = logging.FileHandler(logger_path) 
        hfile.setFormatter(formatter)
        hfile.setLevel(logging.INFO)
        logger.addHandler(hfile)
    return logger

def get_device(dev):
    if torch.cuda.is_available():
        return torch.device(dev)
    else:
        return torch.device("cpu")

def boolean_string(s):
    if s not in {'False','True','0','1'}:
        raise ValueError('Not a valid boolean string.')
    return (s=='True') or (s=='1')

def weight_init(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias,0)
    elif isinstance(m,nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight,mode='fan_out',nonlinearity='relu')
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)