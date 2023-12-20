import os
import glob

import torch
import numpy as np
from easydict import EasyDict
import json

def load_model(args):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/../checkpoints/{args.group}:{args.name}/{args.checkpoint_kind}/*model.ckpt'
    print("::: Loading model from", checkpoint_path)
    checkpoints = glob.glob(checkpoint_path)

    try:
        state_dict = torch.load(checkpoints[-1])
    except Exception as e:
        print("No checkpoints found: ", checkpoint_path)
        raise e

    return state_dict

def load_config(args):
    config_path = f'{os.path.abspath(os.path.dirname(__file__))}/../checkpoints/{args.group}:{args.name}/{args.checkpoint_kind}/*config.json'
    print("::: Loading config from", config_path)
    configs = glob.glob(config_path)

    try:
        with open(configs[-1], 'r') as f:
            config = EasyDict(json.load(f))
    except Exception as e:
        print("No configs found: ", config_path)
        raise e

    return config

def load_model_by_dir(name):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/{name}/*.ckpt'
    print('::: Loading model from:', checkpoint_path)
    checkpoints = glob.glob(checkpoint_path)
    return torch.load(checkpoints[-1])

def load_model_by_name(name, kind = 'best'):
    checkpoint_path = f'{os.path.abspath(os.path.dirname(__file__))}/checkpoints/{name}/{kind}/*.ckpt'
    checkpoints = glob.glob(checkpoint_path)
    return torch.load(checkpoints[-1])

