import argparse
import random

import numpy as np
import torch

from utils.gpu import set_gpu
from utils.parse import parse_yaml

import os
import time
os.environ["TZ"] = "UTC-8"
time.tzset()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def save_logits(output,target,log_path):
    with open(log_path,"w") as fp:
        for x,y in zip(output,target):
            x = [str(round(i,6)) for i in x]
            line = ",".join(x)+","+str(y)+"\n"
            fp.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TPRNet')
    parser.add_argument('--use_cuda', default='true', type=str,
                        help='whether use cuda. default: true')
    parser.add_argument('--gpu', default='all', type=str,
                        help='use gpu device. default: all')
    parser.add_argument('--config', default='cfgs/default.yaml', type=str,
                        help='configuration file. default=cfgs/default.yaml')
    parser.add_argument('--model', default='tprnet_model', type=str,
                        help='choose model. default=tprnet_model')
    parser.add_argument('--net', default='tprnet', type=str,
                        help='choose net. default=tprnet')
    args = parser.parse_args()

    num_gpus = set_gpu(args.gpu)

    config = parse_yaml(args.config)
    network_params = config['network']
    network_params['device'] = "cuda" if str2bool(args.use_cuda) else "cpu"
    network_params['num_gpus'] = num_gpus
    network_params['model_name'] = args.net
    print(args.model)
    if args.model == "tpr_model":
        from models.tpr_model import Model
    else:
        raise NotImplementedError(args.model+" not implemented")
    config['eval']['ckpt_path'] = "path-to-checkpoint.pth"
    model = Model(config)
    # predictions of internal test dataset
    output, target = model._eval(0,"test")
    save_logits(output,target,"internal_predict.txt")
    # predictions of external test dataset
    output, target = model._eval(0,"val")
    save_logits(output,target,"external_predict.txt")


