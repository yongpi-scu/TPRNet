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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TPRNet')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed for training. default=-1')
    parser.add_argument('--use_cuda', default='true', type=str,
                        help='whether use cuda. default: true')
    parser.add_argument('--gpu', default='all', type=str,
                        help='use gpu device. default: all')
    parser.add_argument('--config', default='cfgs/default.yaml', type=str,
                        help='configuration file. default=cfgs/default.yaml')
    parser.add_argument('--model', default='sil_model', type=str,
                        help='choose model. default=tprnet_model')
    parser.add_argument('--net', default='tprnet', type=str,
                        help='choose net. default=tprnet')
    args = parser.parse_args()

    num_gpus = set_gpu(args.gpu)
    if args.seed>0:
        # set seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    config = parse_yaml(args.config)
    network_params = config['network']
    network_params['seed'] = args.seed
    network_params['device'] = "cuda" if str2bool(args.use_cuda) else "cpu"
    network_params['num_gpus'] = num_gpus
    network_params['model_name'] = args.net
    print(args.model)
    if args.model == "tpr_model":
        from models.tpr_model import Model
    else:
        raise NotImplementedError(args.model+" not implemented")
    model = Model(config)
    model.run()