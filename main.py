import random
import argparse
import pprint
import torch
import numpy as np
import train
from data_loader import get_loaders

ver_description = {2: "KoBERT cross-entropy",
                   3: "KoBERT contrastive",
                   4: "KoBERT(freeze) contrastvie"
                   }

def define_argparser():
    parser = argparse.ArgumentParser(description='Kibo Document Recommenda (Pytorch)')

    ## version (model)
    parser.add_argument('--version', type=str, default = "dpr_aihbub")

    parser.add_argument('--max_token_len', type=int, default = 128)
    parser.add_argument('--embedding_size', type=int, default = 128)  
    parser.add_argument('--hidden_size', type=int, default = 128)
        
    ## training details
    parser.add_argument('--num_epochs', type=int, default = 20)
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--clip', type=float, default = 5.0)
    parser.add_argument('--learning_rate', type=float, default= 1e-4)
    parser.add_argument('--dropout_p', type=float, default= 0.3)

    parser.add_argument('--random_seed', type=int, default = 20)
    
    ## hyper-parameter of contrastive  learning
    parser.add_argument('--dist_metric', type=str, default= 'euc',
                    help='euclidean-distance / cosine-similarity ')
    
    # data path

    parser.add_argument('--data_path', type=str, default='./data/AIHub/paper_summary/train/paper/csv/paper_train_total_2.csv',
                        help='path for raw text data')  # './data/data_class_6(all).csv   or data_class_5.csv'

    config = parser.parse_args()

    return config

def print_config(config):
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(config))

def set_random_seed(config):
    # set random seed
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed) # for CPU
    torch.cuda.manual_seed(config.random_seed) # for CUDA
    random.seed(config.random_seed) 
    torch.set_default_tensor_type('torch.FloatTensor')


if __name__ == '__main__':
    config = define_argparser() 
    set_random_seed(config)
    print_config(config)

    train_loader, test_loader = get_loaders(config)
    train.initiate(config, train_loader, test_loader)