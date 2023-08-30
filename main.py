
import argparse
from train_fairGNN import call_fairGNN
from b_training1 import call_BIND_training1
import torch

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--paper', type=str, choices=('fairGNN, BIND'), default='BIND')
parser.add_argument('--dataset', type=str, default='nba', choices=['pokec_z','pokec_n','nba', 'bail'])
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

# parser.add_argument("--size", choices=["S", "M", "L", "XL"], default="M")
# python size.py --size S

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)
print('PAPER CALLED: ', args.paper)
if args.paper == 'fairGNN': 
    call_fairGNN()
elif args.paper == 'BIND': 
    call_BIND_training1() 