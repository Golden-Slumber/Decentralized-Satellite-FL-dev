import os
import argparse

parser = argparse.ArgumentParser(description='SNN training')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--dataset', default='EuroSAT', type=str)
parser.add_argument('--train_batch_size', default=64, type=int)
parser.add_argument('--test_batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--width_lr', default=0.001, type=float)
parser.add_argument('--bn_type', default='tdbn', type=str)
parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--optim', default='SGDM', type=str)
parser.add_argument('--act', default='mns_sig', type=str)
parser.add_argument('--alpha', default=0.2, type=float)

parser.add_argument('--granularity', default='layer', type=str)
parser.add_argument('--decay', default=0.5, type=float)
parser.add_argument('--thresh', default=1.0, type=float)
parser.add_argument('--p', default=0.2, type=float)
parser.add_argument('--gamma', default=1.0, type=float)
parser.add_argument('--train_decay', default=False, action='store_true')
parser.add_argument('--train_thresh', default=False, action='store_true')
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--T', default=2, type=int, help='num of time steps')
parser.add_argument('--log_interval', default='20', type=int)

parser.add_argument('--means', default=1.0, type=float, metavar='N',
                    help='make all the potential increment around the means')
parser.add_argument('--lamb', default=1e-3, type=float, metavar='N', help='adjust the norm factor to avoid outlier')
parser.add_argument('--train_width', default=True, action='store_true')
parser.add_argument('--ns_milestone', default=[0, 0.2, 0.4, 0.6, 0.8, 0.95], type=float, nargs='*', )

args = parser.parse_args()
