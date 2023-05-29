import torch

import os
import numpy as np
from numpy import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn

from datetime import datetime
import argparse


import copy

from data import get_loader, get_loader1
import argparse
import torch
import torch.backends.cudnn as cudnn
# from torchvision import models
from Mixup import Mixup

from data import get_loader, get_loader2, get_loader3,get_loader_mix_all,get_loader_subject_out,get_loader_DEAP_SEED



# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                          ' | '.join(model_names) +
#                          ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
args = parser.parse_args()

args = parser.parse_args()
assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
# check if gpu training is available
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1

def callback(model, loss):
  n = model.n_epochs
  if n % 1 == 0:

    if loss < model.min_loss:
      model.min_loss = loss
      # model.save(f'test_run/model_{n}.pkl')
      # model.save("test_run/base_model.pkl")
  return callback

####################################################

# Load Model
# model = base_Model(configs).to(device)
# temporal_contr_model = TC(configs, device).to(device)
#
# optimizer = torch.optim.AdamW(list(model.parameters()) + list(temporal_contr_model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
#
# scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0, warm_epochs=configs.warm_epochs, gamma=configs.gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)
# t = time.time()

cur_dim = 'valence'
data_file = 'all32'
# fold = 4
# num_labeled = 0.4
# batch_labeled = 40
# batch_unlabeled = 100

fold = 4
num_labeled = 1
batch_labeled = 100
batch_unlabeled = 100

# fold = 4
# num_labeled = 0.1
# batch_labeled = 10
# batch_unlabeled = 100

# fold = 5 #P19
# num_labeled = 0.1
# batch_labeled = 10
# batch_unlabeled = 100

# fold = 4 #Epilepsy
# num_labeled = 0.1
# batch_labeled = 6
# batch_unlabeled = 60

# fold = 4 #HAR
# num_labeled = 0.1
# batch_labeled = 6
# batch_unlabeled = 60

# sourcedata_path = '/mnt/disk1/data0/caihlFiles/chl/ts2vec-revised20230408/Epilepsy'
# sourcedata_path = '/mnt/disk1/data0/caihlFiles/chl/ts2vec-revised20230408/HAR'

# dict_performance_fold = {}
loop=2
loop_mean_acc = []
loop_mean_pre = []
loop_mean_recall = []
loop_mean_f1 = []
loop_mean_auroc = []
loop_mean_auprc = []
# t = time.time()
start = datetime.now()

for looptime in range(loop):
    # dict_performance_fold = {}
    dict_performance = {}
    for curr_fold in range(fold):

        # curr_fold = curr_fold + 1  # P19


        # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader(cur_dim, curr_fold, data_file,
        #                                                                                num_labeled, batch_labeled,
        #                                                                                batch_unlabeled)

        train_labeled_loader, train_unlabeled_loader, test_loader = get_loader_DEAP_SEED(cur_dim, curr_fold, data_file,
                                                                               num_labeled, batch_labeled,
                                                                               batch_unlabeled)


        # train_unlabeled_loader, test_loader = get_loader1(cur_dim, curr_fold, data_file)

        # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader_mix_all(cur_dim, curr_fold, data_file,
        #                                                                                num_labeled, batch_labeled,
        #                                                                                batch_unlabeled)

        # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader_subject_out(cur_dim, curr_fold, data_file,
        #                                                                                num_labeled, batch_labeled,
        #                                                                                batch_unlabeled)

        # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader2('P19', curr_fold, num_labeled,
        #                                                                         batch_labeled, batch_unlabeled)

        # train_labeled_loader, train_unlabeled_loader, test_loader \
        #     = get_loader3(sourcedata_path, curr_fold, num_labeled, batch_labeled, batch_unlabeled)

        model = Mixup()
        # model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

        # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

        # model = TS_TCC(after_epoch_callback=callback)

        model.min_loss = float('inf')

        loss_log = model.fit(train_unlabeled_loader,verbose=True,n_epochs=1)

        model_copy = copy.deepcopy(model)

        loss, performance = model_copy.finetune_fit(train_labeled_loader, test_loader, finetune_epochs=1,finetune_lr=0.0001)

        print(f"当前fold: {curr_fold}")
        print(f"Loss: {loss}")
        print(f"Performance: {performance}")

        dict_performance.setdefault('Accuracy', []).append(performance['Accuracy'])
        dict_performance.setdefault('Precision', []).append(performance['Precision'])
        dict_performance.setdefault('Recall', []).append(performance['Recall'])
        dict_performance.setdefault('F1', []).append(performance['F1'])
        dict_performance.setdefault('AUROC', []).append(performance['AUROC'])
        dict_performance.setdefault('AUPRC', []).append(performance['AUPRC'])

        # scheduler.step()

    # print(f"当前维度: {cur_dim}")
    # print(f"Accuracy: {mean(dict_performance['Accuracy'])}")
    # print(f"Precision: {mean(dict_performance['Precision'])}")
    # print(f"Recall: {mean(dict_performance['Recall'])}")
    # print(f"F1: {mean(dict_performance['F1'])}")
    # print(f"AUROC: {mean(dict_performance['AUROC'])}")
    # print(f"AUPRC: {mean(dict_performance['AUPRC'])}")

    fold_mean_acc = mean(dict_performance['Accuracy'])
    fold_mean_pre = mean(dict_performance['Precision'])
    fold_mean_recall = mean(dict_performance['Recall'])
    fold_mean_f1 = mean(dict_performance['F1'])
    fold_mean_auroc = mean(dict_performance['AUROC'])
    fold_mean_auprc = mean(dict_performance['AUPRC'])

    loop_mean_acc.append(fold_mean_acc)
    loop_mean_pre.append(fold_mean_pre)
    loop_mean_recall.append(fold_mean_recall)
    loop_mean_f1.append(fold_mean_f1)
    loop_mean_auroc.append(fold_mean_auroc)
    loop_mean_auprc.append(fold_mean_auprc)

print(f"平均值：")
print(f"Accuracy: {loop_mean_acc, mean(loop_mean_acc), std(loop_mean_acc)}")
print(f"Precision: {loop_mean_pre, mean(loop_mean_pre), std(loop_mean_pre)}")
print(f"Recall: {loop_mean_recall, mean(loop_mean_recall), std(loop_mean_recall)}")
print(f"F1: {loop_mean_f1, mean(loop_mean_f1), std(loop_mean_f1)}")
print(f"AUROC: {loop_mean_auroc, mean(loop_mean_auroc), std(loop_mean_auroc)}")
print(f"AUPRC: {loop_mean_auprc, mean(loop_mean_auprc), std(loop_mean_auprc)}")

end = datetime.now()
print('Running time: %s Seconds' % (end - start))
print("Finished.")
