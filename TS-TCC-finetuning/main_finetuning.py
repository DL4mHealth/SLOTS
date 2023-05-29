import torch

import os
import numpy as np
from numpy import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from models.loss import NTXentLoss
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
from dataloader.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import *
import copy
from utils import _calc_metrics, copy_Files
from models.model import base_Model
from utils import WarmUpExponentialLR
sys.path.append("/mnt/disk1/data0/chlFiles/TS-TCC-main/dataloader/augmentations.py")
from dataloader.augmentations import DataTransform

from TS_TCC import TS_TCC
# from data import get_loader, get_loader1
from data import get_loader, get_loader2, get_loader3,get_loader_DEAP_SEED
# import warnings
# warnings.simplefilter("error")

import sys
# sys.path.append("/mnt/disk1/data0/chlFiles/TS-TCC-main/config_files")
sys.path.append("/data0/caihlFiles/TS_TCC_20230314/config_files")
# Args selections
from config_files.Epilepsy_Configs import Config as Configs


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--gpu', type=int, default=2,
                        help='The gpu no. used for training and inference (defaults to 0)')
args = parser.parse_args()


device = init_dl_program(args.gpu)
# device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = 'supervised'
# training_mode = args.training_mode
run_description = args.run_description

def callback(model, loss):
  n = model.n_epochs
  if n % 1 == 0:

    if loss < model.min_loss:
      model.min_loss = loss
      # model.save(f'test_run/model_{n}.pkl')
      # model.save("test_run/base_model.pkl")
  return callback

from config_files.HAR_Configs import Config as Configs
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

# Load Model
# model = base_Model(configs).to(device)
# temporal_contr_model = TC(configs, device).to(device)
#
# optimizer = torch.optim.AdamW(list(model.parameters()) + list(temporal_contr_model.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
#
# scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0, warm_epochs=configs.warm_epochs, gamma=configs.gamma)

# t = time.time()

cur_dim = 'valence'
data_file = 'all32'
# fold = 4
# num_labeled = 0.1
# batch_labeled = 10
# batch_unlabeled = 50

fold = 4
num_labeled = 1
batch_labeled = 100
batch_unlabeled = 100

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

# sourcedata_path = '/data0/caihlFiles/Epilepsy'
sourcedata_path = '/data0/caihlFiles/HAR'

dict_performance_fold = {}
# t = time.time()
# start = datetime.now()
# dict_performance = {}

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

        # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader(cur_dim, curr_fold, data_file,
        #                                                                                num_labeled, batch_labeled,
        #                                                                                batch_unlabeled)

        train_labeled_loader, train_unlabeled_loader, test_loader = get_loader_DEAP_SEED(cur_dim, curr_fold, data_file,
                                                                               num_labeled, batch_labeled,
                                                                               batch_unlabeled)



        # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader2('P19', curr_fold, num_labeled,
        #                                                                         batch_labeled, batch_unlabeled)

        train_labeled_loader, train_unlabeled_loader, test_loader \
            = get_loader3(sourcedata_path, curr_fold, num_labeled, batch_labeled, batch_unlabeled)

        # train_unlabeled_loader, test_loader = get_loader1(cur_dim, curr_fold, data_file)

        model = TS_TCC(after_epoch_callback=callback)

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

# print(f"当前维度: {cur_dim}")
# print(f"Accuracy: {mean(dict_performance['Accuracy'])}")
# print(f"Precision: {mean(dict_performance['Precision'])}")
# print(f"Recall: {mean(dict_performance['Recall'])}")
# print(f"F1: {mean(dict_performance['F1'])}")
# print(f"AUROC: {mean(dict_performance['AUROC'])}")
# print(f"AUPRC: {mean(dict_performance['AUPRC'])}")
#
# end = datetime.now()
# print('Running time: %s Seconds' % (end - start))
# print("Finished.")
