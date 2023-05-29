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
from utils import _calc_metrics, copy_Files
from models.model import base_Model
from utils import WarmUpExponentialLR,Classifier
sys.path.append("/data0/caihlFiles/TS_TCC_20230314/dataloader/augmentations.py")
from dataloader.augmentations import DataTransform
from supervised_contrastive_loss import SupConLoss
from data import get_loader, get_loader2, get_loader3

from sklearn import metrics
from models.model import base_Model,ProjectionHead
from models.TC import TC
from models.loss import NTXentLoss
# import warnings
# warnings.simplefilter("error")

import sys
sys.path.append("/data0/caihlFiles/TS_TCC_20230314/config_files")
# Args selections
from config_files.Epilepsy_Configs import Config as Configs
start_time = datetime.now()


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
parser.add_argument('--gpu', type=int, default=1,
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

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


from config_files.HAR_Configs import Config as Configs
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

# experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
# os.makedirs(experiment_log_dir, exist_ok=True)
#
# # loop through domains
# counter = 0
# src_counter = 0


# Logging
# log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# logger = _logger(log_file_name)
# logger.debug("=" * 45)
# logger.debug(f'Dataset: {data_type}')
# logger.debug(f'Method:  {method}')
# logger.debug(f'Mode:    {training_mode}')
# logger.debug("=" * 45)

# Load datasets

# data_path = f"/mnt/disk1/data0/chlFiles/TS-TCC-main/data_preprocessing/epilepsy/data/"
#
# # data_path = f"./data/{data_type}"
#
# train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
# logger.debug("Data loaded ...")
#
# # Load Model
# model = base_Model(configs).to(device)
# temporal_contr_model = TC(configs, device).to(device)
#
#
# model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
# temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
#
# # Trainer
# Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)
#
# if training_mode != "self_supervised":
#     # Testing
#     outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
#     total_loss, total_acc, pred_labels, true_labels = outs
#     _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
#
# logger.debug(f"Training time is : {datetime.now()-start_time}")

#SEED
# def hotEncoder(v):
#     ret_vec = torch.zeros(v.shape[0], 3).to(device)
#     for s in range(v.shape[0]):
#         ret_vec[s][v[s]] = 1
#     return ret_vec

# P19 Epilepsy
def hotEncoder(v):
    ret_vec = torch.zeros(v.shape[0], 2).to(device)
    for s in range(v.shape[0]):
        ret_vec[s][v[s]] = 1
    return ret_vec

#HAR
# def hotEncoder(v):
#     ret_vec = torch.zeros(v.shape[0], 6).to(device)
#     for s in range(v.shape[0]):
#         ret_vec[s][v[s]] = 1
#     return ret_vec

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    ls = nn.CrossEntropyLoss()(input, labels).to(device)
    return ls

# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)


# classifier = ProjectionHead(input_dims=2304, output_dims=2, hidden_dims=128).to(device)#DEAP
# classifier = ProjectionHead(input_dims=3456, output_dims=3, hidden_dims=128).to(device)#SEED
classifier = ProjectionHead(input_dims=1408, output_dims=2, hidden_dims=128).to(device)  # P19
# classifier = ProjectionHead(input_dims=3072, output_dims=2, hidden_dims=128).to(device)  #Epilepsy
# classifier = ProjectionHead(input_dims=3072, output_dims=6, hidden_dims=128).to(device)  #HAR

# y_labeled = F.one_hot(y_labeled, num_classes=2).to(torch.float)
# y_labeled = F.one_hot(y_labeled, num_classes=3).to(torch.float)  # SEED
# y_labeled = F.one_hot(y_labeled, num_classes=2).to(torch.float)  # P19
# y_labeled = F.one_hot(y_labeled, num_classes=2).to(torch.float)  # Epilepsy
# y_labeled = F.one_hot(y_labeled, num_classes=6).to(torch.float)  # HAR

# classifier = ProjectionHead(input_dims=320, output_dims=6, hidden_dims=128).to(device)
        # optimizer = torch.optim.AdamW(self.finetune_model.parameters(), lr=self.lr)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(list(model.parameters()) + list(temporal_contr_model.parameters()) + list(classifier.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

# scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0, warm_epochs=configs.warm_epochs, gamma=configs.gamma)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)

contrastiveLoss = SupConLoss(temperature=0.1)
# t = time.time()

cur_dim = 'valence'
data_file = 'all32'
# fold = 4
# num_labeled = 1
# batch_labeled = 100
# batch_unlabeled = 100

fold = 5 #P19
num_labeled = 0.1
batch_labeled = 10
batch_unlabeled = 100

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
    dict_performance_fold = {}
    for curr_fold in range(fold):

        curr_fold = curr_fold + 1  # P19

        dict_performance = {}

        # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader(cur_dim, curr_fold, data_file,
        #                                                                                num_labeled, batch_labeled,
        #                                                                                batch_unlabeled)

        train_labeled_loader, train_unlabeled_loader, test_loader = get_loader2('P19', curr_fold, num_labeled,
                                                                                batch_labeled, batch_unlabeled)

        # train_labeled_loader, train_unlabeled_loader, test_loader \
        #     = get_loader3(sourcedata_path, curr_fold, num_labeled, batch_labeled, batch_unlabeled)

        for epoch in range(1, configs.num_epoch + 1):

            model.train()
            temporal_contr_model.train()
            classifier.train()

            train_loss = []

            train_loss1 = []
            train_loss2 = []
            train_corrects = 0
            train_samples_count = 0

            # for x, y in train_loader:

            for [x_labeled, y_labeled], [x_unlabeled] in zip(train_labeled_loader, train_unlabeled_loader):
                optimizer.zero_grad()

                aug1, aug2 = DataTransform(x_unlabeled, configs)
                aug1 = torch.from_numpy(aug1)
                aug2 = np.array(aug2, dtype=float)
                aug2 = torch.tensor(aug2)

                x_labeled = x_labeled.float().to(device)
                x_unlabeled = x_unlabeled.float().to(device)
                y_labeled = y_labeled.to(device)
                aug1 = aug1.float().to(device)
                aug2 = aug2.float().to(device)
                # y = y.to(device).unsqueeze(1)
                # label_vec = hotEncoder(y_labeled)

                predictions1, features1 = model(aug1)
                predictions2, features2 = model(aug2)

                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)



                temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)


                # normalize projection feature vectors
                zis = temp_cont_lstm_feat1
                zjs = temp_cont_lstm_feat2

                zis_shape=zis.size(0)

                lambda1 = 0.5
                lambda2 = 0.5
                # lambda1 = 1
                # lambda2 = 0.7
                nt_xent_criterion = NTXentLoss(device, zis_shape, configs.Context_Cont.temperature,
                                               configs.Context_Cont.use_cosine_similarity)
                loss1 = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2

                '''supervised contrastive_loss + cross_entropy loss'''
                # label_vec = hotEncoder(y_labeled)
                predictions3, features = model(x_labeled)

                y_pred = classifier(features)
                loss2 = criterion(y_pred, y_labeled)

                loss3 = contrastiveLoss(features, y_labeled)


                # losstrain = loss1
                losstrain = loss1 + loss2 + loss3
                torch.autograd.backward(losstrain)

                train_loss.append(losstrain.item())
                # train_loss1.append(loss1.item())
                # train_loss2.append(loss2.item())

                optimizer.step()


            # if epoch % 1 == 0:
            print('Fold: {} \tTrain Epoch: {} \tLoss: {:.6f}'.format(curr_fold, epoch, losstrain.item()))

            # train_loss.append(sum(train_loss) / len(train_loss))

                # train_corrects += (
                #         torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
                # # train_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
                # train_samples_count += x.shape[0]
        # logger.debug("\n################## Training is Done! #########################")

        test_loss = []
        test_loss1 = []
        test_loss2 = []
        test_corrects = 0
        test_samples_count = 0

        total_acc = []

        criterion = nn.CrossEntropyLoss()

        model.eval()
        # projection_layer.eval()
        classifier.eval()

        with torch.no_grad():
            for x, y in test_loader:
                # x = x.float().to(device)
                x, y = x.to(device), y.to(device)

                # y = F.one_hot(y, num_classes=2).to(torch.float) #DEAP
                # y = F.one_hot(y, num_classes=3).to(torch.float) # SEED
                y = F.one_hot(y, num_classes=2).to(torch.float)  # P19
                # y = F.one_hot(y, num_classes=2).to(torch.float)  # Epilepsy
                # y = F.one_hot(y, num_classes=6).to(torch.float)  # HAR

                predictions, features = model(x)

                y_pred_prob = classifier(features).cpu()  # (n,n_classes)
                y_pred = y_pred_prob.argmax(dim=1).cpu()  # (n,)
                y_target = y.argmax(dim=1).cpu()  # (n,)
                y = y.cpu()
                """print(y_pred[:20])
                print(y[:20])
                break"""
                metrics_dict = {}
                metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(y_target, y_pred)
                metrics_dict['Precision'] = sklearn.metrics.precision_score(y_target, y_pred, average='macro')
                metrics_dict['Recall'] = sklearn.metrics.recall_score(y_target, y_pred, average='macro')
                metrics_dict['F1'] = sklearn.metrics.f1_score(y_target, y_pred, average='macro')
                try:
                    auc_bs = metrics.roc_auc_score(y, y_pred_prob, average='macro')
                except:
                    auc_bs = np.float(0)
                # metrics_dict['AUROC'] = metrics.roc_auc_score(y, y_pred_prob, multi_class='ovr')
                metrics_dict['AUROC'] = auc_bs
                metrics_dict['AUPRC'] = metrics.average_precision_score(y, y_pred_prob)
                # metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(y, y_pred_prob, multi_class='ovr')
                # metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(y, y_pred_prob)
                # print(metrics_dict)

                dict_performance.setdefault('Accuracy', []).append(metrics_dict['Accuracy'])
                dict_performance.setdefault('Precision', []).append(metrics_dict['Precision'])
                dict_performance.setdefault('Recall', []).append(metrics_dict['Recall'])
                dict_performance.setdefault('F1', []).append(metrics_dict['F1'])
                dict_performance.setdefault('AUROC', []).append(metrics_dict['AUROC'])
                dict_performance.setdefault('AUPRC', []).append(metrics_dict['AUPRC'])

        print(f"当前fold: {curr_fold}")
        print(f"Accuracy: {mean(dict_performance['Accuracy'])}")
        print(f"Precision: {mean(dict_performance['Precision'])}")
        print(f"Recall: {mean(dict_performance['Recall'])}")
        print(f"F1: {mean(dict_performance['F1'])}")
        print(f"AUROC: {mean(dict_performance['AUROC'])}")
        print(f"AUPRC: {mean(dict_performance['AUPRC'])}")

        dict_performance_fold.setdefault('Accuracy', []).append(mean(dict_performance['Accuracy']))
        dict_performance_fold.setdefault('Precision', []).append(mean(dict_performance['Precision']))
        dict_performance_fold.setdefault('Recall', []).append(mean(dict_performance['Recall']))
        dict_performance_fold.setdefault('F1', []).append(mean(dict_performance['F1']))
        dict_performance_fold.setdefault('AUROC', []).append(mean(dict_performance['AUROC']))
        dict_performance_fold.setdefault('AUPRC', []).append(mean(dict_performance['AUPRC']))

        scheduler.step()

    fold_mean_acc = mean(dict_performance_fold['Accuracy'])
    fold_mean_pre = mean(dict_performance_fold['Precision'])
    fold_mean_recall = mean(dict_performance_fold['Recall'])
    fold_mean_f1 = mean(dict_performance_fold['F1'])
    fold_mean_auroc = mean(dict_performance_fold['AUROC'])
    fold_mean_auprc = mean(dict_performance_fold['AUPRC'])

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

# print(f"十折平均值：")
    # print(f"Accuracy: {mean(dict_performance_fold['Accuracy'])}")
    # print(f"Precision: {mean(dict_performance_fold['Precision'])}")
    # print(f"Recall: {mean(dict_performance_fold['Recall'])}")
    # print(f"F1: {mean(dict_performance_fold['F1'])}")
    # print(f"AUROC: {mean(dict_performance_fold['AUROC'])}")
    # print(f"AUPRC: {mean(dict_performance_fold['AUPRC'])}")
    #
    # end = datetime.now()
    #
    # print('Running time: %s Seconds' % (end - start))




