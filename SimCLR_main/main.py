 # -*- coding: UTF-8 -*-
import torch

import os
import numpy as np
from numpy import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn import metrics

from datetime import datetime
import argparse

# from utils import *
import copy

from data import get_loader, get_loader1
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from loss import info_nce_loss
from supervised_contrastive_loss import SupConLoss
from simclr import SimCLR
from models.resnet_simclr import ResNetSimCLR,ProjectionHead
from augmentations import DataTransform
from data import get_loader, get_loader2, get_loader3


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
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
parser.add_argument('--gpu-index', default=1, type=int, help='Gpu index.')
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



# Load Model
model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
# temporal_contr_model = TC(configs, device).to(device)
classifier = ProjectionHead(input_dims=128, output_dims=2, hidden_dims=64).to(args.device) #DEAP
# classifier = ProjectionHead(input_dims=128, output_dims=3, hidden_dims=64).to(args.device) #SEED
# classifier = ProjectionHead(input_dims=128, output_dims=2, hidden_dims=64).to(args.device) # P19
# classifier = ProjectionHead(input_dims=128, output_dims=2, hidden_dims=64).to(args.device) #Epilepsy
# classifier = ProjectionHead(input_dims=128, output_dims=6, hidden_dims=64).to(args.device) #HAR



        # optimizer = torch.optim.AdamW(self.finetune_model.parameters(), lr=self.lr)
criterion = torch.nn.CrossEntropyLoss().to(args.device)
# criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr,weight_decay=3e-4)
# optimizer1 = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr,weight_decay=3e-4)
# optimizer2 = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr,weight_decay=3e-4)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=100, eta_min=0, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)

contrastiveLoss = SupConLoss(temperature=0.1)
# scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0, warm_epochs=configs.warm_epochs, gamma=configs.gamma)

# contrastiveLoss = SupConLoss(temperature=0.1)
# t = time.time()

cur_dim = 'valence'
data_file = 'all32'
# fold = 4
# num_labeled = 0.1
# batch_labeled = 10
# batch_unlabeled = 100

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

# sourcedata_path = '/mnt/disk1/data0/caihlFiles/chl/ts2vec-revised20230408/Epilepsy'
sourcedata_path = '/mnt/disk1/data0/caihlFiles/chl/ts2vec-revised20230408/HAR'
# dict_performance_fold = {}
# t = time.time()
# start = datetime.now()

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

        # curr_fold = curr_fold + 1  # P19

        dict_performance = {}

        # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader(cur_dim, curr_fold, data_file,
        #                                                                                num_labeled, batch_labeled,
        #                                                                                batch_unlabeled)

        train_labeled_loader, train_unlabeled_loader, test_loader = get_loader(cur_dim, curr_fold, data_file,
                                                                                       num_labeled, batch_labeled,
                                                                                       batch_unlabeled)

        # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader2('P19', curr_fold, num_labeled,
        #                                                                         batch_labeled, batch_unlabeled)

        # train_labeled_loader, train_unlabeled_loader, test_loader \
        #     = get_loader3(sourcedata_path, curr_fold, num_labeled, batch_labeled, batch_unlabeled)

        for epoch in range(1, 2 + 1):

            model.train()
            # temporal_contr_model.train()
            classifier.train()

            train_loss = []

            train_loss1 = []
            train_loss2 = []
            train_corrects = 0
            train_samples_count = 0

            # for x, y in train_loader:

            for [x_labeled, y_labeled], [x_unlabeled] in zip(train_labeled_loader, train_unlabeled_loader):
                optimizer.zero_grad()

                aug1, aug2 = DataTransform(x_unlabeled)
                aug1 = aug1[:, :30, :] #DEAP
                aug2 = aug2[:, :30, :]
                # aug1 = aug1[:, :60, :] # SEED P19
                # aug2 = aug2[:, :60, :]
                # aug1 = aug1[:, :, :177]  # Epilepsy HAR
                # aug2 = aug2[:, :, :177]
                aug1 = aug1.reshape(aug1.shape[0], 3, -1, aug1.shape[2]) # SEED P19
                aug2 = aug2.reshape(aug2.shape[0], 3, -1, aug2.shape[2])
                # aug1 = aug1.reshape(aug1.shape[0], 3, 1, 59) # Epilepsy HAR
                # aug2 = aug2.reshape(aug2.shape[0], 3, 1, 59)
                aug1 = torch.from_numpy(aug1)
                aug2 = np.array(aug2, dtype=float)
                aug2 = torch.tensor(aug2)

                # x_labeled = x_labeled.float().to(device)
                # x_unlabeled = x_unlabeled.float().to(device)
                # y_labeled = y_labeled.to(device)
                # aug1 = aug1.float().to(self.device)
                # aug2 = aug2.float().to(self.device)
                aug1 = aug1.float()
                aug2 = aug2.float()

                features1 = model(aug1).to(args.device)
                features2 = model(aug2).to(args.device)
                features = torch.cat([features1, features2], dim=0)
                # label_num = 200

                logits, labels = info_nce_loss(features, args.device)
                loss1 = criterion(logits, labels)

                '''supervised contrastive_loss + cross_entropy loss'''
                # label_vec = hotEncoder(y_labeled)
                x_labeled = x_labeled[:, :30, :] #DEAP
                # x_labeled = x_labeled[:, :60, :] # SEED P19
                x_labeled = x_labeled.reshape(x_labeled.shape[0], 3, -1, x_labeled.shape[2])
                # x_labeled = x_labeled[:, :, :177]  # Epilepsy HAR
                # x_labeled = x_labeled.reshape(x_labeled.shape[0], 3, 1, 59)

                features1 = model(x_labeled).to(args.device)
                y_labeled = y_labeled.to(args.device)

                y_pred = classifier(features1).to(args.device)

                loss2 = criterion(y_pred, y_labeled)

                features1 = features1.unsqueeze(1)

                loss3 = contrastiveLoss(features1, y_labeled)


                # losstrain = loss1
                losstrain = loss1 + loss2 + loss3
                torch.autograd.backward(losstrain)

                train_loss.append(losstrain.item())
                # train_loss1.append(loss1.item())
                # train_loss2.append(loss2.item())

                optimizer.step()


            if epoch % 1 == 0:
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
                x = x[:, :30, :] # DEAP
                # x = x[:, :60, :] # SEED P19
                x = x.reshape(x.shape[0], 3, -1, x.shape[2])
                # x = x[:, :, :177]   # Epilepsy HAR
                # x = x.reshape(x.shape[0], 3, 1, 59)
                # x = x.float().to(device)
                # x, y = x.to(device), y.to(device)
                y=y.to(args.device)

                # y = F.one_hot(y, num_classes=2).to(torch.float)
                y = F.one_hot(y, num_classes=2).to(torch.float) #DEAP
                # y = F.one_hot(y, num_classes=3).to(torch.float) # SEED
                # y = F.one_hot(y, num_classes=2).to(torch.float)  # P19
                # y = F.one_hot(y, num_classes=2).to(torch.float)  # Epilepsy
                # y = F.one_hot(y, num_classes=6).to(torch.float)  # HAR

                features = model(x).to(args.device)

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

end = datetime.now()

print('Running time: %s Seconds' % (end - start))




