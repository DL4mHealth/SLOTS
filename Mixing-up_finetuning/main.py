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

import torch as th

import configs
from supervised_contrastive_loss import SupConLoss

from data import get_loader


from train_model import FCN, ProjectionHead,MixUpLoss
import sys

# Args selections

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
args = parser.parse_args()



device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = 'supervised'
# training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################



def hotEncoder(v):
    ret_vec = torch.zeros(v.shape[0], 2).to(device)
    for s in range(v.shape[0]):
        ret_vec[s][v[s]] = 1
    return ret_vec

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    ls = nn.CrossEntropyLoss()(input, labels).to(device)
    return ls


# Load Model
model = FCN(32).to(device)


classifier = ProjectionHead(input_dims=128).to(device)

        # optimizer = torch.optim.AdamW(self.finetune_model.parameters(), lr=self.lr)


criterion1 = MixUpLoss(device)
criterion2 = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr = 1e-4, betas=(0.9, 0.99), weight_decay=3e-4)

# scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0, warm_epochs=configs.warm_epochs, gamma=configs.gamma)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)

contrastiveLoss = SupConLoss(temperature=0.1)
# t = time.time()

cur_dim = 'valence'
data_file = 'all32'
# fold = 4
# num_labeled = 0.2
# batch_labeled = 20
# batch_unlabeled = 100

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
# batch_labeled = 2
# batch_unlabeled = 20

fold = 4 #HAR
num_labeled = 0.1
batch_labeled = 6
batch_unlabeled = 60

# sourcedata_path = '/mnt/disk1/data0/caihlFiles/chl/ts2vec-revised20230408/Epilepsy'
sourcedata_path = '/mnt/disk1/data0/caihlFiles/chl/ts2vec-revised20230408/HAR'

dict_performance_fold = {}

alpha = 1.0

# t = time.time()
start = datetime.now()

for curr_fold in range(fold):

    # curr_fold = curr_fold + 1  # P19

    dict_performance = {}

    train_labeled_loader, train_unlabeled_loader, test_loader = get_loader(cur_dim, curr_fold, data_file,
                                                                                   num_labeled, batch_labeled,
                                                                                   batch_unlabeled)

    # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader2('P19', curr_fold, num_labeled,
    #                                                                         batch_labeled, batch_unlabeled)

    # train_labeled_loader, train_unlabeled_loader, test_loader \
    #     = get_loader3(sourcedata_path, curr_fold, num_labeled, batch_labeled, batch_unlabeled)

    for epoch in range(1, 30 + 1):

        model.train()
        classifier.train()

        train_loss = []

        train_loss1 = []
        train_loss2 = []
        train_corrects = 0
        train_samples_count = 0

        # for x, y in train_loader:

        for [x_labeled, y_labeled], [x_unlabeled] in zip(train_labeled_loader, train_unlabeled_loader):
            optimizer.zero_grad()



            x_labeled = x_labeled.float().to(device)
            x_unlabeled = x_unlabeled.float().to(device)
            y_labeled = y_labeled.to(device)

            x_1 = x_unlabeled
            x_2 = x_unlabeled[th.randperm(len(x_unlabeled))]

            lam = np.random.beta(alpha, alpha)

            x_aug = lam * x_1 + (1 - lam) * x_2

            z_1, _ = model(x_1)
            z_2, _ = model(x_2)
            z_aug, _ = model(x_aug)

            loss1 = criterion1(z_aug, z_1, z_2, lam)


            '''supervised contrastive_loss + cross_entropy loss'''
            # label_vec = hotEncoder(y_labeled)
            _, features = model(x_labeled)

            y_pred = classifier(features)
            loss2 = criterion2(y_pred, y_labeled)

            features1 = features.unsqueeze(1)

            loss3 = contrastiveLoss(features1, y_labeled)


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

            y = F.one_hot(y, num_classes=2).to(torch.float)

            _, features = model(x)

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
            metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(y, y_pred_prob, multi_class='ovr')
            metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(y, y_pred_prob)
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

print(f"十折平均值：")
print(f"Accuracy: {mean(dict_performance_fold['Accuracy'])}")
print(f"Precision: {mean(dict_performance_fold['Precision'])}")
print(f"Recall: {mean(dict_performance_fold['Recall'])}")
print(f"F1: {mean(dict_performance_fold['F1'])}")
print(f"AUROC: {mean(dict_performance_fold['AUROC'])}")
print(f"AUPRC: {mean(dict_performance_fold['AUPRC'])}")

end = datetime.now()

print('Running time: %s Seconds' % (end - start))




