import pandas as pd
import torch
import numpy
from numpy import *
import numpy as np
# import data as dataset
# import DEAP_dataprocess
from tasks.classification import eval_classification
import sklearn
np.set_printoptions(threshold=np.inf)
import argparse
import os
import sys
import time
import datetime
from datetime import date
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
# import DEAP_load_dataset
# import leave_one_subject_out
from ts2vec import TS2Vec
import tasks
import random
import copy
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save
import sys

# sys.path.append("/mnt/disk1/data0/chlFiles/ts2vec-main/DEAP_load_dataset.py")
# sys.path.append("/data0/caihlFiles/ts2vec_main/ts2vec-revised20230223/DEAP_dataprocess.py")
# from DEAP_load_dataset import construct_EEG_dataset

import numpy as np
from utils import *
from models import TSEncoder, ProjectionHead
from models.losses import hierarchical_contrastive_loss
from utils import WarmUpExponentialLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.utils import shuffle
import supervised_contrastive_loss
from sklearn import metrics
from data import get_loader, get_loader2, get_loader3


import model_projection

# def hotEncoder(v):
#     ret_vec = torch.zeros(v.shape[0], 3).to(device)
#     for s in range(v.shape[0]):
#         ret_vec[s][v[s]] = 1
#     return ret_vec

def hotEncoder(v):
    ret_vec = torch.zeros(v.shape[0], 2).to(device)
    for s in range(v.shape[0]):
        ret_vec[s][v[s]] = 1
    return ret_vec

# def hotEncoder(v):
#     ret_vec = torch.zeros(v.shape[0], 6).to(device)
#     for s in range(v.shape[0]):
#         ret_vec[s][v[s]] = 1
#     return ret_vec

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    ls = nn.CrossEntropyLoss()(input, labels).to(device)
    return ls


class ProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(320, 64)

    def forward(self, x):
        x = self.fc(x)

        return x

def eval_with_pooling(model, x, mask=None, slicing=None, encoding_window=None):
    out = model(x.to(device, non_blocking=True), mask)
    if encoding_window == 'full_series':
        if slicing is not None:
            out = out[:, slicing]
        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size=out.size(1),
        ).transpose(1, 2)

    elif isinstance(encoding_window, int):
        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size=encoding_window,
            stride=1,
            padding=encoding_window // 2
        ).transpose(1, 2)
        if encoding_window % 2 == 0:
            out = out[:, :-1]
        if slicing is not None:
            out = out[:, slicing]

    elif encoding_window == 'multiscale':
        p = 0
        reprs = []
        while (1 << p) + 1 < out.size(1):
            t_out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=(1 << (p + 1)) + 1,
                stride=1,
                padding=1 << p
            ).transpose(1, 2)
            if slicing is not None:
                t_out = t_out[:, slicing]
            reprs.append(t_out)
            p += 1
        out = torch.cat(reprs, dim=-1)

    else:
        if slicing is not None:
            out = out[:, slicing]

    # return out.cpu()
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="s02", help='The dataset name')
    parser.add_argument('--run_name', default="save_model",
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, default="DEAP",
                        help='The data loader used to load the exper·imental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=100, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', default=True, action="store_true",
                        help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--warm_epochs', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--temporal_unit', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0.1)

    args = parser.parse_args()
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    model=TSEncoder(input_dims=32, output_dims=320, hidden_dims=64, depth=3).to(device) # DEAP
    # model = FCN(62).to(device) # SEED
    # classnum = 3
    # model = FCN(60).to(device) # P19
    # classnum = 2
    # model = FCN(1).to(device) # Epilepsy
    # classnum = 2
    # model = FCN(1).to(device)  # HAR
    # model = TSEncoder(input_dims=62, output_dims=320, hidden_dims=64, depth=3).to(device)# SEED
    # model = TSEncoder(input_dims=60, output_dims=320, hidden_dims=64, depth=3).to(device)# P19
    # model = TSEncoder(input_dims=1, output_dims=320, hidden_dims=64, depth=3).to(device)# Epilepsy
    # model = TSEncoder(input_dims=1, output_dims=320, hidden_dims=64, depth=3).to(device)# HAR


    # model = ecnn.ECNN(128).to(device)

    # classifier = ProjectionHead(input_dims=320, output_dims=3, hidden_dims=128).to(device)
    #
    classifier = ProjectionHead(input_dims=320, output_dims=2, hidden_dims=128).to(device)

    # classifier = ProjectionHead(input_dims=320, output_dims=6, hidden_dims=128).to(device)

    contrastiveLoss = supervised_contrastive_loss.SupConLoss(temperature=args.temperature)
    # contrastiveLoss = hierarchical_contrastive_loss()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=1e-5)

    scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0, warm_epochs=args.warm_epochs, gamma=args.gamma)

    projection_layer = model_projection.ProjectionModel().to(device)
    # t = time.time()

    cur_dim = 'valence'
    data_file = 'all32'
    # fold = 4
    # num_labeled = 0.1
    # batch_labeled = 5
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

    # fold = 4  # HAR
    # num_labeled = 0.1
    # batch_labeled = 6
    # batch_unlabeled = 60

    # sourcedata_path = '/mnt/disk1/data0/caihlFiles/chl/ts2vec-revised20230408/Epilepsy'
    # sourcedata_path = '/mnt/disk1/data0/caihlFiles/chl/ts2vec-revised20230408/HAR'
    # dict_performance_fold = {}
    # t = time.time()
    # start = datetime.now()

    loop = 2
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

            # train_loader, test_loader = dataset.get_loader(cur_dim, curr_fold, data_file)

            # train_labeled_loader, train_unlabeled_loader, test_loader = dataset.get_loader(cur_dim, curr_fold, data_file,
            #                                                                                num_labeled, batch_labeled,
            #                                                                                batch_unlabeled)

            train_labeled_loader, train_unlabeled_loader, test_loader = get_loader(cur_dim, curr_fold, data_file,
                                                                                           num_labeled, batch_labeled,
                                                                                           batch_unlabeled)



            # train_labeled_loader, train_unlabeled_loader, test_loader = get_loader2('P19', curr_fold, num_labeled,
            #                                                                         batch_labeled, batch_unlabeled)

            # train_labeled_loader, train_unlabeled_loader, test_loader \
            #     = get_loader3(sourcedata_path, curr_fold, num_labeled, batch_labeled, batch_unlabeled)

            for epoch in range(1, args.epochs + 1):

                model.train()
                # projection_layer.train()
                classifier.train()

                train_loss = []

                train_loss1 = []
                train_loss2 = []
                train_corrects = 0
                train_samples_count = 0

                # for x, y in train_loader:

                for [x_labeled, y_labeled], [x_unlabeled] in zip(train_labeled_loader, train_unlabeled_loader):


                    # optimizer.zero_grad()

                    # x = x.float().to(device)
                    # x, y = x.to(device), y.to(device)

                    x_labeled = x_labeled.to(device)
                    x_unlabeled = x_unlabeled.to(device)
                    y_labeled = y_labeled.to(device)
                    # y = y.to(device).unsqueeze(1)
                    label_vec = hotEncoder(y_labeled)

                    if args.max_train_length is not None and x_unlabeled.size(1) > args.max_train_length:
                        window_offset = np.random.randint(x_unlabeled.size(1) - args.max_train_length + 1)
                        x_unlabeled = x_unlabeled[:, window_offset: window_offset + args.max_train_length]
                    # x = x.to(device)

                    ts_l = x_unlabeled.size(1)
                    crop_l = np.random.randint(low=2 ** (args.temporal_unit + 1), high=ts_l + 1)
                    crop_left = np.random.randint(ts_l - crop_l + 1)
                    crop_right = crop_left + crop_l
                    crop_eleft = np.random.randint(crop_left + 1)
                    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x_unlabeled.size(0))

                    optimizer.zero_grad()

                    out1 = model(take_per_row(x_unlabeled, crop_offset + crop_eleft, crop_right - crop_eleft))
                    out1 = out1[:, -crop_l:]
                    out1 = F.normalize(out1, dim=0)

                    out2 = model(take_per_row(x_unlabeled, crop_offset + crop_left, crop_eright - crop_left))
                    out2 = out2[:, :crop_l]
                    out2 = F.normalize(out2, dim=0)

                    loss1 = hierarchical_contrastive_loss(
                        out1,
                        out2,
                        temporal_unit=args.temporal_unit
                    )


                    # out = model(x.to(device, non_blocking=True))
                    # out = F.normalize(out, dim=0)
                    # out = projection_layer(out)
                    out = eval_with_pooling(model, x_labeled, encoding_window='full_series')
                    out = F.normalize(out, dim=0)
                    # if encoding_window == 'full_series':
                    out = out.squeeze(1)

                    # y_pred_prob = classifier(out).squeeze(1)
                    y_pred_prob = classifier(out)
                    loss2 = cross_entropy_one_hot(y_pred_prob, label_vec)

                    '''supervised contrastive_loss + cross_entropy loss'''
                    # label_vec = hotEncoder(y_labeled)
                    # x_labeled = x_labeled.unsqueeze(1)
                    # out = model(x_labeled).to(device)
                    # out = F.normalize(out, dim=0)
                    #
                    y_proj = projection_layer(out)
                    y_proj = F.normalize(y_proj, dim=0)
                    y_proj = y_proj.unsqueeze(1)
                    #
                    loss3 = contrastiveLoss(y_proj, y_labeled)



                    # loss2 = criterion(y_pred, y)


                    # losstrain = loss1
                    losstrain = loss1 + loss2 + loss3
                    torch.autograd.backward(losstrain)

                    train_loss.append(losstrain.item())
                    # train_loss1.append(loss1.item())
                    # train_loss2.append(loss2.item())

                    optimizer.step()

                    # y_pred = y_pred_prob.argmax(dim=1).cpu # (n,)
                    # y_target = y.argmax(dim=1).cpu  # (n,)
                    # y = y.cpu()
                    # y_pred_prob=y_pred_prob.cpu()

                    # metrics_dict = {}
                    # metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(y_target, y_pred)
                    # metrics_dict['Precision'] = sklearn.metrics.precision_score(y_target, y_pred, average='macro')
                    # metrics_dict['Recall'] = sklearn.metrics.recall_score(y_target, y_pred, average='macro')
                    # metrics_dict['F1'] = sklearn.metrics.f1_score(y_target, y_pred, average='macro')
                    # metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(y, y_pred_prob, multi_class='ovr')
                    # metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(y, y_pred_prob)

                if epoch % 1 == 0:
                    print('Fold: {} \tTrain Epoch: {} \tLoss: {:.6f}'.format(curr_fold, epoch, losstrain.item()))

                # train_loss.append(sum(train_loss) / len(train_loss))

                    # train_corrects += (
                    #         torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
                    # # train_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
                    # train_samples_count += x.shape[0]

            test_loss = []
            test_loss1 = []
            test_loss2 = []
            test_corrects = 0
            test_samples_count = 0

            model.eval()
            # projection_layer.eval()
            classifier.eval()

            with torch.no_grad():
                for x, y in test_loader:
                    # x = x.float().to(device)
                    x, y = x.to(device), y.to(device)

                    out = eval_with_pooling(model, x, encoding_window='full_series')
                    # out = model(x.to(device, non_blocking=True))
                    # out = F.normalize(out, dim=0)
                    # out = projection_layer(out)
                    out = F.normalize(out, dim=0)

                    # if encoding_window == 'full_series':
                    out = out.squeeze(1)

                    # y_pred_prob = classifier(out).squeeze(1).cpu
                    y_pred_prob = classifier(out).cpu()
                    y=y.cpu()
                    test_labels_onehot = (F.one_hot(y.long(), 2)).numpy() # DEAP
                    # test_labels_onehot = (F.one_hot(y.long(), 3)).numpy() # SEED
                    # test_labels_onehot = (F.one_hot(y.long(), 2)).numpy()  # P19
                    # test_labels_onehot = (F.one_hot(y.long(), 2)).numpy()  # Epilepsy
                    # test_labels_onehot = (F.one_hot(y.long(), 6)).numpy()  # HAR


                    pred_prob = y_pred_prob
                    # print(pred_prob.shape)
                    pred = pred_prob.argmax(axis=1)
                    target = y
                    target_prob = test_labels_onehot
                    # print(target_prob.shape)
                    metrics_dict = {}
                    metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(target, pred)
                    metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
                    metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
                    metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
                    try:
                        auc_bs = metrics.roc_auc_score(target_prob, pred_prob, average='macro')
                    except:
                        auc_bs = np.float(0)
                    # metrics_dict['AUROC'] = metrics.roc_auc_score(y, y_pred_prob, multi_class='ovr')
                    metrics_dict['AUROC'] = auc_bs
                    metrics_dict['AUPRC'] = metrics.average_precision_score(target_prob, pred_prob)
                    # metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, average='macro',                                                                      multi_class='ovr')
                    # metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob, average='macro')
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


    end = datetime.now()

    print('Running time: %s Seconds' % (end - start))

    #     print(f"平均值：")
    #     print(f"Accuracy: {mean(dict_performance_fold['Accuracy'])}")
    #     print(f"Precision: {mean(dict_performance_fold['Precision'])}")
    #     print(f"Recall: {mean(dict_performance_fold['Recall'])}")
    #     print(f"F1: {mean(dict_performance_fold['F1'])}")
    #     print(f"AUROC: {mean(dict_performance_fold['AUROC'])}")
    #     print(f"AUPRC: {mean(dict_performance_fold['AUPRC'])}")
    #
    #         # print('The accuracy is : {:.6f}'.format(test_acc))
    #         # mean_accuracy = mean_accuracy + test_acc
    #
    # #            print('当前维度:{}\n' \
    # # \
    # #                  '当前受试者:{}\n' \
    # #                  '\tmethod-mean_accuracy: {:.6f}\n'.format(
    # #                cur_dim,
    # #                data_file,
    # #                mean_accuracy / 10))
    #
    #     # t = time.time() - t
    #     # print(f"\nRunning time: {datetime.timedelta(seconds=t)}\n")
    #     # print("Finished.")
    #     end = datetime.now()
    #
    #     print('Running time: %s Seconds' % (end - start))
