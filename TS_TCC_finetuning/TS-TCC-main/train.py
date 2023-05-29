import pandas as pd
import torch
import numpy
from numpy import *
import numpy as np
import data as dataset
import DEAP_dataprocess
from tasks.classification import eval_classification
np.set_printoptions(threshold=np.inf)
import argparse
import os
import sys
import time
import datetime
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
import DEAP_load_dataset
import leave_one_subject_out
from ts2vec import TS2Vec
import tasks
import random
import copy
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save
import sys
# sys.path.append("/mnt/disk1/data0/chlFiles/ts2vec-main/DEAP_load_dataset.py")
sys.path.append("/mnt/disk1/data0/chlFiles/ts2vec-main/DEAP_dataprocess.py")
from DEAP_load_dataset import construct_EEG_dataset
RANDOM_SEED = 42
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from utils import WarmUpExponentialLR
from sklearn.utils import shuffle
# def seed_everything(seed=42):
#     """"
#     Seed everything.
#     """
#     random.seed(seed)
#     # os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     """torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True"""

# seed_everything(RANDOM_SEED)

#Cai 模型保存
# def save_checkpoint_callback(
#     save_every=1,
#     unit='epoch'
# ):
#     assert unit in ('epoch', 'iter')
def callback(model, loss):
  n = model.n_epochs
  if n % 1 == 0:
    y_score, acc = eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear', fraction=1)
    # print(acc)
    if loss < model.min_loss:
      model.min_loss = loss
      # model.save(f'test_run/model_{n}.pkl')
      # model.save("test_run/base_model.pkl")
  return callback

def print_result(y_test, y_pred):
  print(f"Accuracy is: {accuracy_score(y_test, y_pred)*100:0.2f}%")
  precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
  print(f"Precision is: {precision*100:0.2f}%")
  print(f"Recall is: {recall*100:0.2f}%")
  print(f"F1 score is: {fscore*100:0.2f}%")
  auroc = roc_auc_score(y_test, y_pred, average='macro')
  print(f"AUROC is: {auroc*100:0.2f}%")
  auprc = average_precision_score(y_test, y_pred, average='macro')
  print(f"AUPRC is: {auprc*100:0.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="s02", help='The dataset name')
    parser.add_argument('--run_name', default="save_model", help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, default="DEAP", help='The data loader used to load the exper·imental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=30, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', default=True, action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
    parser.add_argument('--window_size',type=int,default=1)
    args = parser.parse_args()
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)


    cur_dim = 'valence'
    data_file = 'all32'
    fold = 10
    dict_performance = {}
    t = time.time()
    for curr_fold in range(fold):
        train_data, train_labels, test_data, test_labels = dataset.get_loader1(cur_dim, curr_fold, data_file)

        config = dict(
            batch_size=args.batch_size,
            lr=args.lr,
            output_dims=args.repr_dims,
            max_train_length=args.max_train_length
        )


        # RANDOM_SEED=42
        # seed_everything(RANDOM_SEED)

        model = TS2Vec(
            input_dims=train_data.shape[-1], # channels=32
            device=0,
            batch_size=128,
            depth=12,
            output_dims=320,
            after_epoch_callback=callback,
        )

        model.min_loss = float('inf')

        loss_log = model.fit(
            train_data,
            verbose=True,
            n_epochs=args.epochs,
        )

        model_copy = copy.deepcopy(model)
        # Train a TS2vec model
        # seed_everything(RANDOM_SEED)
        # model.finetune(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
        loss, performance = model_copy.finetune_fit(train_data, train_labels, test_data, test_labels,
                                                  encoding_window='full_series', finetune_epochs=20,
                                                  finetune_lr=0.0001)



        # epoch_loss_list = model_copy.finetune_fit(train_data, train_labels, test_data, test_labels,
        #                                           encoding_window='full_series', finetune_epochs=20,
        #                                           finetune_lr=0.001)
        print(f"当前fold: {curr_fold}")
        print(f"Loss: {loss}")
        print(f"Performance: {performance}")


        dict_performance.setdefault('Accuracy',[]).append(performance['Accuracy'])
        dict_performance.setdefault('Precision', []).append(performance['Precision'])
        dict_performance.setdefault('Recall', []).append(performance['Recall'])
        dict_performance.setdefault('F1', []).append(performance['F1'])
        dict_performance.setdefault('AUROC', []).append(performance['AUROC'])
        dict_performance.setdefault('AUPRC', []).append(performance['AUPRC'])

        # scheduler.step()

    print(f"当前维度: {cur_dim}")
    print(f"Accuracy: {mean(dict_performance['Accuracy'])}")
    print(f"Precision: {mean(dict_performance['Precision'])}")
    print(f"Recall: {mean(dict_performance['Recall'])}")
    print(f"F1: {mean(dict_performance['F1'])}")
    print(f"AUROC: {mean(dict_performance['AUROC'])}")
    print(f"AUPRC: {mean(dict_performance['AUPRC'])}")



    t = time.time() - t
    print(f"\nRunning time: {datetime.timedelta(seconds=t)}\n")
    print("Finished.")
