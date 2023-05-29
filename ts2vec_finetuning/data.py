import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle

from utils_rd import *

# def get_loader(cur_dim, curr_fold, data_file,num_labeled,batch_labeled,batch_unlabeled):
#     cnn_suffix = ".mat_win_128_rnn_dataset.pkl"
#     label_suffix = ".mat_win_128_labels.pkl"
#     dataset_dir = "/mnt/disk1/data0/chlFiles/deap_shuffled_data/" + cur_dim + "/" #+ "rnn" + "/"
#
#     with open(dataset_dir + data_file + cnn_suffix, 'rb') as fp:
#         source_subject_data = pickle.load(fp)
#     with open(dataset_dir + data_file + label_suffix, 'rb') as fp:
#         source_subject_labels = pickle.load(fp)
#
#     # source_subject_labels = source_subject_labels > 3
#
#     # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32), (0, 1, 2))
#     # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32, 1), (0, 3, 1, 2))
#
#     # source_subject_labels = source_subject_labels.shape[0]
#     indexes_list = np.array([k for k in range(source_subject_labels.shape[0])])
#     # indexes_list = np.array([k for k in range(len(source_subject_labels))])
#     index_test = np.array([j for j in range(curr_fold * 10, (curr_fold + 1) * 10)])  # (0*372,1*372)
#     # index_test = np.array([j for j in range(curr_fold * 4, (curr_fold + 1) * 4)])
#     # index_test = np.array([j for j in range(curr_fold * 240, (curr_fold + 1) * 240)])  # (0*372,1*372)
#     index_train = np.array(list(set(indexes_list) ^ set(index_test)))
#
#     train_data_1=source_subject_data[index_train]
#     # train_data_2=train_data_1.reshape(-1,32,128)
#     train_data_2 = np.transpose(train_data_1.reshape(-1, 32, 128), (0, 2, 1))
#     y_train_1=source_subject_labels[index_train]
#     y_train_2=y_train_1.reshape(-1,)
#
#     index = np.array(range(0, len(y_train_2)))
#     np.random.shuffle(index)
#     shuffled_data = train_data_2[index]
#     shuffled_label = y_train_2[index]
#
#     labeled_num = int(num_labeled * len(shuffled_label))
#     labeled_train_data=shuffled_data[:labeled_num,:,:]
#     y_labeled=shuffled_label[:labeled_num]
#     train_data_labeled = torch.from_numpy(labeled_train_data).type(torch.FloatTensor)
#     y_train_labeled=torch.from_numpy(y_labeled).type(torch.LongTensor)
#
#     # unlabeled_train_data=shuffled_data[labeled_num:,:,:]
#     unlabeled_train_data = shuffled_data
#     train_data_unlabeled = torch.from_numpy(unlabeled_train_data).type(torch.FloatTensor)
#
#     # train_data = torch.from_numpy(train_data_2).type(torch.FloatTensor)
#     # y_train = torch.from_numpy(y_train_2).type(torch.LongTensor)
#
#     test_data_1 = source_subject_data[index_test]
#     # test_data_2 = test_data_1.reshape(-1, 32, 128)
#     test_data_2 = np.transpose(test_data_1.reshape(-1, 32, 128), (0, 2, 1))
#     y_test_1 = source_subject_labels[index_test]
#     y_test_2 = y_test_1.reshape(-1, )
#
#     test_data = torch.from_numpy(test_data_2).type(torch.FloatTensor)
#     y_test = torch.from_numpy(y_test_2).type(torch.LongTensor)
#     # test_data = torch.from_numpy(source_subject_data[index_test]).type(torch.FloatTensor)
#     # y_test = torch.from_numpy(source_subject_labels[index_test]).type(torch.LongTensor)
#     # #
#     train_labeled_dataset = TensorDataset(train_data_labeled, y_train_labeled)
#     train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=batch_labeled, shuffle=True)
#
#     train_unlabeled_dataset = TensorDataset(train_data_unlabeled)
#     # train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_unlabeled, shuffle=True)
#     train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_unlabeled, shuffle=True)
#
#     # train_dataset = TensorDataset(train_data, y_train)
#     # train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
#
#     test_dataset = TensorDataset(test_data, y_test)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
#
#     # return train_data, y_train, test_data, y_test
#     return train_labeled_loader, train_unlabeled_loader, test_loader
#
#
#
# # def get_loader(cur_dim, curr_fold, data_file):
# #     cnn_suffix = ".mat_win_128_rnn_dataset.pkl"
# #     label_suffix = ".mat_win_128_labels.pkl"
# #     dataset_dir = "/mnt/disk1/data0/chlFiles/deap_shuffled_data/" + cur_dim + "/" #+ "rnn" + "/"
# #
# #     with open(dataset_dir + data_file + cnn_suffix, 'rb') as fp:
# #         source_subject_data = pickle.load(fp)
# #     with open(dataset_dir + data_file + label_suffix, 'rb') as fp:
# #         source_subject_labels = pickle.load(fp)
# #
# #     # source_subject_labels = source_subject_labels > 3
# #
# #     # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32), (0, 1, 2))
# #     # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32, 1), (0, 3, 1, 2))
# #
# #     # source_subject_labels = source_subject_labels.shape[0]
# #     indexes_list = np.array([k for k in range(source_subject_labels.shape[0])])
# #     # indexes_list = np.array([k for k in range(len(source_subject_labels))])
# #     index_test = np.array([j for j in range(curr_fold * 4, (curr_fold + 1) * 4)])  # (0*372,1*372)
# #     # index_test = np.array([j for j in range(curr_fold * 240, (curr_fold + 1) * 240)])  # (0*372,1*372)
# #     index_train = np.array(list(set(indexes_list) ^ set(index_test)))
# #
# #     train_data_1=source_subject_data[index_train]
# #     train_data_2=np.transpose(train_data_1.reshape(-1,32,128),(0,2,1))
# #     y_train_1=source_subject_labels[index_train]
# #     y_train_2=y_train_1.reshape(-1,)
# #
# #     train_data = torch.from_numpy(train_data_2).type(torch.FloatTensor)
# #     y_train = torch.from_numpy(y_train_2).type(torch.LongTensor)
# #     # train_data = torch.from_numpy(source_subject_data[index_train]).type(torch.FloatTensor)
# #     # y_train = torch.from_numpy(source_subject_labels[index_train]).type(torch.LongTensor)
# #
# #     test_data_1 = source_subject_data[index_test]
# #     test_data_2 = np.transpose(test_data_1.reshape(-1, 32, 128), (0, 2, 1))
# #     y_test_1 = source_subject_labels[index_test]
# #     y_test_2 = y_test_1.reshape(-1, )
# #
# #     test_data = torch.from_numpy(test_data_2).type(torch.FloatTensor)
# #     y_test = torch.from_numpy(y_test_2).type(torch.LongTensor)
# #     # test_data = torch.from_numpy(source_subject_data[index_test]).type(torch.FloatTensor)
# #     # y_test = torch.from_numpy(source_subject_labels[index_test]).type(torch.LongTensor)
# #     # #
# #     train_dataset = TensorDataset(train_data, y_train)
# #     train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
# #
# #     test_dataset = TensorDataset(test_data, y_test)
# #     test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
# #
# #     # return train_data, y_train, test_data, y_test
# #     return train_loader, test_loader
#
# def get_loader1(cur_dim, curr_fold, data_file):
#     cnn_suffix = ".mat_win_128_rnn_dataset.pkl"
#     label_suffix = ".mat_win_128_labels.pkl"
#     dataset_dir = "/mnt/disk1/data0/chlFiles/deap_shuffled_data/" + cur_dim + "/" #+ "rnn" + "/"
#
#     with open(dataset_dir + data_file + cnn_suffix, 'rb') as fp:
#         source_subject_data = pickle.load(fp)
#     with open(dataset_dir + data_file + label_suffix, 'rb') as fp:
#         source_subject_labels = pickle.load(fp)
#
#     # source_subject_labels = source_subject_labels > 3
#
#     # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32), (0, 1, 2))
#     # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32, 1), (0, 3, 1, 2))
#
#     # source_subject_labels = source_subject_labels.shape[0]
#     indexes_list = np.array([k for k in range(source_subject_labels.shape[0])])
#     # indexes_list = np.array([k for k in range(len(source_subject_labels))])
#     index_test = np.array([j for j in range(curr_fold * 4, (curr_fold + 1) * 4)])  # (0*372,1*372)
#     # index_test = np.array([j for j in range(curr_fold * 240, (curr_fold + 1) * 240)])  # (0*372,1*372)
#     index_train = np.array(list(set(indexes_list) ^ set(index_test)))
#
#     train_data_1=source_subject_data[index_train]
#     train_data=np.transpose(train_data_1.reshape(-1,32,128),(0,2,1))
#     y_train_1=source_subject_labels[index_train]
#     y_train=y_train_1.reshape(-1,)
#
#     # train_data = torch.from_numpy(train_data_2).type(torch.FloatTensor)
#     # y_train = torch.from_numpy(y_train_2).type(torch.LongTensor)
#     # train_data = torch.from_numpy(source_subject_data[index_train]).type(torch.FloatTensor)
#     # y_train = torch.from_numpy(source_subject_labels[index_train]).type(torch.LongTensor)
#
#     test_data_1 = source_subject_data[index_test]
#     test_data = np.transpose(test_data_1.reshape(-1, 32, 128), (0, 2, 1))
#     y_test_1 = source_subject_labels[index_test]
#     y_test = y_test_1.reshape(-1, )
#
#     # test_data = torch.from_numpy(test_data_2).type(torch.FloatTensor)
#     # y_test = torch.from_numpy(y_test_2).type(torch.LongTensor)
#     # test_data = torch.from_numpy(source_subject_data[index_test]).type(torch.FloatTensor)
#     # y_test = torch.from_numpy(source_subject_labels[index_test]).type(torch.LongTensor)
#     # #
#     # train_dataset = TensorDataset(train_data, y_train)
#     # train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
#
#     # test_dataset = TensorDataset(test_data, y_test)
#     # test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
#
#     return train_data, y_train, test_data, y_test
#     # return train_loader, test_loader
#
#
#
def get_loader(cur_dim, curr_fold, data_file,num_labeled,batch_labeled,batch_unlabeled):
    cnn_suffix = ".mat_win_128_rnn_dataset.pkl"
    label_suffix = ".mat_win_128_labels.pkl"
    dataset_dir = "/data0/caihlFiles/deap_shuffled_data/" + cur_dim + "/" #+ "rnn" + "/"

    # with open(dataset_dir + data_file + cnn_suffix, 'rb') as fp:
    #     source_subject_data = pickle.load(fp)
    # with open(dataset_dir + data_file + label_suffix, 'rb') as fp:
    #     source_subject_labels = pickle.load(fp)

    # dataset_dir = "/data0/caihlFiles/seed_shuffled_data/" + "/"  # + "rnn" + "/"

    with open(dataset_dir + data_file + cnn_suffix, 'rb') as fp:
        source_subject_data = pickle.load(fp)
    with open(dataset_dir + data_file + label_suffix, 'rb') as fp:
        source_subject_labels = pickle.load(fp)

    # source_subject_data = np.transpose(source_subject_data, (1, 0, 2, 3))
    #
    # source_subject_labels = np.transpose(source_subject_labels, (1, 0))

    # SEED dataset
    source_subject_label = source_subject_labels.shape[0]
    indexes_list = np.array([k for k in range(source_subject_label)])
    index_test = np.array([j for j in range(curr_fold * 10, (curr_fold + 1) * 10)])
    index_train = np.array(list(set(indexes_list) ^ set(index_test)))
    # source_subject_labels = source_subject_labels > 3

    # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32), (0, 1, 2))
    # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32, 1), (0, 3, 1, 2))


    # indexes_list = np.array([k for k in range(source_subject_labels.shape[0])])
    # index_test = np.array([j for j in range(curr_fold * 4, (curr_fold + 1) * 4)])  # (0*372,1*372)
    # index_train = np.array(list(set(indexes_list) ^ set(index_test)))

    train_data_1=source_subject_data[index_train]
    # train_data_2=train_data_1.reshape(-1,32,128)
    # train_data_2 = np.transpose(train_data_1.reshape(-1, 62, 200), (0, 2, 1))
    train_data_2 = np.transpose(train_data_1.reshape(-1, 32, 128), (0, 2, 1))
    y_train_1=source_subject_labels[index_train]
    y_train_2=y_train_1.reshape(-1,)

    index = np.array(range(0, len(y_train_2)))
    np.random.shuffle(index)
    shuffled_data = train_data_2[index]
    shuffled_label = y_train_2[index]

    labeled_num = int(num_labeled * len(shuffled_label))
    labeled_train_data=shuffled_data[:labeled_num,:,:]
    y_labeled=shuffled_label[:labeled_num]
    train_data_labeled = torch.from_numpy(labeled_train_data).type(torch.FloatTensor)
    y_train_labeled=torch.from_numpy(y_labeled).type(torch.LongTensor)

    # unlabeled_train_data=shuffled_data[labeled_num:,:,:]
    unlabeled_train_data = shuffled_data
    train_data_unlabeled = torch.from_numpy(unlabeled_train_data).type(torch.FloatTensor)

    # train_data = torch.from_numpy(train_data_2).type(torch.FloatTensor)
    # y_train = torch.from_numpy(y_train_2).type(torch.LongTensor)

    test_data_1 = source_subject_data[index_test]
    # test_data_2 = test_data_1.reshape(-1, 32, 128)
    # test_data_2 = np.transpose(test_data_1.reshape(-1, 62, 200), (0, 2, 1))
    test_data_2 = np.transpose(test_data_1.reshape(-1, 32, 128), (0, 2, 1))
    y_test_1 = source_subject_labels[index_test]
    y_test_2 = y_test_1.reshape(-1, )

    test_data = torch.from_numpy(test_data_2).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test_2).type(torch.LongTensor)
    # test_data = torch.from_numpy(source_subject_data[index_test]).type(torch.FloatTensor)
    # y_test = torch.from_numpy(source_subject_labels[index_test]).type(torch.LongTensor)
    # #
    train_labeled_dataset = TensorDataset(train_data_labeled, y_train_labeled)
    train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=batch_labeled, shuffle=True)

    train_unlabeled_dataset = TensorDataset(train_data_unlabeled)
    # train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_unlabeled, shuffle=True)
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_unlabeled, shuffle=True)

    # train_dataset = TensorDataset(train_data, y_train)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

    test_dataset = TensorDataset(test_data, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

    # return train_data, y_train, test_data, y_test
    return train_labeled_loader, train_unlabeled_loader, test_loader


def get_loader1(cur_dim, curr_fold, data_file):
    cnn_suffix = ".mat_win_128_rnn_dataset.pkl"
    label_suffix = ".mat_win_128_labels.pkl"
    dataset_dir = "/mnt/disk1/data0/chlFiles/deap_shuffled_data/" + cur_dim + "/" #+ "rnn" + "/"

    with open(dataset_dir + data_file + cnn_suffix, 'rb') as fp:
        source_subject_data = pickle.load(fp)
    with open(dataset_dir + data_file + label_suffix, 'rb') as fp:
        source_subject_labels = pickle.load(fp)

    # source_subject_labels = source_subject_labels > 3

    # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32), (0, 1, 2))
    # source_subject_data = np.transpose(source_subject_data.reshape(-1, 128, 32, 1), (0, 3, 1, 2))

    # source_subject_labels = source_subject_labels.shape[0]
    indexes_list = np.array([k for k in range(source_subject_labels.shape[0])])
    # indexes_list = np.array([k for k in range(len(source_subject_labels))])
    index_test = np.array([j for j in range(curr_fold * 4, (curr_fold + 1) * 4)])  # (0*372,1*372)
    # index_test = np.array([j for j in range(curr_fold * 240, (curr_fold + 1) * 240)])  # (0*372,1*372)
    index_train = np.array(list(set(indexes_list) ^ set(index_test)))

    train_data_1=source_subject_data[index_train]
    train_data=np.transpose(train_data_1.reshape(-1,32,128),(0,2,1))
    y_train_1=source_subject_labels[index_train]
    y_train=y_train_1.reshape(-1,)

    # train_data = torch.from_numpy(train_data_2).type(torch.FloatTensor)
    # y_train = torch.from_numpy(y_train_2).type(torch.LongTensor)
    # train_data = torch.from_numpy(source_subject_data[index_train]).type(torch.FloatTensor)
    # y_train = torch.from_numpy(source_subject_labels[index_train]).type(torch.LongTensor)

    test_data_1 = source_subject_data[index_test]
    test_data = np.transpose(test_data_1.reshape(-1, 32, 128), (0, 2, 1))
    y_test_1 = source_subject_labels[index_test]
    y_test = y_test_1.reshape(-1, )

    # test_data = torch.from_numpy(test_data_2).type(torch.FloatTensor)
    # y_test = torch.from_numpy(y_test_2).type(torch.LongTensor)
    # test_data = torch.from_numpy(source_subject_data[index_test]).type(torch.FloatTensor)
    # y_test = torch.from_numpy(source_subject_labels[index_test]).type(torch.LongTensor)
    # #
    # train_dataset = TensorDataset(train_data, y_train)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    #
    # test_dataset = TensorDataset(test_data, y_test)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

    return train_data, y_train, test_data, y_test
    # return train_loader, test_loader


def get_loader2(dataset, split_idx, num_labeled,batch_labeled,batch_unlabeled):
    baseline = False  # always False for Raindrop
    split = 'random'  # possible values: 'random', 'age', 'gender'
    reverse = False  # False or True
    feature_removal_level = 'no_removal'  # 'set', 'sample'

    if dataset == 'P19':
        base_path = '/mnt/disk1/data0/caihlFiles/chl/ts2vec-revised20230407/P19data'
    # elif dataset == 'P19':
    #     base_path = '../P19data'
    # elif dataset == 'eICU':
    #     base_path = '../eICUdata'
    # elif dataset == 'PAM':
    #     base_path = '../PAMdata'

    # if dataset == 'P12':
    #     if subset == True:
    #         split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
    #     else:
    #         split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
    if dataset == 'P19':
        split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'

    # prepare the data:
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, split_type=split, reverse=reverse,
                                                              baseline=baseline, dataset=dataset,
                                                              predictive_label='mortality')
    print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

    if dataset == 'P12' or dataset == 'P19' or dataset == 'eICU':
        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])

        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))

        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=dataset)

        Ptrain_tensor, ytrain_tensor = tensorize_normalize(Ptrain, ytrain, mf, stdf, ms, ss)
        Pval_tensor, yval_tensor = tensorize_normalize(Pval, yval, mf, stdf, ms,ss)
        Ptest_tensor, ytest_tensor = tensorize_normalize(Ptest, ytest, mf, stdf,ms,ss)

        train_data_unlabeled = (torch.cat((Ptrain_tensor,Pval_tensor),dim=0)).transpose(1,2)
        y_train_labeled = torch.cat((ytrain_tensor,yval_tensor),dim=0)
        # ytrain_all = np.concatenate((ytrain,yval),axis=0)

        labeled_num = int(num_labeled * len(y_train_labeled))
        train_data_labeled = train_data_unlabeled[:labeled_num]
        y_train_labeled = y_train_labeled[:labeled_num]

        test_data = Ptest_tensor.transpose(1,2)
        y_test = ytest_tensor


        train_labeled_dataset = TensorDataset(train_data_labeled, y_train_labeled)
        train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=batch_labeled, shuffle=True)

        train_unlabeled_dataset = TensorDataset(train_data_unlabeled)
        # train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_unlabeled, shuffle=True)
        train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_unlabeled, shuffle=True)

        # train_dataset = TensorDataset(train_data, y_train)
        # train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

        test_dataset = TensorDataset(test_data, y_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

        # return train_data, y_train, test_data, y_test
        return train_labeled_loader, train_unlabeled_loader, test_loader

def get_loader3(sourcedata_path,curr_fold,num_labeled,batch_labeled,batch_unlabeled):
    batch_size = 32  # 64 #  128
    target_batch_size = 16
    drop_last = True
    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    finetune_dataset = torch.load(os.path.join(sourcedata_path, "val.pt"))  # train.pt
    test_dataset = torch.load(os.path.join(sourcedata_path, "test.pt"))  # test.pt
    """In pre-training:
    train_dataset: [371055, 1, 178] from SleepEEG.
    finetune_dataset: [60, 1, 178], test_dataset: [11420, 1, 178] from Epilepsy"""

    X_train = train_dataset["samples"]
    """Align the TS length between source and target datasets"""
    X_train = X_train[:, :1, :178].transpose(1,2)  # take the first 178 samples
    y_train = train_dataset["labels"]

    X_finetune = finetune_dataset["samples"]
    X_finetune = X_finetune[:, :1, :178].transpose(1,2)
    y_finetune = finetune_dataset["labels"]

    X_test = test_dataset["samples"]
    X_test = X_test[:, :1, :178].transpose(1,2)
    y_test = test_dataset["labels"]


    source_subject_data = torch.cat((X_train,X_finetune),dim=0)
    source_subject_labels = torch.cat((y_train,y_finetune),dim=0)


    index = np.array(range(0, len(source_subject_labels)))
    np.random.shuffle(index)
    shuffled_data = source_subject_data[index]
    shuffled_label = source_subject_labels[index]

    labeled_num = int(num_labeled * len(shuffled_label))
    labeled_train_data = shuffled_data[:labeled_num, :, :].type(torch.FloatTensor)

    y_labeled = shuffled_label[:labeled_num].type(torch.LongTensor)


    unlabeled_train_data = shuffled_data.type(torch.FloatTensor)

    test_data_1 = X_test.type(torch.FloatTensor)
    y_test_1 = y_test.type(torch.LongTensor)


    # y_test = torch.from_numpy(y_test_2).type(torch.LongTensor)
    # test_data = torch.from_numpy(source_subject_data[index_test]).type(torch.FloatTensor)
    # y_test = torch.from_numpy(source_subject_labels[index_test]).type(torch.LongTensor)
    # #
    train_labeled_dataset = TensorDataset(labeled_train_data, y_labeled)
    train_labeled_loader = DataLoader(dataset=train_labeled_dataset, batch_size=batch_labeled, shuffle=True)

    train_unlabeled_dataset = TensorDataset(unlabeled_train_data)
    # train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_unlabeled, shuffle=True)
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_unlabeled, shuffle=True)

    # train_dataset = TensorDataset(train_data, y_train)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

    test_dataset = TensorDataset(test_data_1, y_test_1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=50, shuffle=True)

    # return train_data, y_train, test_data, y_test
    return train_labeled_loader, train_unlabeled_loader, test_loader
