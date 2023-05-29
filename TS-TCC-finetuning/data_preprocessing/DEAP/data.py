import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle

def get_loader(cur_dim, curr_fold, data_file,num_labeled,batch_labeled,batch_unlabeled):
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
    train_data_2=np.transpose(train_data_1.reshape(-1,32,128),(0,2,1))
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

    unlabeled_train_data=shuffled_data[labeled_num:,:,:]
    train_data_unlabeled = torch.from_numpy(unlabeled_train_data).type(torch.FloatTensor)

    # train_data = torch.from_numpy(train_data_2).type(torch.FloatTensor)
    # y_train = torch.from_numpy(y_train_2).type(torch.LongTensor)

    test_data_1 = source_subject_data[index_test]
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
    train_unlabeled_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_unlabeled, shuffle=True)

    # train_dataset = TensorDataset(train_data, y_train)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

    test_dataset = TensorDataset(test_data, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

    # return train_data, y_train, test_data, y_test
    return train_labeled_loader, train_unlabeled_loader, test_loader
    # return train_loader, test_loader

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



