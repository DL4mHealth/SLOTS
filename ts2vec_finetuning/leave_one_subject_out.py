import os
from math import floor

import scipy.io as scio
import numpy as np
import h5py
from util_function import save2json, save2pkl, load_pkl_file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def data_LOSO(index):
    dataset = load_pkl_file('/mnt/disk1/data0/chlFiles/DEAP/valence_all/subject_all.pkl')
    # dataset = load_pkl_file('D:/keti/Contrastive_Learning/Ts2Vec/ts2vec-main/datasets/DEAP/{}_{}.pkl'.format(dataset_name, window_size))
    data = dataset['data']
    data = data[0:52360,:,:]
    labels = dataset['label']
    labels = labels[0:52360]
    labels = np.array(labels, dtype=np.int32)
    if index == 0:
        data_train = data[4760:,:,:]
        train_label = labels[4760:]
        state1 = np.random.get_state()
        np.random.shuffle(data_train)
        np.random.set_state(state1)
        np.random.shuffle(train_label)
        data_test = data[0:4760,:,:]
        test_label = labels[0:4760]
        state2 = np.random.get_state()
        np.random.shuffle(data_test)
        np.random.set_state(state2)
        np.random.shuffle(test_label)

    else:
        data_train = np.concatenate((data[0:4760*index,:,:], data[4760*(index+1):,:,:]), axis=0)
        train_label = np.concatenate((labels[0:4760*index],labels[4760*(index+1):]), axis=0)
        state1 = np.random.get_state()
        np.random.shuffle(data_train)
        np.random.set_state(state1)
        np.random.shuffle(train_label)
        data_test = data[4760*index:4760*(index+1),:,:]
        test_label = labels[4760*index:4760*(index+1)]
        state2 = np.random.get_state()
        np.random.shuffle(data_test)
        np.random.set_state(state2)
        np.random.shuffle(test_label)
    return data_train, data_test, train_label, test_label

def construct_EEG_dataset(index):
    # for i in range(0,32):
    train_data, test_data, train_labels, test_labels = data_LOSO(index)

    p = 0
    q = 0
    for i in range(train_labels.shape[0]):
        if train_labels[i] == 1:
            p = p + 1
        elif train_labels[i] == 0:
            q = q + 1

    train_data_high = np.empty([p, 32, 128])
    train_data_low = np.empty([q, 32, 128])
    train_data_high_label = np.empty([p])
    train_data_low_label = np.empty([q])
    m = 0
    n = 0
    for i in range(train_labels.shape[0]):
        if train_labels[i] == 1:
            train_data_high[m,:,:] = train_data[i,:,:]
            train_data_high_label[m] = train_labels[i]
            m = m + 1
        elif train_labels[i] == 0:
            train_data_low[n,:,:] = train_data[i,:,:]
            train_data_low_label[n] = train_labels[i]
            n = n + 1

    train_data_high_label_sep10=np.int32(train_data_high_label[:floor(0.1*m)])
    train_data_low_label_sep10=np.int32(train_data_low_label[:floor(0.1*n)])
    train_data_high_sep10 = train_data_high[:floor(0.1*m),:,:]
    train_data_low_sep10 = train_data_low[:floor(0.1*n),:,:]
    train_data_sep10 = np.concatenate((train_data_high_sep10,train_data_low_sep10),axis=0)
    train_data_label_sep10 = np.concatenate((train_data_high_label_sep10,train_data_low_label_sep10),axis=0)

    train_data_high_label_sep30 = np.int32(train_data_high_label[:floor(0.3 * m)])
    train_data_low_label_sep30 = np.int32(train_data_low_label[:floor(0.3 * n)])
    train_data_high_sep30 = train_data_high[:floor(0.3 * m), :, :]
    train_data_low_sep30 = train_data_low[:floor(0.3 * n), :, :]
    train_data_sep30 = np.concatenate((train_data_high_sep30, train_data_low_sep30), axis=0)
    train_data_label_sep30 = np.concatenate((train_data_high_label_sep30, train_data_low_label_sep30), axis=0)

    train_data_high_label_sep50 = np.int32(train_data_high_label[:floor(0.5 * m)])
    train_data_low_label_sep50 = np.int32(train_data_low_label[:floor(0.5 * n)])
    train_data_high_sep50 = train_data_high[:floor(0.5 * m), :, :]
    train_data_low_sep50 = train_data_low[:floor(0.5 * n), :, :]
    train_data_sep50 = np.concatenate((train_data_high_sep50, train_data_low_sep50), axis=0)
    train_data_label_sep50 = np.concatenate((train_data_high_label_sep50, train_data_low_label_sep50), axis=0)

    train_data_high_label_sep70 = np.int32(train_data_high_label[:floor(0.7 * m)])
    train_data_low_label_sep70 = np.int32(train_data_low_label[:floor(0.7 * n)])
    train_data_high_sep70 = train_data_high[:floor(0.7 * m), :, :]
    train_data_low_sep70 = train_data_low[:floor(0.7 * n), :, :]
    train_data_sep70 = np.concatenate((train_data_high_sep70, train_data_low_sep70), axis=0)
    train_data_label_sep70 = np.concatenate((train_data_high_label_sep70, train_data_low_label_sep70), axis=0)

    train_data_high_label_sep90 = np.int32(train_data_high_label[:floor(0.9 * m)])
    train_data_low_label_sep90 = np.int32(train_data_low_label[:floor(0.9 * n)])
    train_data_high_sep90 = train_data_high[:floor(0.9 * m), :, :]
    train_data_low_sep90 = train_data_low[:floor(0.9 * n), :, :]
    train_data_sep90 = np.concatenate((train_data_high_sep90, train_data_low_sep90), axis=0)
    train_data_label_sep90 = np.concatenate((train_data_high_label_sep90, train_data_low_label_sep90), axis=0)

    print(f"""train_data={train_data.shape}, train_label={train_labels.shape},
        train_data_high={train_data_high.shape}, train_data_low={train_data_low.shape}, 
        train_data_high_sep10={train_data_high_sep10.shape}, train_data_low_sep10={train_data_low_sep10.shape},
        train_data_high_label_sep10={train_data_high_label_sep10.shape}, train_data_low_label_sep10={train_data_low_label_sep10.shape},
        train_data_sep10={train_data_sep10.shape}, train_data_label_sep10={train_data_label_sep10.shape},
        train_data_sep30={train_data_sep30.shape}, train_data_label_sep30={train_data_label_sep30.shape},
        train_data_sep50={train_data_sep50.shape}, train_data_label_sep50={train_data_label_sep50.shape},
        train_data_sep70={train_data_sep70.shape}, train_data_label_sep70={train_data_label_sep70.shape},
        train_data_sep90={train_data_sep90.shape}, train_data_label_sep90={train_data_label_sep90.shape},
        test_data={test_data.shape}, test_label={test_labels.shape}""")
    return np.transpose(train_data[:, :, :], (0, 2, 1)), train_labels, np.transpose(train_data_sep10[:, :, :],(0, 2, 1)), train_data_label_sep10, \
           np.transpose(train_data_sep30[:, :, :], (0, 2, 1)), train_data_label_sep30, \
           np.transpose(train_data_sep50[:, :, :], (0, 2, 1)), train_data_label_sep50, \
           np.transpose(train_data_sep70[:, :, :], (0, 2, 1)), train_data_label_sep70, \
           np.transpose(train_data_sep90[:, :, :], (0, 2, 1)), train_data_label_sep90, \
           np.transpose(test_data[:, :, :], (0, 2, 1)), test_labels



