import os
from math import floor

import scipy.io as scio
import numpy as np
import h5py
from util_function import save2json, save2pkl, load_pkl_file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# subject_no = "s01"

# dataset_name = 's02' ## 每次修改
#

# def load_EEG_data(subject_no,window_size):
#     dataFile = 'D:/DEAP/data_preprocessed_matlab/{}.mat'.format(subject_no)
#
#     # data=(40,40,8064) video/trial x channel x data
#     # labels=(40,4) video/trial x label (valence, arousal, dominance, liking)
#     data_dict = scio.loadmat(dataFile)
#
#     data = data_dict["data"]
#     labels = data_dict["labels"]
#
#     label = [x[1] for x in labels] # labels矩阵第一列
#     data_re = data[:, :32, 384:]
#
#     # window_size = 2
#     sections = int(60 / window_size)
#
#     label_rep = []
#     for i in range(len(label)):
#         label_sep = np.tile(label[i], sections).tolist()
#         label_rep[sections*i:sections*(i+1)] = label_sep
#     label1 = np.zeros(len(label_rep), dtype="int32")
#     p = 0
#     for i in label_rep:
#         if i > 5:
#             label1[p] = int(1)
#         else:
#             label1[p] = int(0)
#         p = p + 1
#
#
#     data_sep = np.concatenate(np.array_split(data_re, sections, axis=2), axis=0)
#
#     print(f"data_sep={data_sep.shape}, label_len={len(label1)}")
#
#     dataset_dict = {}
#     dataset_dict['data'] = data_sep
#     dataset_dict['label'] = label1
#
#     save2pkl('D:/keti/Contrastive_Learning/Ts2Vec/ts2vec-main/datasets/DEAP/', dataset_dict, '{}_{}'.format(subject_no, window_size))



def construct_EEG_dataset(dataset_name,window_size):
    dataset = load_pkl_file('/mnt/disk1/data0/chlFiles/DEAP/Valence/{}_{}.pkl'.format(dataset_name,window_size))
    # dataset = load_pkl_file('D:/keti/Contrastive_Learning/Ts2Vec/ts2vec-main/datasets/DEAP/{}_{}.pkl'.format(dataset_name, window_size))
    data = dataset['data']
    labels = dataset['label']
    labels = np.array(labels, dtype=np.int32)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, shuffle=True)
    # print(f"train_data={train_data[1]}")

    p = 0
    q = 0
    for i in range(train_labels.shape[0]):
        if train_labels[i] == 1:
            p = p + 1
        elif train_labels[i] == 0:
            q =q + 1

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
            n =n + 1

    # train_data_high_label_sep10=np.int32(train_data_high_label[:floor(0.1*m)])
    # train_data_low_label_sep10=np.int32(train_data_low_label[:floor(0.1*n)])
    # train_data_high_sep10 = train_data_high[:floor(0.1*m),:,:]
    # train_data_low_sep10 = train_data_low[:floor(0.1*n),:,:]
    # train_data_sep10 = np.concatenate((train_data_high_sep10,train_data_low_sep10),axis=0)
    # train_data_label_sep10 = np.concatenate((train_data_high_label_sep10,train_data_low_label_sep10),axis=0)
    #
    # train_data_high_label_sep30 = np.int32(train_data_high_label[:floor(0.3 * m)])
    # train_data_low_label_sep30 = np.int32(train_data_low_label[:floor(0.3 * n)])
    # train_data_high_sep30 = train_data_high[:floor(0.3 * m), :, :]
    # train_data_low_sep30 = train_data_low[:floor(0.3 * n), :, :]
    # train_data_sep30 = np.concatenate((train_data_high_sep30, train_data_low_sep30), axis=0)
    # train_data_label_sep30 = np.concatenate((train_data_high_label_sep30, train_data_low_label_sep30), axis=0)
    #
    # train_data_high_label_sep50 = np.int32(train_data_high_label[:floor(0.5 * m)])
    # train_data_low_label_sep50 = np.int32(train_data_low_label[:floor(0.5 * n)])
    # train_data_high_sep50 = train_data_high[:floor(0.5 * m), :, :]
    # train_data_low_sep50 = train_data_low[:floor(0.5 * n), :, :]
    # train_data_sep50 = np.concatenate((train_data_high_sep50, train_data_low_sep50), axis=0)
    # train_data_label_sep50 = np.concatenate((train_data_high_label_sep50, train_data_low_label_sep50), axis=0)
    #
    # train_data_high_label_sep70 = np.int32(train_data_high_label[:floor(0.7 * m)])
    # train_data_low_label_sep70 = np.int32(train_data_low_label[:floor(0.7 * n)])
    # train_data_high_sep70 = train_data_high[:floor(0.7 * m), :, :]
    # train_data_low_sep70 = train_data_low[:floor(0.7 * n), :, :]
    # train_data_sep70 = np.concatenate((train_data_high_sep70, train_data_low_sep70), axis=0)
    # train_data_label_sep70 = np.concatenate((train_data_high_label_sep70, train_data_low_label_sep70), axis=0)
    #
    # train_data_high_label_sep90 = np.int32(train_data_high_label[:floor(0.9 * m)])
    # train_data_low_label_sep90 = np.int32(train_data_low_label[:floor(0.9 * n)])
    # train_data_high_sep90 = train_data_high[:floor(0.9 * m), :, :]
    # train_data_low_sep90 = train_data_low[:floor(0.9 * n), :, :]
    # train_data_sep90 = np.concatenate((train_data_high_sep90, train_data_low_sep90), axis=0)
    # train_data_label_sep90 = np.concatenate((train_data_high_label_sep90, train_data_low_label_sep90), axis=0)
    #
    # print(f"""train_data={train_data.shape}, train_label={train_labels.shape},
    # train_data_high={train_data_high.shape}, train_data_low={train_data_low.shape},
    # train_data_high_sep10={train_data_high_sep10.shape}, train_data_low_sep10={train_data_low_sep10.shape},
    # train_data_high_label_sep10={train_data_high_label_sep10.shape}, train_data_low_label_sep10={train_data_low_label_sep10.shape},
    # train_data_sep10={train_data_sep10.shape}, train_data_label_sep10={train_data_label_sep10.shape},
    # train_data_sep30={train_data_sep30.shape}, train_data_label_sep30={train_data_label_sep30.shape},
    # train_data_sep50={train_data_sep50.shape}, train_data_label_sep50={train_data_label_sep50.shape},
    # train_data_sep70={train_data_sep70.shape}, train_data_label_sep70={train_data_label_sep70.shape},
    # train_data_sep90={train_data_sep90.shape}, train_data_label_sep90={train_data_label_sep90.shape},
    # test_data={test_data.shape}, test_label={test_labels.shape}""")
    # return np.transpose(train_data[:, :, :],(0,2,1)), train_labels, np.transpose(train_data_sep10[:, :, :],(0,2,1)), train_data_label_sep10, \
    #        np.transpose(train_data_sep30[:, :, :],(0,2,1)), train_data_label_sep30, \
    #        np.transpose(train_data_sep50[:, :, :], (0, 2, 1)), train_data_label_sep50, \
    #        np.transpose(train_data_sep70[:, :, :], (0, 2, 1)), train_data_label_sep70, \
    #        np.transpose(train_data_sep90[:, :, :], (0, 2, 1)), train_data_label_sep90, \
    #        np.transpose(test_data[:, :, :],(0,2,1)), test_labels


    print(f"train_data={train_data.shape}, train_label={train_labels}, test_data={test_data.shape}, test_label={test_labels}")
    return np.transpose(train_data[:, :, :],(0,2,1)), train_labels, np.transpose(test_data[:, :, :],(0,2,1)), test_labels

    # train_data = data[:int(0.9 * data.shape[0]), :, :]
    # train_data = np.array(train_data)
    # train_labels = np.array(labels[:int(0.9 * data.shape[0])])

    # mean = np.mean(train_data, axis=2)
    # mean = np.array(mean)
    # std = np.std(train_data, axis=2)
    # std = np.array(std)
    #
    #
    # # test_data = data[int(0.9 * data.shape[0]):, :, :]
    # # test_data = np.array(test_data)
    # # test_labels = np.array(labels[int(0.9 * data.shape[0]):])
    #
    # train_norm = np.empty_like(train_data)
    # for i in range(train_data.shape[0]):
    #     for j in range(train_data.shape[1]):
    #         for m in range(train_data.shape[2]):
    #             train_norm[i, j, m] = (train_data[i, j, m] - mean[i, j]) / std[i, j]
    #
    # test_norm = np.empty_like(test_data)
    # for i in range(test_data.shape[0]):
    #     for j in range(test_data.shape[1]):
    #         for m in range(test_data.shape[2]):
    #             test_norm[i, j, m] = (test_data[i, j, m] - mean[i,j]) / std[i,j]
    #
    # print(f"train_data={train_data.shape}, train_label={train_labels}, test_data={test_data.shape}, test_label={test_labels}")
    # return np.transpose(train_data[:, :, :],(0,2,1)), train_labels, np.transpose(test_data[:, :, :],(0,2,1)), test_labels

    # save all training data set to .h5
    # with h5py.File('D:/keti/Contrastive_Learning/Ts2Vec/ts2vec-main/datasets/DEAP/{}_{}.h5'.format(dataset_name,window_size), 'w') as h5f:
    #     h5f.create_dataset('EEG_train_data', data=train_norm)
    #     h5f.create_dataset('EEG_tain_label', data=train_labels)
    #     h5f.create_dataset('EEG_test_data', data=test_norm)
    #     h5f.create_dataset('EEG_test_label', data=test_labels)
    #
    # h5f.close()
    # check_h5('D:/keti/Contrastive_Learning/Ts2Vec/ts2vec-main/datasets/DEAP/{}.h5'.format(dataset_name))

    # return np.transpose(train_norm[:, :, :],(0,2,1)), train_labels, np.transpose(test_norm[:, :, :],(0,2,1)), test_labels

    # return train_norm[:, 1, :, np.newaxis], train_labels, test_norm[:, 1, :, np.newaxis], test_labels
#
#
# def check_h5(fname):
#     with h5py.File(fname, 'r') as f:
#         for fkey in f.keys():
#             print(f[fkey], fkey)
#     f.close()
#
# #
# #
# #
# if __name__ == '__main__':
#     window_size = 2
#     dataset_name = 's02' ## 每次修改
#
#     load_EEG_data(dataset_name,window_size)
#     construct_EEG_dataset(dataset_name,window_size)

