import os
import scipy.io as scio
import numpy as np
import h5py
from util_function import save2json, save2pkl, load_pkl_file
import matplotlib.pyplot as plt
import torch

# subject_no = "s01"

# dataset_name = 's02' ## 每次修改
#

def load_EEG_data(subject_no,window_size):
    dataFile = '/mnt/disk1/data0/chlFiles/data_preprocessed_matlab/{}.mat'.format(subject_no)

    # data=(40,40,8064) video/trial x channel x data
    # labels=(40,4) video/trial x label (valence, arousal, dominance, liking)
    data_dict = scio.loadmat(dataFile)

    data = data_dict["data"]
    labels = data_dict["labels"]

    label = [x[1] for x in labels] # labels矩阵第一列
    data_re = np.array(data[:, :32, 384:])

    # 预处理 激励信号减基线信号
    data_baseline = np.array(data[:, :32, :384])
    data_baselinemean = np.empty([40,32,128])
    data_baseline1 = np.empty([3,128])
    for i in range(data_baseline.shape[0]):
        for j in range(data_baseline.shape[1]):
            for m in range(3):
                data_baseline1[m,:] = data_baseline[i,j,m*128:(m+1)*128]
            data_baselinemean[i,j,:] = np.mean(data_baseline1,axis=0)

    data_pre = np.empty_like(data_re)
    for i in range(data_re.shape[0]):
        for j in range(data_re.shape[1]):
            for m in range(0,60):
                data_pre[i,j,m*128:(m+1)*128] = data_re[i,j,m*128:(m+1)*128] - data_baselinemean[i,j,:]


    # print(f"data_pre={data_pre[15,16,:]}")
    # window_size = 2
    # sections = int(60 / window_size)
    sections = int(60+59)

    label_rep = []
    for i in range(len(label)):
        label_sep = np.tile(label[i], sections).tolist()
        label_rep[sections*i:sections*(i+1)] = label_sep
    label1 = np.zeros(len(label_rep), dtype="int32")
    p = 0
    for i in label_rep:
        if i > 5:
            label1[p] = int(1)
        else:
            label1[p] = int(0)
        p = p + 1

    data_sep = np.empty([4760,32,128])
    # for m in range(0, 7553, 64):
    #     for i in range(data_pre.shape[0]):
    for i in range(data_pre.shape[0]):
        for j in range(data_pre.shape[1]):
            for m in range(0,7553,64):
                data_sep[119*i+(m//64),j,:] = data_pre[i,j,m:m+128]


    mean = np.mean(data_sep, axis=2)

    # mean = np.array(mean)

    std = np.std(data_sep, axis=2)

    # std = np.array(std)
    data_norm = np.empty_like(data_sep)
    for i in range(data_sep.shape[0]):
        for j in range(data_sep.shape[1]):
            for m in range(data_sep.shape[2]):
                data_norm[i, j, m] = (data_sep[i, j, m] - mean[i, j]) / std[i, j]

    # test_norm = np.empty_like(test_data)
    # for i in range(test_data.shape[0]):
    #     for j in range(test_data.shape[1]):
    #         for m in range(test_data.shape[2]):
    #             test_norm[i, j, m] = (test_data[i, j, m] - mean[i,j]) / std[i,j]


    # print(f"data_pre={data_pre[0,0,64]}, data_sep={data_sep[1,0,0]}")

    # data_sep = np.concatenate(np.array_split(data_re, sections, axis=2), axis=0)
    # data_sep = np.concatenate(np.array_split(data_pre, sections, axis=2), axis=0)


    print(f"data_norm={data_norm.shape}, label_len={len(label1)}")
    # # print(f"data_sep={data_sep.shape}, label_len={len(label1)}, data_baseline={data_baseline.shape}")
    #
    dataset_dict = {}
    dataset_dict['data'] = data_norm
    dataset_dict['label'] = label1
    # dataset_dict['data_baseline'] = data_baseline

    save2pkl('/mnt/disk1/data0/chlFiles/DEAP/Arousal/', dataset_dict, '{}_{}'.format(subject_no, window_size))

if __name__ == '__main__':
    window_size = 1
    dataset_name = 's32' ## 每次修改

    load_EEG_data(dataset_name,window_size)
