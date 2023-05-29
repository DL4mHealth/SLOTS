import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle

import torch

from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt


# def load_UCR(dataset):
"""
ECG200数据集描述
Train size: 100 训练集样本数100
Test size: 100 测试集样本数100
Missing value: No
Number of classses: 2 两类
Time series length: 96 时序信号长度96
"""
train_file = os.path.join("datasets/UCR/ECG200/ECG200_TRAIN.tsv")
# train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
test_file = os.path.join("datasets/UCR/ECG200/ECG200_TEST.tsv")
# test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")

# train_df={DataFrame:(100,97)} test_df={DataFrame:((100,97)}
train_df = pd.read_csv(train_file, sep='\t', header=None)
test_df = pd.read_csv(test_file, sep='\t', header=None)

# train_array={ndarray:(100,97)} test_array={ndarray:(100,97)}
train_array = np.array(train_df)
test_array = np.array(test_df)

# plt.plot(train_array[1,:])
# plt.show()
# Move the labels to {0, ..., L-1}
# np.unique( )的用法 该函数是去除数组中的重复数字,并进行排序之后输出。labels={ndarray:(2,) [-1, 1]}
labels = np.unique(train_array[:, 0])
transform = {}
# transform={dict:2}{-1.0:0,1.0:1} 标签-1转成0 标签1转成1
for i, l in enumerate(labels):
    transform[l] = i

# train={ndarray:(100,96)} train_labels={ndarray:(100,)}
train = train_array[:, 1:].astype(np.float64)
train_labels = np.vectorize(transform.get)(train_array[:, 0])
# test={ndarray:(100,96)} test_labels={ndarray:(100,)}
test = test_array[:, 1:].astype(np.float64)
test_labels = np.vectorize(transform.get)(test_array[:, 0])

# Normalization for non-normalized datasets
# To keep the amplitude information, we do not normalize values over
# individual time series, but on the whole dataset
#np.nanmean np.nanstd 忽略nan值
mean = np.nanmean(train)
std = np.nanstd(train)
# train={ndarray:(100,96)} test={ndarray:(10,96)}
train = (train - mean) / std
test = (test - mean) / std

train1 = torch.tensor(train).new_tensor(0.)

input_dims=train.shape[-1]

# def pad_nan_to_target(array, target_length, axis=0, both_side=False):
#     assert array.dtype in [np.float16, np.float32, np.float64]
#     pad_size = target_length - array.shape[axis]
#     if pad_size <= 0:
#         return array
#     #Cai: 使用array的维度数扩充[(0,0),....]
#     npad = [(0, 0)] * array.ndim
#     if both_side:
#         npad[axis] = (pad_size // 2, pad_size - pad_size//2)
#     else:
#         npad[axis] = (0, pad_size)
#     #Cai: 把array填充nan值矩阵扩充
#     return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)
# def split_with_nan(x, sections, axis=0):
#     #Cai: Assert statements are a convenient way to insert debugging assertions into a program 调试
#     #Cai: x.dtype的类型是以下的三种之一
#     assert x.dtype in [np.float16, np.float32, np.float64]
#     #Cai:def array_split(ary, indices_or_sections, axis=0): Split an array into multiple sub-arrays.
#     arrs = np.array_split(x, sections, axis=axis)
#     target_length = arrs[0].shape[axis]
#     for i in range(len(arrs)):
#         arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
#     return arrs
# sections = train.shape[1] // 20
# if sections >= 2:
#     train_data = np.concatenate(split_with_nan(train, sections, axis=1), axis=0)
# print(f"train_data={train_data}")


ts_l = 96
crop_l = np.random.randint(low=2 ** (0 + 1), high=ts_l + 1)
crop_left = np.random.randint(ts_l - crop_l + 1)
crop_right = crop_left + crop_l
crop_eleft = np.random.randint(crop_left + 1)
crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=100)

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

out1 = take_per_row(train, crop_offset + crop_eleft, crop_right - crop_eleft)


out2 = take_per_row(train, crop_offset + crop_left, crop_eright - crop_left)

out11 = out1[:, -crop_l:]
out22 = out2[:, :crop_l]



print(f"train={train[..., np.newaxis]},\n tain_labels={train_labels},\n test={test[..., np.newaxis]},\n test_labels={test_labels}")
# return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels