
import os
from math import floor

import scipy.io as scio
import numpy as np
import h5py
from util_function import save2json, save2pkl, load_pkl_file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

filePath = r'/mnt/disk1/data0/chlFiles/DEAP/Valence/'
name = os.listdir(filePath)
name.sort()
p=0
data_cat=np.empty([152320,32,128]) # one subject has 4760 samples, there are 32 subjects, therefore 4760*32=152320
label_cat=np.empty([152320])
for i in name:
    print(i)
    dataset = load_pkl_file('/mnt/disk1/data0/chlFiles/DEAP/Valence/{}'.format(i))
    # dataset = load_pkl_file('D:/keti/Contrastive_Learning/Ts2Vec/ts2vec-main/datasets/DEAP/{}_{}.pkl'.format(dataset_name, window_size))
    data = dataset['data']
    labels = dataset['label']
    for j in range(data.shape[0]):
        data_cat[p,:,:]=data[j,:,:]
        label_cat[p]=labels[j]
        p=p+1
label_cat = np.array(label_cat, dtype=np.int32)

dataset_dict = {}
dataset_dict['data'] = data_cat
dataset_dict['label'] = label_cat
# dataset_dict['data_baseline'] = data_baseline

save2pkl('/mnt/disk1/data0/chlFiles/DEAP/Valence/', dataset_dict,'subject_all')

# def leave_one_subject_out(data_cat, label_cat, index):
#     data_train = np.concatenate((data_cat[]))