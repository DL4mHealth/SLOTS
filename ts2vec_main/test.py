import numpy as np
import torch
import os, shutil

import argparse
torch.set_default_tensor_type(torch.FloatTensor)
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

fc_weight_path = torch.load('aa.pth')


# fc_weight = np.load(fc_weight_path,encoding='bytes',allow_pickle=True)


# fc_weight_1 = torch.tensor(fc_weight_path, dtype=torch.float)
fc= fc_weight_path.cpu()
fc1 = fc.numpy()
channel_sort = np.sort(-fc1,axis=1)*(-1)
channel_argsort = np.argsort(-fc1,axis=1)

channel_argsort_3 = channel_argsort[:,0:3]
key = np.unique(channel_argsort_3)

results = {}
for k in key:
    v = channel_argsort_3[channel_argsort_3 == k].size
    results[k] = v
print(f"fc_weight={fc1}, fc_weight_shape={fc1.shape}")