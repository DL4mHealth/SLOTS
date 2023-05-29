import torch
import random
import numpy as np
import torch
from torch import nn
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger





def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("dataloader/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("dataloader/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))



import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime


# Cai: 保存模型
def pkl_save(name, var):
    # Cai: 打开文件夹
    # with... as... 先执行open()，然后执行该表达式返回的对象实例的__enter__函数，然后将该函数的返回值赋给as后面的变量。
    with open(name, 'wb') as f:
        # Cai: def dump(self, obj): """Write a pickled representation of obj to the open file."""
        pickle.dump(var, f)


# Cai: 模型加载
def pkl_load(name):
    with open(name, 'rb') as f:
        # Cai: def load(self):"""Read a pickled object representation from the open file.
        return pickle.load(f)


# Cai: 补充序列长度，填充nan值
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        # Cai: list列表
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    # Cai: 使用array的维度数扩充[(0,0),....]
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    # Cai: 把array填充nan值矩阵扩充
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    # Cai: Assert statements are a convenient way to insert debugging assertions into a program 调试
    # Cai: x.dtype的类型是以下的三种之一
    assert x.dtype in [np.float16, np.float32, np.float64]
    # Cai:def array_split(ary, indices_or_sections, axis=0): Split an array into multiple sub-arrays.
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B * T, False, dtype=np.bool)
    # Cai: Generate a uniform random sample from np.arange(5) of size 3 without replacement:>>> np.random.choice(5, 3, replace=False) array([3,1,0]) # random
    ele_sel = np.random.choice(
        B * T,
        size=int(B * T * p),
        replace=False
    )
    mask[ele_sel] = True
    # Cai: 复制arr
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res


def name_with_datetime(prefix='default'):
    # Cai: from datetime import datetime datetime.datetime(2022, 9, 1, 20, 20, 15, 326864)
    now = datetime.now()
    # Cai: 'default_20220901_202049'
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")


def init_dl_program(
        device_name,
        seed=None,
        use_cudnn=True,
        deterministic=False,
        benchmark=False,
        use_tf32=False,
        max_threads=None
):
    import torch
    # Cai: 设置num_threads为None
    if max_threads is not None:
        # Cai: set_num_threads Sets the number of threads used for intraop parallelism on CPU.
        torch.set_num_threads(max_threads)  # intraop
        # Cai: get_num_interop_threads # Returns the number of threads used for inter-op parallelism on CPU
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        # Cai: seed Sets the seed for generating random numbers to a non-deterministic random number.
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    # Cai: Return whether an object is an instance of a class or of a subclass thereof. False or True
    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    # Cai: reversed: Return a reverse iterator over the values of the given sequence.
    for t in reversed(device_name):
        # Cai: device:type: str  # THPDevice_type index: _int
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]

class WarmUpStepLR(torch.optim.lr_scheduler._LRScheduler):

	def __init__(self, optimizer: torch.optim.Optimizer, cold_epochs: int, warm_epochs: int, step_size: int,
			gamma: float = 0.1, last_epoch: int = -1):

		super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)
		self.cold_epochs = cold_epochs
		self.warm_epochs = warm_epochs
		self.step_size = step_size
		self.gamma = gamma



	def get_lr(self):
		if self.last_epoch < self.cold_epochs:
			return [base_lr * 0.1 for base_lr in self.base_lrs]
		elif self.last_epoch < self.cold_epochs + self.warm_epochs:
			return [
				base_lr * 0.1 + (1 + self.last_epoch - self.cold_epochs) * 0.9 * base_lr / self.warm_epochs
				for base_lr in self.base_lrs
				]
		else:
			return [
				base_lr * self.gamma ** ((self.last_epoch - self.cold_epochs - self.warm_epochs) // self.step_size)
				for base_lr in self.base_lrs
				]

class WarmUpExponentialLR(WarmUpStepLR):

	def __init__(self, optimizer: torch.optim.Optimizer, cold_epochs: int, warm_epochs: int,  # lr=lr*gamma
                 	gamma: float = 0.1, last_epoch: int = -1):    # last_epoch: int = -1学习率设置为初始值

		self.cold_epochs = cold_epochs
		self.warm_epochs = warm_epochs
		self.step_size = 1
		self.gamma = gamma

		super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)


class Classifier(nn.Module):
    def __init__(self,input_dims, output_dims):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fc = nn.Linear(input_dims, output_dims)  # 2048 self.fc = nn.Linear(128, config.class_numbers)
        self.repr_dropout = nn.Dropout(p=0.1)



    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        # x = self.repr_dropout(x)

        return x