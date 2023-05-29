import os
import json
import queue
import torch
import pickle
import itertools
import torch.nn as nn
import numpy as np
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Callable, Union, Any, TypeVar, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def loss_plot(axs, loss_data, name=None):
    axs.plot(range(len(loss_data)), loss_data, label=name)
    axs.tick_params(labelsize=13)
    axs.set_ylabel(name, size=13)
    axs.set_xlabel('Epochs', size=13)
    axs.legend()
    # axs.set_title(name)


def norm(x):
    x = np.array(x)
    x = x / x.sum()
    return x.tolist()


def save_model(M, save_file='', file_name=None):
    mkdir('./saved_model/' + save_file + '/')
    torch.save(M.state_dict(), './saved_model/' + save_file + '/{}.pkl'.format(file_name))


def load_model(M, tag='', station='', save_file=''):
    g_load_path = ''
    if tag == 'fed':
        g_load_path = './saved_model/' + save_file + '/G.pkl'
    elif tag == 'idpt':
        g_load_path = './saved_model/' + save_file + '/independent_' + station + '_G.pkl'

    M.load_state_dict(torch.load(g_load_path))


def save2csv(fpt, data, columns, index):
    data = {key: value for key, value in zip(columns, data)}
    print('*** data: \n', data)
    dataframe = pd.DataFrame(data, columns=columns, index=index)
    # 转置
    dataframe = pd.DataFrame(dataframe.values.T, index=dataframe.columns, columns=dataframe.index)
    dataframe.to_csv(fpt, index=True, sep=',')


def save2json(fpt, data, name):
    with open(fpt + '/' + name + '.json', 'w+') as jsonf:
        json.dump(data, jsonf)


def loadjson(fpt, name):
    if '.json' in name:
        name = name[:-5]
    with open(fpt + '/' + name + '.json', 'r+') as jsonf:
        data = json.load(jsonf)
    return data


def save2pkl(fpt, data, name):
    with open(fpt + '/' + name + '.pkl', 'wb') as pkf:
       pickle.dump(data, pkf)


def load_pkl_file(fpt):
    pkl_file = open(fpt, 'rb')
    data = pickle.load(pkl_file)

    return data


# ============= About construction of datasets ===========
def construct_queue(data):
    q = queue.Queue()
    # 构造队列
    for i in range(len(data)):
        q.put(data[i])

    return q


def get_n_items_from_queue(q, n):
    # 从队列q中取出n个元素
    r = []
    for _ in range(n):
        if not q.empty():
            item = q.get()
            r.append(item)
        else:
            break
    return r, len(r)


# ======================= DNN functions ==================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


# custom weights initialization called on netG and netD
def weights_init_v2(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        # print(group['params'])
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def generate_augment_samples(G, bs, init_imgsize, latent_dim, n_classes):
    """
    利用生成网络进行采样，采样得到增强数据集
    :param G:
    :param bs:
    :param init_imgsize:
    :param latent_dim:
    :param n_classes:
    :return:
    """
    # fixed noise & label
    # label preprocess
    onehot = torch.zeros(n_classes, n_classes)
    onehot = onehot.scatter_(1, torch.LongTensor([i for i in range(n_classes)]).view(n_classes, 1), 1).view(
        n_classes, n_classes, 1, 1)
    fill = torch.zeros([n_classes, n_classes, init_imgsize, init_imgsize])
    for i in range(n_classes):
        fill[i, i, :, :] = 1

    # 直接采用图像进行反卷积
    z = torch.randn((bs, latent_dim), device='cuda').view(-1, latent_dim, 1, 1)
    gen_rand = (torch.rand(bs, 1) * n_classes).type(torch.LongTensor).squeeze()
    gen_labels = onehot[gen_rand]

    z.requires_grad = False
    gen_labels = gen_labels.cuda()
    gen_labels.requires_grad = False

    G.eval()
    with torch.no_grad():
        aug_images = G(z, gen_labels)

    return aug_images, gen_rand.cuda()


def get_ACGAN_synthetic_samples(G, size, latent_dim, n_classes):
    """
    用AC GAN 生成的数据
    :param G:
    :param size:
    :param latent_dim:
    :param n_classes:
    :return:
    """
    noise = torch.randn(size, latent_dim)
    labels = torch.randint(0, n_classes, (size,), dtype=torch.long)

    noise = noise.cuda()
    labels = labels.cuda()

    G.eval()
    with torch.no_grad():
        gen_images = G(noise, labels).detach()

    return gen_images, labels

# =========================== Ploting ===================
def save_all_avg_results(fpt, data, columns, index):
    if len(columns) == len(data):
        data = {key: value for key, value in zip(columns, data)}
    elif len(columns) == 1:
        data = {columns[0]: data}
    print('*** data: \n', data)
    dataframe = pd.DataFrame(data, columns=columns, index=index)
    dataframe.to_csv(fpt, index=True, sep=',')


# 用于计算不同文件夹下的实验指标结果的均值
def compute_avg_of_data_in_file(args, dataset_index, results_logdir, csv_save_fpath, indicator_name, leg=None):
    if leg is None:
        leg = ['Fed', 'Independent']

    datas = get_all_datasets(results_logdir, leg)

    if isinstance(datas, list):
        data = pd.concat(datas)

    unit_sets = data['Unit'].values.tolist()
    unit_sets = set(unit_sets)

    # 针对load_dataset_v2，对不同参与方，每个参与方都有自己的站点数据添加
    if type(args.selected_stations[0]) == list:
        station_names = args.clients
    else:
        station_names = args.selected_stations

    indicator_avg_list = []
    for mode in leg:
        avg_t = 0
        for u in unit_sets:
            fed_avg_data = data[data.Condition == mode]
            if indicator_name == 'all_rmse':
                fed_avg_data = fed_avg_data[fed_avg_data.Unit == u][indicator_name].values
            else:
                fed_avg_data = fed_avg_data[fed_avg_data.Unit == u][args.select_dim].values
            avg_t += fed_avg_data
        indicator_avg = avg_t / len(unit_sets)
        indicator_avg_list.append(indicator_avg)

        # 保存到本地
        fed_save_csv_pt = csv_save_fpath + mode + '/' + 'dataset_' + \
                          str(dataset_index) + '_' + mode + '_' + indicator_name + '_avg_resluts.csv'
        if indicator_name == 'all_rmse':
            save_all_avg_results(fed_save_csv_pt, indicator_avg, [indicator_name], station_names)
        else:
            save_all_avg_results(fed_save_csv_pt, indicator_avg, args.select_dim, station_names)
        print('[C] ' + mode + ' avg: ', indicator_avg)


def plot_fam_stat(data, labels, save_name):
    """
    用于绘制数据统计图，
    :param data: 字典数据，每个数据子集的每个标签数据统计个数
    :param labels:
    :param save_name:
    :return:
    """
    fam_names = list(data.keys())
    fam_data_values = data.values()
    data = np.array(list(data.values()), dtype=np.int32)
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(labels, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(fam_names, widths, left=starts, height=0.8,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # for y, (x, c) in enumerate(zip(xcenters, widths)):
        #     ax.text(x, y, str(int(c)), ha='center', va='center',
        #             color=text_color)
    ax.legend(ncol=len(labels), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    plt.show()
    # plt.savefig('E:/zx/FedLearning/FedFER/datasets_overview/{}_fam_stat.svg'.format(save_name))
    # plt.close(fig)


# =========== Get files ======================
def get_datasets(logdir, file_suffix='', condition=None):
    """
    file_suffix: 文件后缀，例如“results.txt, results.csv”
    """
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        print('files: ', files)
        for file in files:
            if condition not in units:
                units[condition] = 0
            unit = units[condition]
            units[condition] += 1
            try:
                if file_suffix[-3:] == 'txt':
                    exp_data = pd.read_table(os.path.join(root, file))
                if file_suffix[-3:] == 'csv':
                    exp_data = pd.read_csv(os.path.join(root, file))
            except:
                print('Could not read from %s' % os.path.join(root, file))
                continue
            xaxis = [i+1 for i in range(exp_data.shape[0])]
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition', condition)
            if 'station' not in exp_data.columns:
                exp_data.insert(len(exp_data.columns), 'station', xaxis)
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    DIV_LINE_WIDTH = 50
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and (logdir[-1] == os.sep or logdir[-1] == '/'):
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    # 这里是确保legend的数量和要绘制的logdir的数量一样
    assert not legend or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # 从具体的logdirs载入数据
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, 'results.csv', leg)
    else:
        for log in logdirs:
            data += get_datasets(log, 'results.csv')
    return data


def plot_indicator_avg_results_m(logdir, save_path, xaxis, value, save_csv_fpth, leg=None):
    # 重新实现了seaborn的带有均值线的多次实验结果图
    if leg is None:
        leg = ['Federated GAN', 'Local GAN']

    datas = get_all_datasets(logdir, leg)
    # 创建绘图
    fig, ax = plt.subplots()

    if isinstance(datas, list):
        data = pd.concat(datas)
        print('*** data: \n', data)

    unit_sets = data['Unit'].values.tolist()
    unit_sets = set(unit_sets)

    condition_sets = data['Condition'].values.tolist()
    condition_sets = set(condition_sets)

    xaxis_sets = data[xaxis].values.tolist()
    xaxis_sets = set(xaxis_sets)

    indicator_avg_list = []
    # 不同的condition
    for mode in condition_sets:
        avg_t = 0
        # 计算被标记的不同unit之间的均值
        condition_min_list = []
        condition_max_list = []
        condition_unit_data_list = []
        condition_data = data[data.Condition == mode]
        for u in unit_sets:
            condition_unit_data = condition_data[condition_data.Unit == u][value].values
            condition_unit_data_list.append(condition_unit_data.reshape((1, -1)))
            avg_t += condition_unit_data

        # 用于替代seaborn中的tsplot绘图函数
        def tsplot(ax, x, data, **kw):
            est = np.mean(data, axis=0)
            sd = np.std(data, axis=0)
            cis = (est - sd, est + sd)
            x = np.array(x).astype(dtype=np.str)
            ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
            ax.plot(x, est, label=mode, **kw)
            ax.tick_params(labelsize=13)
            ax.set_ylabel('RMSE', size=13)
            ax.set_xlabel('Participant', size=13)
            ax.legend()
            ax.margins(x=0)

            return est, sd

        # 将几次的实验数据进行拼接，形成一个（station， exp_time）形状的数组
        all_condition_unit_data = np.concatenate(condition_unit_data_list, axis=0)
        xaxis_from_sets = [i for i in xaxis_sets]

        indicator_avg, indicator_std = tsplot(ax, xaxis_from_sets, all_condition_unit_data)

        # 保存到本地
        save_avg_csv_pt = save_csv_fpth + mode + '_' + value + '_avg_resluts.csv'
        mode_save_data = {'RMSE': indicator_avg, 'std': indicator_std}
        # 使用pandas保存成csv
        dataframe = pd.DataFrame(mode_save_data)
        dataframe.to_csv(save_avg_csv_pt, index=True, sep=',')

        # save_all_avg_results(fed_save_csv_pt, indicator_avg, [value], xaxis)
    save_path = save_path + value + '_results'
    plt.savefig(save_path + '.eps')
    plt.savefig(save_path + '.svg')

    plt.close()


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                          save_name='', save_path='', if_print=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        if isinstance(cm, np.ndarray) is False:
            cm = cm.cpu().numpy()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if if_print:
            print("Normalized confusion matrix")
    else:
        if if_print:
            print('Confusion matrix, without normalization')
    if if_print:
        print(cm)
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=75)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path+'/'+save_name+'.png', dpi=300)
    plt.close()
