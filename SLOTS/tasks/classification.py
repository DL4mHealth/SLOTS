import numpy as np
import sklearn
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import torch


def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear', fraction=None):
    """
    Args:
      fraction (Union[float, NoneType]): The fraction of training data. It used to do semi-supervised learning.
    """
    assert train_labels.ndim == 1 or train_labels.ndim == 2

    if fraction:
        # use first fraction number of training data
        train_data = train_data[:int(train_data.shape[0] * fraction)]
        train_labels = train_labels[:int(train_labels.shape[0] * fraction)]
        # print(f"Fraction of train data used for semi_supervised learning:{fraction}\n")

    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    if eval_protocol == 'linear':
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
    # test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    test_labels_onehot = test_labels_onehot = (
        F.one_hot(torch.tensor(test_labels).long(), num_classes=int(train_labels.max() + 1))).numpy()

    metrics_dict = {}
    pred_prob = y_score
    # print(pred_prob.shape)
    pred = pred_prob.argmax(axis=1)
    target = test_labels
    target_prob = test_labels_onehot
    # print(target_prob.shape)
    metrics_dict['Acc'] = sklearn.metrics.accuracy_score(target, pred)
    metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
    metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
    metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
    metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, average='macro', multi_class='ovr')
    metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob, average='macro')
    print(metrics_dict)

    # return y_score, { 'acc': acc, 'auprc': auprc }
    return y_score, acc