import logging
import os
import sys
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
from model import TFC,target_classifier
from configs import Config
from loss import *
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# from utils import save_config_file, accuracy, save_checkpoint
import sklearn
from sklearn import metrics

torch.manual_seed(0)
# from train_model import FCN
configs = Config()
class TFC_fine():

    def __init__(self, device='cuda',lr=0.0003,weight_decay=1e-4,after_iter_callback=None,
            after_epoch_callback=None):
        super().__init__()
        # self.args = kwargs['args']
        # self.model = kwargs['model'].to(self.args.device)
        self._net = TFC(configs).to(device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        # self.optimizer = kwargs['optimizer']
        # self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter()
        # logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        # self.criterion = MixUpLoss(device)
        self.device = 'cuda'

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0
        self.lr=0.0003


    def fit(self, train_unlabeled_loader, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model.

        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.

        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''

        # optimizer = torch.optim.AdamW(list(self.net.parameters())+list(self.net2.parameters()), lr=self.lr)
        # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        loss_log = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0
            for [x_unlabeled,x_unlabeled_aug1, x_unlabeled_f, x_unlabeled_f_aug1_f] in train_unlabeled_loader:
                optimizer.zero_grad()


                x_unlabeled = x_unlabeled.float().to(self.device)  # data: [128, 1, 178], labels: [128]
                x_unlabeled_aug1 = x_unlabeled_aug1.float().to(self.device)  # aug1 = aug2 : [128, 1, 178]
                x_unlabeled_f, x_unlabeled_f_aug1_f = x_unlabeled_f.float().to(self.device), x_unlabeled_f_aug1_f.float().to(self.device)  # aug1 = aug2 : [128, 1, 178]

                """Produce embeddings"""
                h_t, z_t, h_f, z_f = self._net(x_unlabeled, x_unlabeled_f)
                h_t_aug, z_t_aug, h_f_aug, z_f_aug = self._net(x_unlabeled_aug1, x_unlabeled_f_aug1_f)

                """Compute Pre-train loss"""
                """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
                nt_xent_criterion = NTXentLoss_poly(self.device, configs.batch_size, configs.Context_Cont.temperature,
                                                    configs.Context_Cont.use_cosine_similarity)  # device, 128, 0.2, True

                loss_t = nt_xent_criterion(h_t, h_t_aug)
                loss_f = nt_xent_criterion(h_f, h_f_aug)
                l_TF = nt_xent_criterion(z_t, z_f)  # this is the initial version of TF loss

                lam = 0.2
                loss = lam * (loss_t + loss_f) + l_TF

                torch.autograd.backward(loss)

                optimizer.step()
                self.net.update_parameters(self._net)
                # self.net2.update_parameters(self.net2)

                cum_loss += loss.item()
                n_epoch_iters += 1

            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        return loss_log


    def finetune_fit(self, train_labeled_loader, test_loader, mask=None, encoding_window=None,
                     casual=False, sliding_length=None, sliding_padding=0, batch_size=None, finetune_epochs=20,
                     finetune_lr=0.001):
        ''' Compute representations using the model.

        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.

        Returns:
            repr: The representations for data.
        '''

        # projection head append after encoder
        self.proj_head = target_classifier(configs).to(self.device)


        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        proj_head_optimizer = torch.optim.AdamW(self.proj_head.parameters(), lr=finetune_lr)

        # optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        # proj_head_optimizer = torch.optim.AdamW(self.proj_head.parameters(), lr=finetune_lr)
        # optimizer = torch.optim.AdamW(self.finetune_model.parameters(), lr=self.lr)

        criterion = nn.CrossEntropyLoss()

        # with torch.no_grad():
        epoch_loss_list, iter_loss_list, epoch_acc_list = [], [], []

        for epoch in range(finetune_epochs):
            for [x_labeled, y_labeled,_, x_labeled_f, _] in train_labeled_loader:

                self.net.train()
                self.proj_head.train()
                optimizer.zero_grad()
                proj_head_optimizer.zero_grad()

                x_labeled = x_labeled.to(self.device)

                _, z_tt, _, z_ff = self.net(x_labeled, x_labeled_f)
                fea_concat = torch.cat((z_tt, z_ff), dim=1)

                # _,features = self.net(x_labeled)
                y_labeled=y_labeled.to(self.device)

                y_pred = self.proj_head(fea_concat).to(self.device)
                loss = criterion(y_pred, y_labeled)
                loss.backward()
                optimizer.step()
                proj_head_optimizer.step()
                iter_loss_list.append(loss.item())

            epoch_loss_list.append(sum(iter_loss_list) / len(iter_loss_list))

            # print(f"Epoch number: {epoch}")
            # print(f"Loss: {epoch_loss_list[-1]}")
            # print(f"Accuracy: {accuracy}")
            performance = self.finetune_predict(test_loader)
            # accuracy = self.finetune_predict(test_data, test_labels, encoding_window=encoding_window)
            # epoch_acc_list.append(accuracy)

        return epoch_loss_list[-1], performance
        # return epoch_loss_list, epoch_acc_list

        # self.net.train(org_training)
        # return output.numpy()
        # return output.cpu().numpy()

    def finetune_predict(self, test_loader, mask=None, encoding_window=None, casual=False,
                         sliding_length=None, sliding_padding=0):
        # test_dataset = TensorDataset(torch.from_numpy(test_data).to(torch.float),
        #                              F.one_hot(torch.from_numpy(test_labels).to(torch.long), num_classes=2).to(
        #                                  torch.float))
        # # treat whole test set as a batch
        # test_loader = DataLoader(test_dataset, batch_size=test_data.shape[0])

        org_training = self.net.training
        self.net.eval()
        self.proj_head.eval()

        with torch.no_grad():
            for [x_labeled, y_labeled,_, x_labeled_f, _] in test_loader:
                x_labeled = x_labeled.to(self.device)

                y_labeled=y_labeled.to(self.device)

                y_labeled = F.one_hot(y_labeled, num_classes=2).to(torch.float)

                _, z_tt, _, z_ff = self.net(x_labeled, x_labeled_f)
                fea_concat = torch.cat((z_tt, z_ff), dim=1)

                # _,features = self.net(x)

                y_pred_prob = self.proj_head(fea_concat).cpu()  # (n,n_classes)
                y_pred = y_pred_prob.argmax(dim=1).cpu()  # (n,)
                y_target = y_labeled.argmax(dim=1).cpu()  # (n,)
                y = y_labeled.cpu()
                """print(y_pred[:20])
                print(y[:20])
                break"""
                metrics_dict = {}
                metrics_dict['Accuracy'] = metrics.accuracy_score(y_target, y_pred)
                metrics_dict['Precision'] = metrics.precision_score(y_target, y_pred, average='macro')
                metrics_dict['Recall'] = metrics.recall_score(y_target, y_pred, average='macro')
                metrics_dict['F1'] = metrics.f1_score(y_target, y_pred, average='macro')
                metrics_dict['AUROC'] = metrics.roc_auc_score(y, y_pred_prob, multi_class='ovr')
                metrics_dict['AUPRC'] = metrics.average_precision_score(y, y_pred_prob)
                # print(metrics_dict)
                # print()
                # print(index)

        self.net.train(org_training)
        self.proj_head.train(org_training)

        # return metrics_dict['Accuracy']

        return metrics_dict


