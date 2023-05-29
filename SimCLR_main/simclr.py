import logging
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# from utils import save_config_file, accuracy, save_checkpoint
import sklearn
from sklearn import metrics
from augmentations import DataTransform
torch.manual_seed(0)
from loss import info_nce_loss
from models.resnet_simclr import ResNetSimCLR,ProjectionHead
class SimCLR():

    def __init__(self, arch='resnet18',out_dim=128,device='cuda',lr=0.0003,weight_decay=1e-4,after_iter_callback=None,
            after_epoch_callback=None):
        super().__init__()
        # self.args = kwargs['args']
        # self.model = kwargs['model'].to(self.args.device)
        self.model = ResNetSimCLR(base_model=arch, out_dim=out_dim)
        # self.optimizer = kwargs['optimizer']
        # self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter()
        # logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.device = 'cuda'
        self.model = ResNetSimCLR(base_model=arch, out_dim=out_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)

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
        # optimizer2 = torch.optim.AdamW(self.net2.parameters(), lr=self.lr)
        loss_log = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0
            for [x_unlabeled] in train_unlabeled_loader:
                aug1, aug2 = DataTransform(x_unlabeled)
                aug1 =aug1[:,:30,:]
                aug2 = aug2[:, :30, :]
                aug1 = aug1.reshape(aug1.shape[0],3,-1,aug1.shape[2])
                aug2 = aug2.reshape(aug2.shape[0], 3, -1, aug2.shape[2])
                aug1 = torch.from_numpy(aug1)
                aug2 = np.array(aug2, dtype=float)
                aug2 = torch.tensor(aug2)

                # x_labeled = x_labeled.float().to(device)
                # x_unlabeled = x_unlabeled.float().to(device)
                # y_labeled = y_labeled.to(device)
                # aug1 = aug1.float().to(self.device)
                # aug2 = aug2.float().to(self.device)
                aug1 = aug1.float()
                aug2 = aug2.float()


                features1 = self.model(aug1).to(self.device)
                features2 = self.model(aug2).to(self.device)
                features = torch.cat([features1, features2], dim=0)
                # label_num = 200

                logits, labels = info_nce_loss(features, self.device)
                loss = self.criterion(logits, labels)
                # losstrain = loss1 + loss2
                torch.autograd.backward(loss)

                self.optimizer.step()
                # self.net.update_parameters(self.net)
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
        self.proj_head = ProjectionHead(input_dims=128, output_dims=2, hidden_dims=64).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        proj_head_optimizer = torch.optim.AdamW(self.proj_head.parameters(), lr=finetune_lr)

        # optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        # proj_head_optimizer = torch.optim.AdamW(self.proj_head.parameters(), lr=finetune_lr)
        # optimizer = torch.optim.AdamW(self.finetune_model.parameters(), lr=self.lr)

        criterion = nn.CrossEntropyLoss()

        # with torch.no_grad():
        epoch_loss_list, iter_loss_list, epoch_acc_list = [], [], []

        for epoch in range(finetune_epochs):
            for x, y in train_labeled_loader:
                x = x[:, :30, :]
                x = x.reshape(x.shape[0], 3, -1, x.shape[2])
                # x, y = x.to(self.device), y.to(self.device)
                self.model.train()
                self.proj_head.train()
                optimizer.zero_grad()
                proj_head_optimizer.zero_grad()

                features = self.model(x).to(self.device)
                y=y.to(self.device)

                y_pred = self.proj_head(features).to(self.device)
                loss = criterion(y_pred, y)
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

        org_training = self.model.training
        self.model.eval()
        self.proj_head.eval()

        with torch.no_grad():
            for index, (x, y) in enumerate(test_loader):
                x = x[:, :30, :]
                x = x.reshape(x.shape[0], 3, -1, x.shape[2])
                # x, y = x.to(self.device), y.to(self.device)
                y=y.to(self.device)

                y = F.one_hot(y, num_classes=2).to(torch.float)

                features = self.model(x).to(self.device)

                y_pred_prob = self.proj_head(features).cpu()  # (n,n_classes)
                y_pred = y_pred_prob.argmax(dim=1).cpu()  # (n,)
                y_target = y.argmax(dim=1).cpu()  # (n,)
                y = y.cpu()
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

        self.model.train(org_training)
        self.proj_head.train(org_training)

        # return metrics_dict['Accuracy']

        return metrics_dict


