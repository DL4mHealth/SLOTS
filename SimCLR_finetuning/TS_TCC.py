import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
# sys.path.append("/mnt/disk1/data0/chlFiles/ts2vec-main/DEAP_load_dataset.py")
sys.path.append("/mnt/disk1/data0/chlFiles/TS-TCC-main/models/")
from models.model import base_Model,ProjectionHead
from models.TC import TC
from models.loss import NTXentLoss

# from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import sklearn
import math
from config_files.HAR_Configs import Config as Configs
configs = Configs()
sys.path.append("/mnt/disk1/data0/chlFiles/TS-TCC-main/dataloader/augmentations.py")
from dataloader.augmentations import DataTransform
# device = torch.device(args.device)

device='cuda'
class TS_TCC:
    '''The TS2Vec model'''

    def __init__(
            self,
            output_dims=320,
            hidden_dims=64,
            depth=10,
            device='cuda',
            lr=1e-4,
            batch_size=16,
            max_train_length=None,
            temporal_unit=0,
            after_iter_callback=None,
            after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.

        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (float): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''

        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit

        self.output_dims = output_dims
        self.hidden_dims = hidden_dims


        self.net = base_Model(configs).to(device)
        self.net = torch.optim.swa_utils.AveragedModel(self.net)
        self.net.update_parameters(self.net)

        self.net2 = TC(configs, device).to(device)
        self.net2 = torch.optim.swa_utils.AveragedModel(self.net2)
        self.net2.update_parameters(self.net2)


        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0

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

        optimizer = torch.optim.AdamW(list(self.net.parameters())+list(self.net2.parameters()), lr=self.lr)
        # optimizer2 = torch.optim.AdamW(self.net2.parameters(), lr=self.lr)
        loss_log = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0
            for [x_unlabeled] in train_unlabeled_loader:
                optimizer.zero_grad()
                aug1, aug2 = DataTransform(x_unlabeled, configs)
                aug1 = torch.from_numpy(aug1)
                aug2 = np.array(aug2, dtype=float)
                aug2 = torch.tensor(aug2)

                # x_labeled = x_labeled.float().to(device)
                # x_unlabeled = x_unlabeled.float().to(device)
                # y_labeled = y_labeled.to(device)
                aug1 = aug1.float().to(device)
                aug2 = aug2.float().to(device)

                predictions1, features1 = self.net(aug1)
                predictions2, features2 = self.net(aug2)

                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_lstm_feat1 = self.net2(features1, features2)
                temp_cont_loss2, temp_cont_lstm_feat2 = self.net2(features2, features1)

                # normalize projection feature vectors
                zis = temp_cont_lstm_feat1
                zjs = temp_cont_lstm_feat2

                zis_shape = zis.size(0)

                lambda1 = 0.5
                lambda2 = 0.5

                # lambda1 = 0.5
                # lambda2 = 0.5
                nt_xent_criterion = NTXentLoss(device, zis_shape, configs.Context_Cont.temperature,
                                               configs.Context_Cont.use_cosine_similarity)
                loss1 = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2

                # predictions3, features3 = model(x_labeled)
                # loss2 = cross_entropy_one_hot(predictions3, label_vec)

                loss = loss1
                # losstrain = loss1 + loss2
                torch.autograd.backward(loss)

                optimizer.step()
                self.net.update_parameters(self.net)
                self.net2.update_parameters(self.net2)

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
        self.proj_head = ProjectionHead(input_dims=2304, output_dims=2, hidden_dims=128).to(self.device)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        proj_head_optimizer = torch.optim.AdamW(self.proj_head.parameters(), lr=finetune_lr)
        # optimizer = torch.optim.AdamW(self.finetune_model.parameters(), lr=self.lr)

        criterion = nn.CrossEntropyLoss()

        # with torch.no_grad():
        epoch_loss_list, iter_loss_list, epoch_acc_list = [], [], []

        for epoch in range(finetune_epochs):
            for x, y in train_labeled_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.net.train()
                self.proj_head.train()
                optimizer.zero_grad()
                proj_head_optimizer.zero_grad()

                predictions, features = self.net(x)

                y_pred = self.proj_head(features)
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

        org_training = self.net.training
        self.net.eval()
        self.proj_head.eval()

        with torch.no_grad():
            for index, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)

                y = F.one_hot(y, num_classes=2).to(torch.float)

                predictions, features = self.net(x)

                y_pred_prob = self.proj_head(features).cpu()  # (n,n_classes)
                y_pred = y_pred_prob.argmax(dim=1).cpu()  # (n,)
                y_target = y.argmax(dim=1).cpu()  # (n,)
                y = y.cpu()
                """print(y_pred[:20])
                print(y[:20])
                break"""
                metrics_dict = {}
                metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(y_target, y_pred)
                metrics_dict['Precision'] = sklearn.metrics.precision_score(y_target, y_pred, average='macro')
                metrics_dict['Recall'] = sklearn.metrics.recall_score(y_target, y_pred, average='macro')
                metrics_dict['F1'] = sklearn.metrics.f1_score(y_target, y_pred, average='macro')
                metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(y, y_pred_prob, multi_class='ovr')
                metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(y, y_pred_prob)
                # print(metrics_dict)
                # print()
                # print(index)

        self.net.train(org_training)
        self.proj_head.train(org_training)

        # return metrics_dict['Accuracy']

        return metrics_dict

    def save(self, fn):
        ''' Save the model to a file.

        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)

    def load(self, fn):
        ''' Load the model from a file.

        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)

