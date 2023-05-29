import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
# sys.path.append("/mnt/disk1/data0/chlFiles/ts2vec-main/DEAP_load_dataset.py")
sys.path.append("/mnt/disk1/data0/chlFiles/ts2vec-main2023/models/encoder.py")
from models import TSEncoder, ProjectionHead
from models.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import sklearn
import math
from models import TSEncoder, ProjectionHead, Classifier
from models.losses import hierarchical_contrastive_loss
import model_projection
import supervised_contrastive_loss



class TS2Vec:
    '''The TS2Vec model'''

    def __init__(
            self,
            input_dims,
            output_dims=320,
            hidden_dims=64,
            depth=3,
            device='cuda',
            lr=0.001,
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

        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(
            self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)



        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model.

        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.

        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True,
                                  drop_last=True)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)


    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=out.size(1),
            ).transpose(1, 2)

        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size=encoding_window,
                stride=1,
                padding=encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]

        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)

        else:
            if slicing is not None:
                out = out[:, slicing]

        # return out.cpu()
        return out

    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0,
               batch_size=None):
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
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)

        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0): min(r, ts_l)],
                            left=-l if l < 0 else 0,
                            right=r - ts_l if r > ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding + sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0

                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size=out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)

                output.append(out)

            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        # return output.numpy()
        return output.cpu().numpy()

    def finetune_fit(self, train_data, train_labels, test_data, test_labels, mask=None, encoding_window=None,
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
        assert self.net is not None, 'please train or load a net first'
        assert train_data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = train_data.shape

        # projection head append after encoder
        self.proj_head = ProjectionHead(input_dims=self.output_dims, output_dims=2, hidden_dims=128).to(self.device)

        # org_training = self.net.training
        # self.net.eval()

        # (n_samples,) is mapped to a target of shape (n_samples, n_classes). Use for calc AUROC
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float),
                                      F.one_hot(torch.from_numpy(train_labels).to(torch.long), num_classes=2).to(
                                          torch.float))
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        proj_head_optimizer = torch.optim.AdamW(self.proj_head.parameters(), lr=finetune_lr)
        # optimizer = torch.optim.AdamW(self.finetune_model.parameters(), lr=self.lr)

        criterion = nn.CrossEntropyLoss()

        # with torch.no_grad():
        epoch_loss_list, iter_loss_list, epoch_acc_list = [], [], []

        for epoch in range(finetune_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.net.train()
                self.proj_head.train()
                optimizer.zero_grad()
                proj_head_optimizer.zero_grad()

                out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)  # B x output_dims

                y_pred = self.proj_head(out).squeeze(1)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                proj_head_optimizer.step()
                iter_loss_list.append(loss.item())

            epoch_loss_list.append(sum(iter_loss_list) / len(iter_loss_list))

            # print(f"Epoch number: {epoch}")
            # print(f"Loss: {epoch_loss_list[-1]}")
            # print(f"Accuracy: {accuracy}")
            performance = self.finetune_predict(test_data, test_labels, encoding_window=encoding_window)
            # accuracy = self.finetune_predict(test_data, test_labels, encoding_window=encoding_window)
            # epoch_acc_list.append(accuracy)

        return epoch_loss_list[-1], performance
        # return epoch_loss_list, epoch_acc_list

        # self.net.train(org_training)
        # return output.numpy()
        # return output.cpu().numpy()

    def finetune_predict(self, test_data, test_labels, mask=None, encoding_window=None, casual=False,
                         sliding_length=None, sliding_padding=0):
        test_dataset = TensorDataset(torch.from_numpy(test_data).to(torch.float),
                                     F.one_hot(torch.from_numpy(test_labels).to(torch.long), num_classes=2).to(
                                         torch.float))
        # treat whole test set as a batch
        test_loader = DataLoader(test_dataset, batch_size=test_data.shape[0])

        org_training = self.net.training
        self.net.eval()
        self.proj_head.eval()

        with torch.no_grad():
            for index, (x, y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)

                out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                if encoding_window == 'full_series':
                    out = out.squeeze(1)  # B x output_dims

                y_pred_prob = self.proj_head(out).squeeze(1).cpu()  # (n,n_classes)
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

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TS2Vec_fine:
    '''The TS2Vec model'''

    def __init__(
            self,
            input_dims,
            output_dims=320,
            hidden_dims=64,
            depth=3,
            device='cuda',
            lr=0.001,
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

        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims).to(
            self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)



        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        self.n_epochs = 0
        self.n_iters = 0

    # def hotEncoder(self,v):
    #     ret_vec = torch.zeros(v.shape[0], 2).to(self.device)
    #     for s in range(v.shape[0]):
    #         ret_vec[s][v[s]] = 1
    #     return ret_vec

    def hotEncoder(self,v):
        ret_vec = torch.zeros(v.shape[0], 3).to(self.device)
        for s in range(v.shape[0]):
            ret_vec[s][v[s]] = 1
        return ret_vec

    def cross_entropy_one_hot(self,input, target):
        _, labels = target.max(dim=1)
        ls = nn.CrossEntropyLoss()(input, labels).to(self.device)
        return ls

    def fit(self, train_unlabeled_loader, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model.
        '''
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        loss_log = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0
            n_epoch_iters = 0
            for [x_unlabeled] in train_unlabeled_loader:
                optimizer.zero_grad()
                # x_labeled = x_labeled.to(self.device)
                x_unlabeled = x_unlabeled.to(self.device)
                # y_labeled = y_labeled.to(self.device)

                '''unsupervised hierarchical_contrastive_loss'''
                x_l = (1 * x_unlabeled).to(self.device)
                x_r = (1 * x_unlabeled).to(self.device)
                mask1 = generate_binomial_mask(x_l.size(0), x_l.size(1)).to(self.device)
                x_l[~mask1] = 0
                mask2 = generate_binomial_mask(x_r.size(0), x_r.size(1)).to(self.device)
                x_r[~mask2] = 0


                x_l = x_l.unsqueeze(1)
                x_r = x_r.unsqueeze(1)

                out1 = self._net(x_l).to(self.device)
                out1 = out1.unsqueeze(1)
                out1 = F.normalize(out1, dim=0)

                out2 = self._net(x_r).to(self.device)
                out2 = out2.unsqueeze(1)
                out2 = F.normalize(out2, dim=0)

                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=0
                )
                # losstrain = loss1 + loss2
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

        '''

        self.projection_layer = model_projection.ProjectionModel().to(self.device)

        contrastiveLoss = supervised_contrastive_loss.SupConLoss(temperature=0.1)

        self.classifier = Classifier(input_dims=128, output_dims=3).to(self.device)

        # contrastiveLoss = hierarchical_contrastive_loss()
        criterion = nn.CrossEntropyLoss()
        # projection head append after encoder
        # self.proj_head = ProjectionHead(input_dims=self.output_dims, output_dims=2, hidden_dims=128).to(self.device)


        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
        proj_head_optimizer = torch.optim.AdamW(self.projection_layer.parameters(), lr=finetune_lr)
        classifier_optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=finetune_lr)
        # optimizer = torch.optim.AdamW(self.finetune_model.parameters(), lr=self.lr)

        criterion = nn.CrossEntropyLoss()

        # with torch.no_grad():
        epoch_loss_list, iter_loss_list, epoch_acc_list = [], [], []

        for epoch in range(finetune_epochs):
            for x_labeled, y_labeled in train_labeled_loader:
                x_labeled, y_labeled = x_labeled.to(self.device), y_labeled.to(self.device)
                self.net.train()
                # self.proj_head.train()
                self.projection_layer.train()
                self.classifier.train()
                optimizer.zero_grad()
                proj_head_optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                label_vec = self.hotEncoder(y_labeled)
                x_labeled = x_labeled.unsqueeze(1)
                out = self.net(x_labeled).to(self.device)
                out = F.normalize(out, dim=0)

                y_proj = self.projection_layer(out)
                y_proj = F.normalize(y_proj, dim=0)
                y_proj = y_proj.unsqueeze(1)

                loss2 = contrastiveLoss(y_proj, y_labeled)

                y_pred_prob = self.classifier(out)
                loss3 = self.cross_entropy_one_hot(y_pred_prob, label_vec)
                # loss = loss2+loss3
                loss = loss3
                loss.backward()
                optimizer.step()
                proj_head_optimizer.step()
                classifier_optimizer.step()
                iter_loss_list.append(loss.item())

            epoch_loss_list.append(sum(iter_loss_list) / len(iter_loss_list))

            # print(f"Epoch number: {epoch}")
            # print(f"Loss: {epoch_loss_list[-1]}")
            # print(f"Accuracy: {accuracy}")
            performance = self.finetune_predict(test_loader, encoding_window=encoding_window)
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
        self.classifier.eval()
        # self.proj_head.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                # out = eval_with_pooling(model, x, encoding_window='full_series')

                x = x.unsqueeze(1)
                out = self.net(x).to(self.device)
                out = F.normalize(out, dim=0)

                y_pred_prob = self.classifier(out).cpu()
                y = y.cpu()
                # test_labels_onehot = (F.one_hot(y.long(), 2)).numpy()
                test_labels_onehot = (F.one_hot(y.long(), 3)).numpy()

                pred_prob = y_pred_prob
                # print(pred_prob.shape)
                pred = pred_prob.argmax(axis=1)
                target = y
                target_prob = test_labels_onehot
                # print(target_prob.shape)
                metrics_dict = {}
                metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(target, pred)
                metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
                metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
                metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
                metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, average='macro',
                                                                      multi_class='ovr')
                metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob, average='macro')
                # print(metrics_dict)
                # print()
                # print(index)

        self.net.train(org_training)
        self.classifier.train(org_training)

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

