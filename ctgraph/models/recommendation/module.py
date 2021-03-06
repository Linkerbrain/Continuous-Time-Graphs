import copy
import time
import torch

import pandas as pd
import pytorch_lightning as pl
from torch import nn

from ctgraph import evaluation, logger
from ctgraph.randomize_time import randomize_time


def cmp_loss(pred, label):
    negatives = pred[~label]
    positives = pred[label]
    return torch.mean(negatives[:, None] - positives[None, :])


class RecommendationModule(pl.LightningModule):
    def __init__(self, graph, params, train_dataloader_gen, val_dataloader_gen):
        super(RecommendationModule, self).__init__()
        self.graph = graph
        self.params = params
        self.train_dataloader_gen = train_dataloader_gen
        self.val_dataloader_gen = val_dataloader_gen

        # evaluation setting, can be set outside
        self.randomize_time = False

        if self.params.loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
            assert self.params.train_style != 'dgsr_softmax'
        elif self.params.loss_fn == 'cmp':
            self.loss_fn = cmp_loss
            assert self.params.train_style != 'dgsr_softmax'
        elif self.params.loss_fn == 'bce':
            self.loss_fn = nn.BCELoss(reduction='mean')
            assert self.params.train_style != 'dgsr_softmax'
        elif self.params.loss_fn == 'ce':
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
            assert self.params.train_style == 'dgsr_softmax'

    @staticmethod
    def add_base_args(parser):
        parser.add_argument('--loss_fn', type=str, default='bce')
        parser.add_argument('--K', type=int, default=12)
        parser.add_argument('--no_MAP_random', action='store_true')
        parser.add_argument('--no_MAP_neighbour', action='store_true')
        parser.add_argument('--train_style', type=str, default='binary',
                            choices=['binary', 'dgsr_softmax', 'eval'])
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--decay', type=float, default=0.0001)
        parser.add_argument('--amsgrad', action='store_true')

    @staticmethod
    def add_args(parser):
        pass

    def train_dataloader(self):
        return self.train_dataloader_gen(self.current_epoch)

    def val_dataloader(self):
        return self.val_dataloader_gen(self.current_epoch)

    def training_step(self, batch, batch_idx, namespace='train'):
        predict_i = None

        if self.params.train_style == 'binary':
            """
            Binary classification edge yes/no per defined u, i pair
            """
            predict_u = batch['u', 's', 'i'].edge_index[0]
            predict_i = batch['u', 's', 'i'].edge_index[1]

            # forward
            predictions = self.forward(batch, predict_u, predict_i, predict_i_ptr=True)

            labels = batch['u', 's', 'i'].label

            # backward
            loss = self.loss_fn(torch.sigmoid(predictions), labels)

            positives_mean = torch.mean(predictions[labels.bool()])
            negatives_mean = torch.mean(predictions[~labels.bool()])
            
        elif self.params.train_style == 'dgsr_softmax':
            """
            Softmax over all items
            """
            predict_u = batch['target'].u_index  # + batch['u'].ptr[:-1]
            target_i_code = batch['target'].i_code

            # make prediction
            predictions = self.forward(batch, predict_u)
            if len(predictions.shape) == 1:
                predictions = predictions[None, :]

            # only real edges in target
            loss = self.loss_fn(predictions, target_i_code)

            positives_mean = torch.mean(predictions[:, target_i_code])
            negatives_mean = (torch.sum(predictions) - positives_mean * len(target_i_code)) / (
                    predictions.shape[1] - len(target_i_code))

        elif self.params.train_style == 'eval':
            predict_u = batch['eval'].u_index
            target_i_code = batch['eval'].i_code

            # forward
            predictions = self.forward(batch, predict_u, target_i_code, predict_i_ptr=False)

            labels = batch['eval'].label

            # backward
            loss = self.loss_fn(torch.sigmoid(predictions), labels)

            positives_mean = torch.mean(predictions[labels.bool()])
            negatives_mean = torch.mean(predictions[~labels.bool()])
        else:
            raise NotImplementedError()

        # This is not the number of users but the number of interactions used for supervision
        batch_size = len(predict_u)

        self.log(f'{namespace}/loss', loss, batch_size=batch_size, on_step=True, on_epoch=True)

        self.log(f'{namespace}/positives_mean', positives_mean, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log(f'{namespace}/negatives_mean', negatives_mean, batch_size=batch_size, on_step=True, on_epoch=True)

        self.log(f'{namespace}/n_customers', float(batch['u'].code.shape[0]), on_step=True, batch_size=batch_size)
        self.log(f'{namespace}/n_articles', float(batch['i'].code.shape[0]), on_step=True, batch_size=batch_size)

        if predict_i is not None:
            self.log(f'{namespace}/n_target_articles', float(len(torch.unique(predict_i))), on_step=True, batch_size=batch_size)
        self.log(f'{namespace}/n_target_customers', float(len(torch.unique(predict_u))), on_step=True, batch_size=batch_size)

        self.log(f'{namespace}/n_transactions', float(len(batch['u', 'b', 'i'].edge_index[0])), on_step=True, batch_size=batch_size)
        self.log(f'{namespace}/n_targets', float(batch_size), on_step=True, batch_size=batch_size)

        self.log(f'{namespace}/time', time.time(), on_step=True, on_epoch=True, batch_size=batch_size)

        return loss

    def random_MAP(self, batch):
        map_predict_u = batch['eval'].u_index
        map_predict_i_codes = batch['eval'].i_code

        # make predictions
        map_predictions = self.forward(batch, map_predict_u, map_predict_i_codes, predict_i_ptr=False)

        # get top k predictions (could be optimized)
        df = pd.DataFrame({
            'u': map_predict_u.cpu().numpy(),
            'i': map_predict_i_codes.cpu().numpy(),
            'p': map_predictions.cpu().numpy()})

        y_pred = []
        for u, ip in df.groupby("u"):
            y_pred.append(ip.nlargest(self.params.K, 'p')['i'].values)

        n = 100  # num random fake items
        y_true = []
        for u, i in df.groupby("u"):
            y_true.append(i['i'].values[n:])

        MAP = evaluation.mean_average_precision(y_true, y_pred, k=self.params.K)

        return MAP

    def random_metrics(self, batch, namespace):
        map_predict_u = batch['eval'].u_index
        map_predict_i_codes = batch['eval'].i_code

        # make predictions
        predictions = self.forward(batch, map_predict_u, map_predict_i_codes, predict_i_ptr=False)

        # get top k predictions (could be optimized)
        pred_df = pd.DataFrame({
            'u': map_predict_u.cpu().numpy(),
            'i': map_predict_i_codes.cpu().numpy(),
            'p': predictions.cpu().numpy()})

        target_df = pd.DataFrame({
            'u': map_predict_u[batch['eval'].label == 1].cpu().numpy(),
            'i': map_predict_i_codes[batch['eval'].label == 1].cpu().numpy(),
        })

        all_ranks = []
        for u, ip in pred_df.groupby("u"):
            ranking = ip.sort_values('p', ascending=False).reset_index()
            purchases = target_df.loc[target_df['u'] == u, 'i']
            ranks = ranking['i'].loc[ranking['i'].isin(purchases)].index
            all_ranks += list(ranks.values)

        recall5, recall10, recall20, dcg5, dcg10, dcg20 = evaluation.compute_eval_metrics(all_ranks, torch.unique(
            map_predict_u).cpu().numpy())

        batch_size = len(map_predict_u)

        self.log(f'{namespace}/recall5', recall5, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log(f'{namespace}/recall10', recall10, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log(f'{namespace}/recall20', recall20, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log(f'{namespace}/dcg5', dcg5, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log(f'{namespace}/dcg10', dcg10, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log(f'{namespace}/dcg20', dcg20, batch_size=batch_size, on_step=True, on_epoch=True)

    def neighbour_MAP(self, batch):
        # TODO: Check

        map_predict_u = batch['u', 'n', 'i'].edge_index[0]
        map_predict_i = batch['u', 'n', 'i'].edge_index[1]

        # make predictions
        map_predictions = self.forward(batch, map_predict_u, map_predict_i, predict_i_ptr=True)

        # get top k predictions (could be optimized)
        pred_df = pd.DataFrame({
            'u': map_predict_u.numpy(),
            'i': map_predict_i.numpy(),
            'p': map_predictions.numpy()}).sort_values('u')

        y_pred = []
        for u, ip in pred_df.groupby("u"):
            y_pred.append(ip.nlargest(self.params.K, 'p')['i'].values)

        # get targets
        supervised_predict_u = batch['u', 's', 'i'].edge_index[0]
        supervised_predict_i = batch['u', 's', 'i'].edge_index[1]

        target_df = pd.DataFrame({
            'u': supervised_predict_u[batch['u', 's', 'i'].label == 1],
            'i': supervised_predict_i[batch['u', 's', 'i'].label == 1],
        }).sort_values('u')

        y_true = []
        for u, i in target_df.groupby("u"):
            y_true.append(i['i'].values)

        MAP = evaluation.mean_average_precision(y_true, y_pred, k=12)
        return MAP

    def validation_step(self, batch, batch_idx, namespace='val', extra=False):
        # This prevents some kind of synchronization issue when num_workers > 0
        # See https://github.com/pytorch/pytorch/issues/973
        batch = copy.deepcopy(batch)

        # Just run a normal training_step, which logs the loss and everything it normally does
        # but doesn't do any training since we are in eval mode right now and the funciton itself
        # never performs gradient descent
        loss = self.training_step(batch, batch_idx, namespace=namespace)

        if self.params.train_style == 'binary':
            # Need it to give tha batch size
            supervised_predict_u = batch['u', 's', 'i'].edge_index[0]

            # COMPUTE MAP ON RANDOM EDGES
            if not self.params.no_MAP_random:
                MAP_random = self.random_MAP(batch)
                self.log(f'{namespace}/MAP_random', MAP_random, batch_size=len(supervised_predict_u), on_step=True,
                         on_epoch=True)

            # COMPUTE MAP ON NEIGHBOURHOOD
            if not self.params.no_MAP_neighbour:
                MAP_neighbourhood = self.neighbour_MAP(batch)
                self.log(f'{namespace}/MAP_neighbour', MAP_neighbourhood, batch_size=len(supervised_predict_u),
                         on_step=True, on_epoch=True)

        # Do a test_step but with the validation data, so the test set remains untouched
        self.random_metrics(batch, namespace)

        return loss

    def test_step(self, batch, batch_idx, namespace='test'):
        if self.randomize_time:
            namespace = 'test_RANDOM'
            batch = randomize_time(batch)

        # Do standard validation as well to compute the MAP scores
        loss = self.validation_step(batch, batch_idx, namespace=namespace, extra=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([ param for param in self.parameters() if param.requires_grad], lr=self.params.lr, weight_decay=self.params.decay,
                                     amsgrad=self.params.amsgrad)
        return optimizer
