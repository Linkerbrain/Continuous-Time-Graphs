import time
import torch

import pandas as pd
import pytorch_lightning as pl
from torch import nn

from sgat.evaluation import mean_average_precision


def cmp_loss(pred, label):
    negatives = pred[~label]
    positives = pred[label]
    return torch.mean(negatives[:, None] - positives[None, :])

class SgatModule(pl.LightningModule):
    def __init__(self, graph, params, train_dataloader_gen, val_dataloader_gen):
        super(SgatModule, self).__init__()
        self.graph = graph
        self.params = params
        self.train_dataloader_gen = train_dataloader_gen
        self.val_dataloader_gen = val_dataloader_gen

        if self.params.loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif self.params.loss_fn == 'cmp':
            self.loss_fn = cmp_loss
        elif self.params.loss_fn == 'bce':
            self.loss_fn = nn.BCELoss(reduction='mean')

    @staticmethod
    def add_base_args(parser):
        parser.add_argument('--loss_fn', type=str, default='bce')
        parser.add_argument('--K', type=int, default=12)
        parser.add_argument('--no_MAP_random', action='store_true')
        parser.add_argument('--no_MAP_neighbour', action='store_true')

    @staticmethod
    def add_args(parser):
        pass

    def train_dataloader(self):
        return self.train_dataloader_gen(self.current_epoch)

    def val_dataloader(self):
        return self.val_dataloader_gen(self.current_epoch)

    def training_step(self, batch, batch_idx):
        # get targets
        predict_u = batch['u', 's', 'i'].edge_index[0]
        predict_i = batch['u', 's', 'i'].edge_index[1]

        # QUICK FIX FOR ILLEGITIMATE BATCHES
        if len(predict_u) == 0:
            return None

        # forward
        predictions = self.forward(batch, predict_u, predict_i)

        labels = batch['u', 's', 'i'].label

        # backward
        loss = self.loss_fn(predictions, labels)

        # log results for the neptune dashboard
        self.log('train/loss', loss, on_step=True)
        self.log('train/n_customers', float(batch['u'].code.shape[0]))
        self.log('train/n_target_customers', float(len(torch.unique(predict_u))))
        self.log('train/n_articles', float(batch['i'].code.shape[0]))
        self.log('train/n_target_articles', float(len(torch.unique(predict_i))))
        self.log('train/n_transactions', float(batch['u', 'b', 'i'].code.shape[0]))
        self.log('train/n_targets', float(batch['u', 's', 'i'].edge_index.shape[1]))
        self.log('train/time', time.time())
        self.log('train/positives_mean', torch.mean(predictions[labels.bool()]))
        self.log('train/negatives_mean', torch.mean(predictions[~labels.bool()]))

        return loss

    def random_MAP(self, batch):
        map_predict_u = batch['u', 'eval', 'i'].edge_index[0]
        map_predict_i = batch['u', 'eval', 'i'].edge_index[1]

        # make predictions
        map_predictions = self.forward(batch, map_predict_u, map_predict_i, predict_i_ptr=False)

        # get top k predictions (could be optimized)
        df = pd.DataFrame({
            'u': map_predict_u.cpu().numpy(),
            'i': map_predict_i.cpu().numpy(),
            'p': map_predictions.cpu().numpy()})

        y_pred = []
        for u, ip in df.groupby("u"):
            y_pred.append(ip.nlargest(self.params.K, 'p')['i'].values)

        n = 100 # num random fake items
        y_true = []
        for u, i in df.groupby("u"):
            y_true.append(i['i'].values[n:])

        MAP = mean_average_precision(y_true, y_pred, k=self.params.K)

        return MAP

    def neighbour_MAP(self, batch):
        map_predict_u = batch['u', 'n', 'i'].edge_index[0]
        map_predict_i = batch['u', 'n', 'i'].edge_index[1]

        # make predictions
        map_predictions = self.forward(batch, map_predict_u, map_predict_i)

        # get top k predictions (could be optimized)
        df = pd.DataFrame({
            'u': map_predict_u.numpy(),
            'i': map_predict_i.numpy(),
            'p': map_predictions.numpy()})

        y_pred = []
        for u, ip in df.groupby("u"):
            y_pred.append(ip.nlargest(self.params.K, 'p')['i'].values)

        # get targets
        supervised_predict_u = batch['u', 's', 'i'].edge_index[0]
        supervised_predict_i = batch['u', 's', 'i'].edge_index[1]

        df2 = pd.DataFrame({
            'u': supervised_predict_u[batch['u', 's', 'i'].label == 1],
            'i': supervised_predict_i[batch['u', 's', 'i'].label == 1],
        })

        y_true = []
        for u, i in df2.groupby("u"):
            y_true.append(i['i'].values)

        MAP = mean_average_precision(y_true, y_pred, k=12)
        return MAP

    def validation_step(self, batch, batch_idx):
        # get targets
        supervised_predict_u = batch['u', 's', 'i'].edge_index[0]
        supervised_predict_i = batch['u', 's', 'i'].edge_index[1]

        # forward
        supervised_predictions = self.forward(batch, supervised_predict_u, supervised_predict_i)

        # backward
        loss = self.loss_fn(supervised_predictions, batch['u', 's', 'i'].label)

        # COMPUTE MAP ON RANDOM EDGES
        if not self.params.no_MAP_random:
            MAP_random = self.random_MAP(batch)
            self.log('val/MAP_random', MAP_random, batch_size=len(supervised_predict_u))

        # COMPUTE MAP ON NEIGHBOURHOOD
        if not self.params.no_MAP_neighbour:
            MAP_neighbourhood = self.neighbour_MAP(batch)
            self.log('val/MAP_neighbour', MAP_neighbourhood, batch_size=len(supervised_predict_u))

        # log results for the neptune dashboard
        self.log('val/loss', loss, batch_size=len(supervised_predict_u))
        self.log('val/n_customers', float(batch['u'].code.shape[0]), batch_size=len(supervised_predict_u))
        self.log('val/n_articles', float(batch['i'].code.shape[0]), batch_size=len(supervised_predict_u))
        self.log('val/n_transactions', float(batch['u', 'b', 'i'].code.shape[0]), batch_size=len(supervised_predict_u))
        self.log('val/time', time.time(), batch_size=len(supervised_predict_u))


        return loss

    # def validation_epoch_end(self, validation_step_outputs):
    #     """
    #         Collects all results from validation batches into a predictions_df, and evaluates the predictions using
    #         the framework
    #     :param validation_step_outputs:
    #     :return:
    #     """
    #     top12 = torch.cat(list(v[0] for v in validation_step_outputs)).cpu().numpy()
    #     customers = torch.cat(list(v[1] for v in validation_step_outputs)).cpu().numpy()

    #     customers_df = self.framework.get_customers_data()
    #     articles_df = self.framework.get_articles_data()

    #     top12 = articles_df['article_id'].to_numpy()[top12]
    #     customers = customers_df['customer_id'].to_numpy()[customers]

    #     predictions_df = pd.DataFrame({"customer_id": customers, "pred": list(list(v) for v in top12)})

    #     scores = self.framework.evaluate(predictions_df) if not self.analyze else self.framework.analyze(predictions_df,
    #                                                                                                      sample_users=self.analyze)
    #     self.logger.log_model_summary(model=self, max_depth=-1)
    #     for k, v in scores.items():
    #         self.log(str(k), float(v))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
