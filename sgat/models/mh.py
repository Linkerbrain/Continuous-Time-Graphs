import time

import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv

from sgat.models import SgatModule

def cmp_loss(pred, label):
    negatives = pred[~label]
    positives = pred[label]
    return torch.mean(negatives[:, None] - positives[None, :])

class MH(SgatModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_vocab_num = self.graph['u'].code.shape[0]
        self.item_vocab_num = self.graph['i'].code.shape[0]

        self.user_embedding = nn.Embedding(self.user_vocab_num, self.params.embedding_size)
        self.item_embedding = nn.Embedding(self.item_vocab_num, self.params.embedding_size)

        self.convs = nn.Sequential()

        self.convs = nn.Sequential()

        # current_size = input_size
        current_size = 0
        for i in range(self.params.conv_layers):
            self.convs.add_module(f"sage_layer_{i}", HeteroConv({
                ('u', 'b', 'i'): SAGEConv(-1, self.params.embedding_size),
                ('i', 'rev_b', 'u'): SAGEConv(-1, self.params.embedding_size),
            }))

        # for the dot product at the end between the complete customer embedding and a candidate article
        self.transform = nn.Linear(self.params.embedding_size, self.params.embedding_size * (self.params.conv_layers+1))

        if self.params.activation is None:
            self.activation = lambda x: x
        else:
            self.activation = eval(f"torch.{self.params.activation}")

        self.dropout = nn.Dropout(self.params.dropout)

        if self.params.loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif self.params.loss_fn == 'cmp':
            self.loss_fn = cmp_loss
        elif self.params.loss_fn == 'bce':
            self.loss_fn = nn.BCELoss(reduction='mean')

    @staticmethod
    def add_args(parser):
        parser.add_argument('--embedding_size', type=int, default=50)
        parser.add_argument('--conv_layers', type=int, default=4)
        parser.add_argument('--activation', type=str, default=None)
        parser.add_argument('--dropout', type=float, default=0.25)
        parser.add_argument('--loss_fn', type=str, default='bce')
        parser.add_argument('--K', type=int, default=12)

    def forward(self, graph, predict_u, predict_i):

        # TODO: Add node features
        x_dict = {
            'u': self.user_embedding(graph['u'].code),
            'i': self.item_embedding(graph['i'].code)
        }

        edge_index_dict = {
            ('u', 'b', 'i'): graph['u', 'b', 'i'].edge_index,
            ('i', 'rev_b', 'u'): graph['u', 'b', 'i'].edge_index.flip(dims=(0,))
        }

        # TODO: edge_attr_dict with positional embeddings and such for GAT

        # TODO: Treat articles and users symmetrically: get layered embedding for both
        layered_embeddings_u = [x_dict['u']]
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}

            layered_embeddings_u.append(x_dict['u'])
        layered_embeddings_u = torch.cat(layered_embeddings_u, dim=1)

        # Grab the embeddings of the users and items who we will predict for
        layered_embeddings_u = layered_embeddings_u[predict_u]
        embeddings_i = x_dict['i'][predict_i]

        # predictions = torch.dot(layered_embeddings_u, self.transform(embeddings_i))
        predictions = torch.sum(layered_embeddings_u * self.transform(embeddings_i), dim=1)

        return torch.sigmoid(predictions)


    def training_step(self, batch, batch_idx):
        predict_u = batch['u', 's', 'i'].edge_index[0]
        predict_i = batch['u', 's', 'i'].edge_index[1]

        predictions = self.forward(batch, predict_u, predict_i)

        loss = self.loss_fn(predictions, batch['u', 's', 'i'].label)


        self.log('train/loss', loss, on_step=True)
        self.log('train/n_customers', float(batch['u'].code.shape[0]))
        self.log('train/n_articles', float(batch['i'].code.shape[0]))
        self.log('train/n_transactions', float(batch['u', 'b', 'i'].code.shape[0]))
        self.log('train/time', time.time())
        return loss

    # def validation_step(self, batch, batch_idx):
    #
    #     supervised_predict_u = batch['u', 's', 'i'].edge_index[0]
    #     supervised_predict_i = batch['u', 's', 'i'].edge_index[1]
    #
    #     supervised_predictions = self.forward(batch, supervised_predict_u, supervised_predict_i)
    #
    #     loss = self.loss_fn(supervised_predictions, batch['u', 's', 'i'].label)
    #
    #     neighbour_predict_u = batch['u', 'n', 'i'].edge_index[0]
    #     neighbour_predict_i = batch['u', 'n', 'i'].edge_index[1]
    #
    #     neighbour_predictions = self.forward(batch, neighbour_predict_u, neighbour_predict_i)
    #
    #     df = pd.DataFrame(columns={'u': neighbour_predict_u.numpy(), 'i': neighbour_predict_i, 'p': neighbour_predictions.numpy()})
    #     # df.groupby(by='u').apply(lambda x: x.sort_values(by='p', ascending=False).head(self.params.K))
    #     # df.sort_values(by='p', ascending=False).groupby('u')
    #     topK = df.groupby(by='u').nlargest(n=self.params.K)
    #
    #     import pdb; pdb.set_trace()
    #
    #     self.log('val/MAP', MAP,)
    #
    #     self.log('val/loss', loss)
    #     self.log('val/n_customers', float(batch['u'].code.shape[0]))
    #     self.log('val/n_articles', float(batch['i'].code.shape[0]))
    #     self.log('val/n_transactions', float(batch['u', 'b', 'i'].code.shape[0]))
    #     self.log('val/time', time.time())
    #     return loss

    # def validation_epoch_end(self, validation_step_outputs):
    #     """
    #         Collects all results from validation batches into a predictions_df, and evaluates the predictions using
    #         the framework
    #     :param validation_step_outputs:
    #     :return:
    #     """
    #     top12 = torch.cat(list(v[0] for v in validation_step_outputs)).cpu().numpy()
    #     customers = torch.cat(list(v[1] for v in validation_step_outputs)).cpu().numpy()
    #
    #     customers_df = self.framework.get_customers_data()
    #     articles_df = self.framework.get_articles_data()
    #
    #     top12 = articles_df['article_id'].to_numpy()[top12]
    #     customers = customers_df['customer_id'].to_numpy()[customers]
    #
    #     predictions_df = pd.DataFrame({"customer_id": customers, "pred": list(list(v) for v in top12)})
    #
    #     scores = self.framework.evaluate(predictions_df) if not self.analyze else self.framework.analyze(predictions_df,
    #                                                                                                      sample_users=self.analyze)
    #     self.logger.log_model_summary(model=self, max_depth=-1)
    #     for k, v in scores.items():
    #         self.log(str(k), float(v))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
