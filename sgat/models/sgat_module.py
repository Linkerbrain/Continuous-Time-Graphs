import time
import torch

import pandas as pd
import pytorch_lightning as pl

class SgatModule(pl.LightningModule):
    def __init__(self, graph, params, train_dataloader_gen, val_dataloader_gen):
        super(SgatModule, self).__init__()
        self.graph = graph
        self.params = params
        self.train_dataloader_gen = train_dataloader_gen
        self.val_dataloader_gen = val_dataloader_gen

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

        # backward
        loss = self.loss_fn(predictions, batch['u', 's', 'i'].label)

        # log results for the neptune dashboard
        self.log('train/loss', loss, on_step=True)
        self.log('train/n_customers', float(batch['u'].code.shape[0]))
        self.log('train/n_articles', float(batch['i'].code.shape[0]))
        self.log('train/n_transactions', float(batch['u', 'b', 'i'].code.shape[0]))
        self.log('train/time', time.time())

        return loss

    def validation_step(self, batch, batch_idx):
        # get targets
        supervised_predict_u = batch['u', 's', 'i'].edge_index[0]
        supervised_predict_i = batch['u', 's', 'i'].edge_index[1]
    
        # forward
        supervised_predictions = self.forward(batch, supervised_predict_u, supervised_predict_i)
    
        # backward
        loss = self.loss_fn(supervised_predictions, batch['u', 's', 'i'].label)
    
        # neighbour_predict_u = batch['u', 'n', 'i'].edge_index[0]
        # neighbour_predict_i = batch['u', 'n', 'i'].edge_index[1]
    
        # neighbour_predictions = self.forward(batch, neighbour_predict_u, neighbour_predict_i)
    
        # df = pd.DataFrame(columns={'u': neighbour_predict_u.numpy(), 'i': neighbour_predict_i, 'p': neighbour_predictions.numpy()})
        # # df.groupby(by='u').apply(lambda x: x.sort_values(by='p', ascending=False).head(self.params.K))
        # # df.sort_values(by='p', ascending=False).groupby('u')
        # topK = df.groupby(by='u').nlargest(n=self.params.K)
    
        # import pdb; pdb.set_trace()
    
        MAP = 0.0

        self.log('val/MAP', MAP, batch_size=len(supervised_predict_u))
    
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