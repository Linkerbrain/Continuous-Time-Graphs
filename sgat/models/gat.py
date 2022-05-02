import torch
from pytorch_lightning import LightningModule
from torch import nn

from sgat.models import SgatModule


class GAT(SgatModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_vocab_num = self.graph['u'].code.shape[0]
        self.item_vocab_num = self.graph['i'].code.shape[0]

        self.user_embedding = nn.Embedding(self.user_vocab_num, self.params.embedding_size)
        self.item_embedding = nn.Embedding(self.item_vocab_num, self.params.embedding_size)

        self.convs = nn.Sequential()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--embedding_size', type=int, default=50)
        parser.add_argument('--conv_layers', type=int, default=4)


    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_size, filter=None):
        """

        :param filter:
        :param batch_size:
        :param x_dict:
        :param edge_index_dict:
        :param edge_attr_dict:
        :return: [n_customers, n_articles]
        """

        # TODO: Incorporate edge attributes
        if self.pre_embed:
            x_dict['customer'] = self.embed_customer(x_dict['customer'])
            x_dict['article'] = self.embed_article(x_dict['article'])

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        poi_embeddings = x_dict['customer'][:batch_size]
        article_embeddings = x_dict['article']


        poi_embeddings_ = poi_embeddings[None, :, :].repeat(article_embeddings.shape[0], 1, 1)
        article_embeddings_ = article_embeddings[:, None, :].repeat(1, poi_embeddings.shape[0], 1)

        pairs = torch.cat((poi_embeddings_, article_embeddings_), dim=2).transpose(0, 1)
        cold_starts = ~torch.isin(torch.arange(batch_size).to(self.device),
                                  edge_index_dict['customer', 'buys', 'article'][0])

        rich_pairs = pairs[~cold_starts]
        cold_pairs = pairs[cold_starts]

        rich_predictions = self.rich(rich_pairs).squeeze(-1)
        cold_predictions = self.cold(cold_pairs).squeeze(-1)

        predictions = torch.zeros((batch_size, article_embeddings.shape[0]), dtype=self.datatype()).to(self.device)
        predictions[~cold_starts] = rich_predictions
        predictions[cold_starts] = cold_predictions

        # pairs_ = pairs.flatten(end_dim=1)[filter] if filter is not None else pairs
        # predictions = self.linear(pairs_)
        # return predictions.squeeze(-1)
        return torch.sigmoid(predictions)

    def training_step(self, train_batch, batch_idx):
        batch_size = train_batch['customer'].batch_size

        # Count rev_buys because these don't get hidden when batch_mask=0
        # Also count them here because they are overriden later
        # This is just for logging
        n_transactions = train_batch['article', 'rev_buys', 'customer'].edge_index.shape[1]

        x_dict = train_batch.x_dict

        edge_index = train_batch['customer', 'buys', 'article'].edge_index
        edge_attr = train_batch['customer', 'buys', 'article'].edge_attr
        n_edges = edge_index.shape[1]

        # Rearrange the reverse edges so that they match up with the forward edges
        # since they are the same anyways this comes down to just recreating them
        rev_edge_index = edge_index.flip(dims=(0,))
        rev_edge_attr = edge_attr

        # If batch_mask is false, then the dataset is already masked so edge_mask will not filter anything
        edge_mask = torch.ones(n_edges, dtype=torch.bool).to(self.device)
        poie = edge_index[0] < batch_size
        if self.batch_mask:
            # Create a random filter for the edges, 1 => Edge is kept in the graph, 0 => Edge is used for the supervised part
            # Remove random transactions from the batch customers (persons of interest)
            edge_mask[poie] = (1 - torch.div(
                torch.randint(self.partition_ratio, (torch.sum(poie).item(),)).to(self.device),
                (self.partition_ratio - 1), rounding_mode='floor'
            )).bool()

        # Apply the filter to the graph
        edge_index_dict = train_batch.edge_index_dict
        edge_attr_dict = train_batch.edge_attr_dict
        edge_index_dict['customer', 'buys', 'article'] = edge_index[:, edge_mask]
        edge_attr_dict['customer', 'buys', 'article'] = edge_attr[edge_mask]
        edge_index_dict['article', 'rev_buys', 'customer'] = rev_edge_index[:, edge_mask]
        edge_attr_dict['article', 'rev_buys', 'customer'] = rev_edge_attr[edge_mask]

        # All transactions that have happened on this graph for the poi's, masked or not
        # Customer-mayor ordering
        purchased = torch.zeros(batch_size, x_dict['article'].shape[0], dtype=torch.bool).to(self.device)
        poieip = edge_index[:, poie]
        if not self.batch_mask:
            # Base the purchased transactions on the hidden edges
            hidden_purchases = train_batch['customer', 'buys_hidden', 'article'].edge_index
            # Hidden transactions from poi's with an article that was already in the graph
            hpoie = (hidden_purchases[0] < batch_size) & (torch.isin(hidden_purchases[1], edge_index[1]))
            poieip = torch.cat((poieip, hidden_purchases[:, hpoie]), dim=1)
        purchased[poieip[0], poieip[1]] = 1

        # Masked transactions
        masked = torch.zeros(batch_size, x_dict['article'].shape[0], dtype=torch.bool).to(self.device)
        if not self.batch_mask:
            poieim = hidden_purchases[:, hpoie]
        else:
            poieim = edge_index[:, ~edge_mask & poie]
        masked[poieim[0], poieim[1]] = 1

        z = self.forward(x_dict, edge_index_dict, edge_attr_dict, batch_size, filter=None)

        negatives = ~purchased
        positives = masked

        if not self.do_rich and self.do_cold:
            cold_starts = ~torch.isin(torch.arange(batch_size).to(self.device),
                                      edge_index_dict['customer', 'buys', 'article'][0])
            negative_examples = z.flatten()[(negatives & cold_starts[:, None]).flatten()]
            positive_examples = z.flatten()[(positives & cold_starts[:, None]).flatten()]
        else:
            negative_examples = z.flatten()[negatives.flatten()]
            positive_examples = z.flatten()[positives.flatten()]

        # loss = F.binary_cross_entropy(z.flatten()[filter], y)
        if self.loss == 'mse':
            negative_loss = torch.sum(negative_examples ** 2)
            positive_loss = torch.sum((1 - positive_examples) ** 2)
            # a/x + b/y = (a + b/y*x)/x '= (a*y + b*x)/x/y
            loss = positive_loss / len(positive_examples) + negative_loss / len(negative_examples)
            # loss = (positive_loss * len(negative_examples) + negative_loss * len(positive_examples)) / (
            #             len(negative_examples) + len(positive_examples))
        elif self.loss == 'cmp':
            loss = torch.sum(negative_examples[:, None] - positive_examples[None, :]) / len(negative_examples) / len(
                positive_examples)
        else:
            raise Exception
        negatives_mean = torch.mean(z.flatten()[negatives.flatten()])
        positives_mean = torch.mean(z.flatten()[positives.flatten()])

        # This hook can be used for debugging but otherwise does nothing
        self.hook('training_step', locals())

        if self.do_log:
            self.log('negatives_mean', negatives_mean)
            self.log('positives_mean', positives_mean)
            self.log('train_loss', loss, on_step=True)
            if self.loss == 'mse':
                self.log('positive_loss', positive_loss, on_step=True)
                self.log('negative_loss', negative_loss, on_step=True)
            self.log('n_customers', float(train_batch['customer'].x.shape[0]))
            self.log('n_articles', float(train_batch['article'].x.shape[0]))
            self.log('n_transactions', float(n_transactions))
            # self.log('_y', y.sum().item(), on_step=True)
            # self.log('_z', z.mean().sum().item(), on_step=True)
            # self.log('_amount', float(len(z)), on_step=True)
            # self.log('_maxN', torch.max(negative_examples).item())
            # self.log('_maxP', torch.max(positive_examples).item())
            # self.log('_minN', torch.min(negative_examples).item())
            # self.log('_minP', torch.min(positive_examples).item())
            self.log('time', time.time())
        return loss

    def validation_step(self, test_batch, batch_index):
        batch_size = test_batch['customer'].batch_size

        n_transactions = test_batch['article', 'rev_buys', 'customer'].edge_index.shape[1]

        x_dict = test_batch.x_dict

        edges = test_batch['customer', 'buys', 'article']
        edge_index = edges.edge_index
        edge_attr = edges.edge_attr

        # Rearrange the reverse edges so that they match up with the forward edges
        # since they are the same anyway this comes down to just recreating them
        rev_edge_index = edge_index.flip(dims=(0,))
        rev_edge_attr = edge_attr

        # Add reverse edges
        edge_index_dict = test_batch.edge_index_dict
        edge_attr_dict = test_batch.edge_attr_dict
        edge_index_dict['article', 'rev_buys', 'customer'] = rev_edge_index
        edge_attr_dict['article', 'rev_buys', 'customer'] = rev_edge_attr

        z = self.forward(x_dict, edge_index_dict, edge_attr_dict, batch_size, filter=None)

        ranking = z.argsort(descending=True, axis=1)
        top12 = test_batch['article'].code[ranking[:, :12]]
        customers = test_batch['customer'].code[:batch_size]

        self.log('n_customers_val', float(test_batch['customer'].x.shape[0]), batch_size=batch_size)
        self.log('n_articles_val', float(test_batch['article'].x.shape[0]), batch_size=batch_size)
        self.log('n_transactions_val', float(n_transactions), batch_size=batch_size)

        return top12, customers

    def validation_epoch_end(self, validation_step_outputs):
        """
            Collects all results from validation batches into a predictions_df, and evaluates the predictions using
            the framework
        :param validation_step_outputs:
        :return:
        """
        top12 = torch.cat(list(v[0] for v in validation_step_outputs)).cpu().numpy()
        customers = torch.cat(list(v[1] for v in validation_step_outputs)).cpu().numpy()

        customers_df = self.framework.get_customers_data()
        articles_df = self.framework.get_articles_data()

        top12 = articles_df['article_id'].to_numpy()[top12]
        customers = customers_df['customer_id'].to_numpy()[customers]

        predictions_df = pd.DataFrame({"customer_id": customers, "pred": list(list(v) for v in top12)})

        scores = self.framework.evaluate(predictions_df) if not self.analyze else self.framework.analyze(predictions_df,
                                                                                                         sample_users=self.analyze)
        self.logger.log_model_summary(model=self, max_depth=-1)
        for k, v in scores.items():
            self.log(str(k), float(v))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
