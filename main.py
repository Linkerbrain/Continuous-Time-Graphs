import argparse
import logging

import torch
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.strategies import DDPStrategy

import sgat.models.gat
from sgat import data, graphs, models

import pytorch_lightning as pl

from sgat import logger

def make_dataset(params):
    if params.dataset in data.AMAZON_DATASETS:
        if params.days is not None:
            logging.warning("Amazon datasets do not support subsetting with --days")
        graph = data.amazon_dataset(params.dataset)
    elif params.dataset == 'hm':
        hm_full = data.HMData()
        hm = hm_full.subset(params.days) if params.days is not None else hm_full
        graph = hm.as_graph()
    else:
        raise NotImplementedError()

    # mutates graph
    graphs.add_transaction_order(graph)

    return graph


def make_datalaoders(graph, params):
    if params.sampler == 'periodic':
        temporal_ds = graphs.TemporalDataset(graph, params)

        train_dataloader = list(temporal_ds.train_dataloader())
        val_dataloader = list(temporal_ds.val_dataloader())
        test_dataloader = list(temporal_ds.test_dataloader())
        import pdb; pdb.set_trace()
    elif params.sampler == 'ordered':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    train_dataloader_gen = lambda _epoch: iter(train_dataloader)
    val_dataloader_gen = lambda _epoch: iter(val_dataloader)
    test_dataloader_gen = lambda _epoch: iter(test_dataloader)

    return train_dataloader_gen, val_dataloader_gen, test_dataloader_gen


def make_model(graph, params, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen):
    if params.model == 'DGSD':
        raise NotImplementedError()
    elif params.model == 'GAT':
        model = models.gat.GAT(graph, params, train_dataloader_gen, val_dataloader_gen)
    else:
        raise NotImplementedError()
    return model


def main(params):
    graph = make_dataset(params)

    logger.info(f"There are {graph['u'].code.shape[0]} users")
    logger.info(f"There are {graph['i'].code.shape[0]} items")
    logger.info(f"There are {graph['u', 'b', 'i'].code.shape[0]} transactions")

    train_dataloader_gen, val_dataloader_gen, test_dataloader_gen = make_datalaoders(graph, params)

    model = make_model(graph, params, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen)

    if not params.nologger:
        # Api key and proj name in env. NEPTUNE_API_TOKEN and NEPTUNE_PROJECT
        if params.load_checkpoint is None:
            neptune = NeptuneLogger(
                tags=["training", "graph_nn"],
            )
        else:
            import neptune.new as neptune
            run = neptune.init(run=params.load_checkpoint)
            neptune = NeptuneLogger(run=run)
            checkpoint_path = neptune.experiment["training/model/best_model_path"].fetch()
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
        for k, v in vars(params).items():
            neptune.experiment[f'global/params/{k}'] = str(v)
        neptune.experiment[f'global/info'] = params.info
    else:
        neptune = None
        if params.load_checkpoint is not None:
            raise NotImplementedError("Need to use logger (neptune) for checkpoints")

    # training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=2, monitor=params.monitor, mode='max')
    trainer = pl.Trainer(max_epochs=params.epochs, logger=neptune,  # track_grad_norm=2,
                         precision=int(params.precision) if params.precision.isdigit() else params.precision,
                         accelerator=params.accelerator,
                         devices=params.devices,
                         log_every_n_steps=50, check_val_every_n_epoch=10, callbacks=[checkpoint_callback],
                         num_sanity_val_steps=2 if not params.novalidate else 0,
                         strategy=DDPStrategy(find_unused_parameters=False,
                                              static_graph=True) if params.devices > 1 else None)

    if not params.notrain:
        logger.info('Training model...')
        trainer.fit(model)
        logger.info('Done')

    if not params.novalidate:
        logger.info('Validating model...')
        trainer.validate(model)
        logger.info('Done')

    if not params.notest:
        logger.info('Testing model...')
        trainer.test(model, test_dataloader_gen(model.current_epoch))
        logger.info('Done')

    # For interactive sessions/debugging
    return locals()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seasonal Graph Attention")
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--dataset', type=str, default='beauty', help='dataset name',
                        choices=['hm'] + data.AMAZON_DATASETS)
    parser.add_argument('--info', type=str, default='', help='Random info you might want to log with the model')

    task_subparser = parser.add_subparsers(dest='task')

    parser_submit = task_subparser.add_parser('submit')
    parser_submit.add_argument('--save', type=str)

    parser_train = task_subparser.add_parser('train')
    # parser_train.add_argument('--model', type=str, default='DGSR', choices={'DGSR', "GAT"})
    parser_train.add_argument('--days', type=int, default=None, help='subset of the data to train and test with')
    parser_train.add_argument('--epochs', type=int, default=1000)
    parser_train.add_argument('--batch_size', type=int, default=128)
    parser_train.add_argument('--accelerator', type=str, default='gpu')
    parser_train.add_argument('--precision', type=str, default='32')
    parser_train.add_argument('--devices', type=int, default=1)
    parser_train.add_argument('--load_checkpoint', type=str, default=None)
    parser_train.add_argument('--monitor', type=str, default='MAP', choices=['MAP', 'NDCG'])
    parser_train.add_argument('--notrain', action='store_true')
    parser_train.add_argument('--novalidate', action='store_true')
    parser_train.add_argument('--norich', action='store_true')
    parser_train.add_argument('--nocold', action='store_true')
    parser_train.add_argument('--nologger', action='store_true')
    parser_train.add_argument('--notest', action='store_true')

    model_subparser = parser_train.add_subparsers(dest='model')

    parser_dgsr = model_subparser.add_parser('DGSR')
    # models.dgsr.add_args(parser_dgsr)

    parser_gat = model_subparser.add_parser('GAT')
    models.gat.GAT.add_args(parser_gat)

    gat_sampler_subparser = parser_gat.add_subparsers(dest='sampler')

    parser_periodic = gat_sampler_subparser.add_parser('periodic')
    graphs.TemporalDataset.add_args(parser_periodic)
    # parser_periodic.add_argument('--loader', type=str, default='full')
    # parser_periodic.add_argument('--group_noise', type=float, default=1)

    # parser_ordered = gat_sampler_subparser.add_parser('ordered')
    # Add args for ordered sampling here

    params = parser.parse_args()

    r = main(params)
