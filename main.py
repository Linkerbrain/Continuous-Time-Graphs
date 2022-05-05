import argparse
import logging
import random
from typing import Sized, Iterable

import torch
import pytorch_lightning
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.loader import DataLoader

from sgat import data, graphs, models, Task, task

import pytorch_lightning as pl

from sgat import logger
from sgat.graphs import numpy_to_torch
from sgat.models.sgat_module import SgatModule
from sgat.simple_dataloader import SimpleDataLoaders
from sgat.models import mh
from sgat.models import dgsr

from sgat.no_traceback import no_traceback

from tqdm.auto import tqdm


class PrecomputedDataset(Iterable, Sized):
    def __init__(self, batches, shuffle=True, n_batches=None):
        self.batches = []
        for batch in tqdm(batches, total=n_batches):
            # add_oui(batch)
            batch = numpy_to_torch(batch)
            self.batches.append(batch)
        self.shuffle = shuffle

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        return iter(self.batches)


@task('Making dataset')
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


@task('Making data loaders')
def make_dataloaders(graph, params):
    if params.sampler == 'periodic':
        temporal_ds = graphs.TemporalDataset(graph, params)

        # PrecomputedDataset converts the arrays in the graphs to torch
        job = Task('Precomputing training set').start()
        train_data = PrecomputedDataset(temporal_ds.train_data(), shuffle=not params.noshuffle,
                                        n_batches=temporal_ds.train_data_len())
        job.done()

        job = Task('Precomputing validation and testing sets').start()
        val_data = PrecomputedDataset(temporal_ds.val_data(), shuffle=not params.noshuffle)
        test_data = PrecomputedDataset(temporal_ds.test_data(), shuffle=not params.noshuffle)
        job.done()

    elif params.sampler == 'simple':
        train_data, val_data, test_data = SimpleDataLoaders(graph, params)
    else:
        raise NotImplementedError()

    train_dataloader_gen = lambda _epoch: train_data
    val_dataloader_gen = lambda _epoch: val_data
    test_dataloader_gen = lambda _epoch: test_data

    return train_dataloader_gen, val_dataloader_gen, test_dataloader_gen


@task('Making model')
def make_model(graph, params, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen):
    if params.model == 'DGSR':
        model = models.dgsr.DGSR(graph, params, train_dataloader_gen, val_dataloader_gen)
    elif params.model == 'MH':
        model = models.mh.MH(graph, params, train_dataloader_gen, val_dataloader_gen)
    elif params.model == 'DUMMY':
        model = models.dummy.Dummy(graph, params, train_dataloader_gen, val_dataloader_gen)
    else:
        raise NotImplementedError()
    return model


def main(params):
    graph = make_dataset(params)

    logger.info(f"There are {graph['u'].code.shape[0]} users")
    logger.info(f"There are {graph['i'].code.shape[0]} items")
    logger.info(f"There are {graph['u', 'b', 'i'].code.shape[0]} transactions")

    train_dataloader_gen, val_dataloader_gen, test_dataloader_gen = make_dataloaders(graph, params)

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
                         log_every_n_steps=1, check_val_every_n_epoch=1 if not params.novalidate else int(10e9),
                         callbacks=[checkpoint_callback],
                         num_sanity_val_steps=2 if not params.novalidate else 0,
                         strategy=DDPStrategy(find_unused_parameters=False,
                                              static_graph=True) if params.devices > 1 else None)

    if not params.notrain:
        task = Task('Training model').start()
        trainer.fit(model)
        task.done()

    if not params.novalidate:
        task = Task('Validating model').start()
        trainer.validate(model)
        task.done()

    if not params.notest:
        task = Task('Testing model').start()
        trainer.test(model, test_dataloader_gen(model.current_epoch))
        task.done()

    # For interactive sessions/debugging
    return locals()


def subparse_model(subparser, name, module):
    parser_module = subparser.add_parser(name)
    models.sgat_module.SgatModule.add_base_args(parser_module)
    module.add_args(parser_module)

    gat_sampler_subparser = parser_module.add_subparsers(dest='sampler')

    parser_simple = gat_sampler_subparser.add_parser('simple')
    SimpleDataLoaders.add_args(parser_simple)

    parser_periodic = gat_sampler_subparser.add_parser('periodic')
    graphs.TemporalDataset.add_args(parser_periodic)


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
    # parser_train.add_argument('--batch_size', type=int, default=128)
    parser_train.add_argument('--accelerator', type=str, default='gpu')
    parser_train.add_argument('--precision', type=str, default='32')
    parser_train.add_argument('--devices', type=int, default=1)
    parser_train.add_argument('--load_checkpoint', type=str, default=None)
    parser_train.add_argument('--monitor', type=str, default='val/MAP_neighbour',
                              choices=['val/MAP_neighbour', 'val/MAP_random'])
    parser_train.add_argument('--notrain', action='store_true')
    parser_train.add_argument('--novalidate', action='store_true')
    parser_train.add_argument('--norich', action='store_true')
    parser_train.add_argument('--nocold', action='store_true')
    parser_train.add_argument('--nologger', action='store_true')
    parser_train.add_argument('--notest', action='store_true')
    parser_train.add_argument('--noshuffle', action='store_true')

    model_subparser = parser_train.add_subparsers(dest='model')

    subparse_model(model_subparser, 'DGSR', models.dgsr.DGSR)
    subparse_model(model_subparser, 'MH', models.mh.MH)
    subparse_model(model_subparser, 'DUMMY', models.dummy.Dummy)

    # Now parse the actual params
    params = parser.parse_args()

    no_traceback(main, params)
