import argparse
import logging
import random
from typing import Sized, Iterable

import numpy as np
import torch
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import Dataset

import ctgraph.datasets.periodic_dataset
from ctgraph import data, models, Task, task

import pytorch_lightning as pl

from ctgraph import logger
from ctgraph.graphs import numpy_to_torch, add_oui_and_oiu

from ctgraph.datasets.neighbour_dataset import NeighbourDataset

from ctgraph.cool_traceback import cool_traceback

from tqdm.auto import tqdm


class PrecomputedDataset(Dataset, Iterable, Sized):
    def __init__(self, batches, shuffle=True, n_batches=None):
        self.batches = []
        for batch in tqdm(batches, total=n_batches):
            # add extra information
            add_oui_and_oiu(batch)

            # convert to pytorch
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

    return graph


@task('Making data loaders')
def make_dataloaders(graph, params):
    if params.sampler == 'periodic':
        temporal_ds = ctgraph.datasets.periodic_dataset.PeriodicDataset(graph, params)

        # PrecomputedDataset converts the arrays in the graphs to torch
        job = Task('Precomputing training set').start()
        train_data = PrecomputedDataset(temporal_ds.train_data(), shuffle=not params.noshuffle,
                                        n_batches=temporal_ds.train_data_len())
        job.done()

        job = Task('Precomputing validation and testing sets').start()
        val_data = PrecomputedDataset(temporal_ds.val_data(), shuffle=not params.noshuffle)
        test_data = PrecomputedDataset(temporal_ds.test_data(), shuffle=not params.noshuffle)
        job.done()

    elif params.sampler == 'neighbour':
        job = Task('Sampling neighbour dataset').start()
        neighbour_ds = NeighbourDataset(graph, params)
        job.done()

        # PrecomputedDataset converts the arrays in the graphs to torch
        job = Task('Creating neighbour training set').start()
        train_data = neighbour_ds.make_train_dataloader(batch_size=params.batch_size, shuffle=not params.noshuffle)
        job.done()

        job = Task('Precomputing validation and testing sets').start()
        val_data = neighbour_ds.make_val_dataloader(batch_size=params.batch_size, shuffle=not params.noshuffle)
        test_data = neighbour_ds.make_test_dataloader(batch_size=params.batch_size, shuffle=not params.noshuffle)
        job.done()
    else:
        raise NotImplementedError()

    train_dataloader_gen = lambda _epoch: train_data
    val_dataloader_gen = lambda _epoch: val_data
    test_dataloader_gen = lambda _epoch: test_data

    return train_dataloader_gen, val_dataloader_gen, test_dataloader_gen


@task('Making model')
def make_model(graph, params, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen):
    if params.model == 'DGSR':
        model = ctgraph.models.recommendation.dgsr.DGSR(graph, params, train_dataloader_gen, val_dataloader_gen)
    elif params.model == 'MH':
        model = ctgraph.models.recommendation.mh.MH(graph, params, train_dataloader_gen, val_dataloader_gen)
    elif params.model == 'DUMMY':
        model = ctgraph.models.recommendation.dummy.Dummy(graph, params, train_dataloader_gen, val_dataloader_gen)
    else:
        raise NotImplementedError()
    return model


def main(params):
    # Set all the seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

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
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=0, monitor=params.monitor, mode='max')

    trainer = pl.Trainer(max_epochs=params.epochs, logger=neptune,  # track_grad_norm=2,
                        precision=int(params.precision) if params.precision.isdigit() else params.precision,
                        accelerator=params.accelerator,
                        devices=params.devices,
                        log_every_n_steps=1, check_val_every_n_epoch=params.val_epochs if not params.novalidate else int(10e9),
                        callbacks=[checkpoint_callback],
                        num_sanity_val_steps=2 if not params.novalidate else 0,
                        strategy='ddp_sharded' if params.devices > 1 else None)

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
    ctgraph.models.recommendation.sgat_module.RecommendationModule.add_base_args(parser_module)
    module.add_args(parser_module)

    gat_sampler_subparser = parser_module.add_subparsers(dest='sampler')

    parser_simple = gat_sampler_subparser.add_parser('neighbour')
    NeighbourDataset.add_args(parser_simple)

    parser_periodic = gat_sampler_subparser.add_parser('periodic')
    ctgraph.datasets.periodic_dataset.PeriodicDataset.add_args(parser_periodic)


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
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--accelerator', type=str, default='gpu')
    parser_train.add_argument('--val_epochs', type=int, default=10)
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
    parser_train.add_argument('--seed', type=int, default=0)

    model_subparser = parser_train.add_subparsers(dest='model')

    subparse_model(model_subparser, 'DGSR', ctgraph.models.recommendation.dgsr.DGSR)
    subparse_model(model_subparser, 'MH', ctgraph.models.recommendation.mh.MH)
    subparse_model(model_subparser, 'DUMMY', ctgraph.models.recommendation.dummy.Dummy)

    # Now parse the actual params
    params = parser.parse_args()

    logger.info(params)

    cool_traceback(main, params)

    # main(params)
