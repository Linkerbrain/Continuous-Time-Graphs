import argparse
import logging
import random

import numpy as np
import torch
from pytorch_lightning.loggers import NeptuneLogger

import ctgraph.datasets.periodic_dataset
from ctgraph import data, models, Task, task

import pytorch_lightning as pl

from ctgraph import logger
from ctgraph.graphs import numpy_to_torch, add_oui_and_oiu, get_last

from ctgraph.datasets.precomputed_dataset import PrecomputedDataset
from ctgraph.datasets.most_recent_neighbour_loader import MostRecentNeighbourLoader
from ctgraph.datasets.periodic_dataset import PeriodicDataset

from ctgraph.models.recommendation.module import RecommendationModule

from ctgraph.cool_traceback import cool_traceback

from tqdm.auto import tqdm

from os import path

import neptune.new as neptune


@task('Loading dataset')
def load_dataset(name, params):
    logger.info(f"Loading dataset '{name}' from disk...")
    dataset = PrecomputedDataset.load_from_disk(name)

    # get dataloaders
    train_data, val_data, test_data = dataset.get_loaders()

    train_dataloader_gen = lambda _epoch: train_data
    val_dataloader_gen = lambda _epoch: val_data
    test_dataloader_gen = lambda _epoch: test_data

    return dataset.graph, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen


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

    logger.info(f"There are {graph['u'].code.shape[0]} users")
    logger.info(f"There are {graph['i'].code.shape[0]} items")
    logger.info(f"There are {graph['u', 'b', 'i'].code.shape[0]} transactions")

    return graph


@task('Making data loaders')
def make_dataloaders(graph, name, params, neptune_logger=None):
    if params.sampler == 'periodic':
        # temporal_ds = ctgraph.datasets.periodic_dataset.PeriodicDataset(graph, params)
        raise NotImplementedError("periodic sampling is deprecated.")

    elif params.sampler == 'neighbour':
        # PrecomputedDataset converts the arrays in the graphs to torch

        logger.info(f"Creating dataset '{name}', this may take a bit...")

        job = Task('Sampling neighbour data...').start()
        neighbours = MostRecentNeighbourLoader(graph, params)
        job.done()

        job = Task('Creating dataset..').start()
        dataset = PrecomputedDataset(neighbours.yield_train, neighbours.yield_val, neighbours.yield_test, graph,
                                     batch_size=params.batch_size, noshuffle=params.noshuffle,
                                     num_workers=params.num_loader_workers, neptune_logger=neptune_logger)
        if not params.dontsave:
            dataset.save_to_disk(name)
            logger.info(f"Saved dataset '{name}' to disk!")

        # get dataloaders
        train_data, val_data, test_data = dataset.get_loaders()

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
    elif params.model == 'CTGR':
        model = ctgraph.models.recommendation.ctgr.CTGR(graph, params, train_dataloader_gen, val_dataloader_gen)
    elif params.model == 'DUMMY':
        model = ctgraph.models.recommendation.dummy.Dummy(graph, params, train_dataloader_gen, val_dataloader_gen)
    else:
        raise NotImplementedError()
    return model


@task('Making logger')
def make_logger(params):
    if not params.nologger:
        import neptune.new as neptune
        # Api key and proj name in env. NEPTUNE_API_TOKEN and NEPTUNE_PROJECT
        if params.load_checkpoint is None:
            run = neptune.init(tags=["training", "graph_nn"])
            neptune = NeptuneLogger(run=run)
        else:
            raise NotImplementedError
            # run = neptune.init(run=params.load_checkpoint)
            # neptune = NeptuneLogger(run=run)
            # checkpoint_path = neptune.experiment["training/model/best_model_path"].fetch()
            # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # model.load_state_dict(checkpoint['state_dict'])
        for k, v in vars(params).items():
            neptune.experiment[f'global/params/{k}'] = str(v)
        neptune.experiment[f'global/info'] = params.info
    else:
        neptune = None
        if params.load_checkpoint is not None:
            raise NotImplementedError("Need to use logger (neptune) for checkpoints")

    return neptune


def subparse_model(subparser, name, module):
    """
    Adds the parameters of the chosen model to the parser
    """
    parser_module = subparser.add_parser(name)
    RecommendationModule.add_base_args(parser_module)
    module.add_args(parser_module)

    gat_sampler_subparser = parser_module.add_subparsers(dest='sampler')

    parser_simple = gat_sampler_subparser.add_parser('neighbour')
    MostRecentNeighbourLoader.add_args(parser_simple)

    parser_periodic = gat_sampler_subparser.add_parser('periodic')
    PeriodicDataset.add_args(parser_periodic)


def parse_params():
    """
    Parse parameters
    """
    parser = argparse.ArgumentParser(description="Seasonal Graph Attention")
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--dataset', type=str, default='beauty', help='dataset name',
                        choices=['hm'] + data.AMAZON_DATASETS)
    parser.add_argument('--info', type=str, default='', help='Random info you might want to log with the model')

    task_subparser = parser.add_subparsers(dest='task')

    parser_submit = task_subparser.add_parser('submit')
    parser_submit.add_argument('--save', type=str)

    parser_train = task_subparser.add_parser('train')
    parser_train.add_argument('--days', type=int, default=None, help='subset of the data to train and test with')
    parser_train.add_argument('--epochs', type=int, default=1000)
    parser_train.add_argument('--batch_size', type=int, default=4)
    parser_train.add_argument('--batch_accum', type=int, default=32)
    parser_train.add_argument('--accelerator', type=str, default='gpu')
    parser_train.add_argument('--val_epochs', type=int, default=1)
    parser_train.add_argument('--num_loader_workers', type=int, default=1)
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
    parser_train.add_argument('--dontloadfromdisk', action='store_true')
    parser_train.add_argument('--dontsave', action='store_true')
    parser_train.add_argument('--seed', type=int, default=0)

    model_subparser = parser_train.add_subparsers(dest='model')

    subparse_model(model_subparser, 'DGSR', ctgraph.models.recommendation.dgsr.DGSR)
    subparse_model(model_subparser, 'CTGR', ctgraph.models.recommendation.ctgr.CTGR)
    subparse_model(model_subparser, 'DUMMY', ctgraph.models.recommendation.dummy.Dummy)

    # Now parse the actual params
    params = parser.parse_args()

    return params


def main(params):
    # Set all the seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    # initiate (Neptune) loader
    neptune_logger = make_logger(params)

    # experiment name of data
    data_name = f"{params.dataset}_{params.sampler}_n{params.n_max_trans}_m{params.m_order}_numuser{params.num_users}"

    if params.newsampler:
        data_name += "_newsampled"
        if params.sample_all:
            data_name += "_sampleall"

    # load from disk
    if path.exists(path.join("./precomputed_data/", data_name)) and not params.dontloadfromdisk:
        graph, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen = load_dataset(data_name, params)
    # or compute new data
    else:
        # parse entire dataset
        graph = make_dataset(params)

        # sample and make loaders
        train_dataloader_gen, val_dataloader_gen, test_dataloader_gen = make_dataloaders(graph, data_name, params,
                                                                                         neptune_logger)

    # initiate model
    model = make_model(graph, params, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen)

    # make trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=0, monitor=params.monitor, mode='max')

    trainer = pl.Trainer(max_epochs=params.epochs, logger=neptune_logger,  # track_grad_norm=2,
                         accumulate_grad_batches=params.batch_accum,
                         precision=int(params.precision) if params.precision.isdigit() else params.precision,
                         accelerator=params.accelerator,
                         devices=params.devices,
                         log_every_n_steps=1,
                         check_val_every_n_epoch=params.val_epochs if not params.novalidate else int(10e9),
                         callbacks=[checkpoint_callback],
                         num_sanity_val_steps=2 if not params.novalidate else 0,
                         strategy='ddp_sharded' if params.devices > 1 else None)

    # train
    if not params.notrain:
        task = Task('Training model').start()
        trainer.fit(model)
        task.done()

    # validate
    if not params.novalidate:
        task = Task('Validating model').start()
        trainer.validate(model)
        task.done()

    # test
    if not params.notest:
        task = Task('Testing model').start()
        trainer.test(model, test_dataloader_gen(model.current_epoch))
        task.done()

    # For interactive sessions/debugging
    return locals()


if __name__ == "__main__":
    # parse parameters
    params = parse_params()
    logger.info(params)

    # start model
    cool_traceback(main, params)  # main(params)
