import getpass
import logging
import random
import sys

import numpy as np
import torch
from pytorch_lightning.loggers import NeptuneLogger
from torch_geometric.loader import DataLoader

import ctgraph.datasets.periodic_dataset
from ctgraph import data, models, Task, task

import pytorch_lightning as pl

from ctgraph import logger
from ctgraph.graphs import numpy_to_torch, add_oui_and_oiu, add_last

from ctgraph.datasets.precomputed_dataset import PrecomputedDataset
from ctgraph.datasets.most_recent_neighbour_loader import MostRecentNeighbourLoader
from ctgraph.datasets.periodic_dataset import PeriodicDataset

from ctgraph.models.recommendation.module import RecommendationModule

from ctgraph.cool_traceback import cool_traceback

from tqdm.auto import tqdm

from os import path

import neptune.new as neptune

def add_args(parser):
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batch_accum', type=int, default=32)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--val_epochs', type=int, default=1)
    parser.add_argument('--num_loader_workers', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--monitor', type=str, default='val/dcg10_epoch')
    parser.add_argument('--notrain', action='store_true')
    parser.add_argument('--novalidate', action='store_true')
    parser.add_argument('--norich', action='store_true')
    parser.add_argument('--nocold', action='store_true')
    parser.add_argument('--nologger', action='store_true')
    parser.add_argument('--notest', action='store_true')
    parser.add_argument('--noshuffle', action='store_true')

    parser.add_argument('--save_embed', type=str, default=None)
    parser.add_argument('--load_embed', type=str, default=None)
    parser.add_argument('--freeze_embed', action='store_true')

    parser.add_argument('--dontloadfromdisk', action='store_true')
    parser.add_argument('--dontsave', action='store_true')
    parser.add_argument('--partial_save', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--replacement', action='store_true')
    parser.add_argument('--pin_memory', action='store_true')
    model_subparser = parser.add_subparsers(dest='model')

    subparse_model(model_subparser, 'DGSR', ctgraph.models.recommendation.dgsr.DGSR)
    subparse_model(model_subparser, 'CKCONV', ctgraph.models.recommendation.ckconv_model.CKConvModel)
    subparse_model(model_subparser, 'CTGR', ctgraph.models.recommendation.ctgr.CTGR)
    subparse_model(model_subparser, 'DUMMY', ctgraph.models.recommendation.dummy.Dummy)

    pass

@task('Loading dataset')
def load_dataset(name, params):
    logger.info(f"Loading dataset '{name}' from disk...")
    dataset = PrecomputedDataset.load_from_disk(name)

    # get dataloaders
    train_data, val_data, test_data = dataset.train_data, dataset.val_data, dataset.test_data

    return dataset.graph, train_data, val_data, test_data


@task('Making dataset')
def make_graph(params):
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


def make_dataloaders(train_data, val_data, test_data, params):
    train_sampler = torch.utils.data.RandomSampler(train_data, replacement=params.replacement, num_samples=params.samples)
    train_dataloader = DataLoader(train_data, batch_size=params.batch_size, pin_memory=params.pin_memory,
                                  shuffle=not params.noshuffle, num_workers=params.num_loader_workers)
    val_dataloader = DataLoader(val_data, batch_size=params.batch_size, pin_memory=params.pin_memory,
                                shuffle=False, num_workers=params.num_loader_workers)
    test_dataloader = DataLoader(test_data, batch_size=params.batch_size, pin_memory=params.pin_memory,
                                 shuffle=False, num_workers=params.num_loader_workers)

    train_dataloader_gen = lambda _epoch: train_dataloader
    val_dataloader_gen = lambda _epoch: val_dataloader
    test_dataloader_gen = lambda _epoch: test_dataloader

    return train_dataloader_gen, val_dataloader_gen, test_dataloader_gen


@task('Making data loaders')
def make_datasets(graph, name, params, neptune_logger=None):
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
                                     name=name, partial_save=params.partial_save, neptune_logger=neptune_logger)

        if not params.dontsave:
            dataset.save_to_disk()
            logger.info(f"Saved dataset '{name}' to disk!")

        job.done()

        # get dataloaders
        train_data, val_data, test_data = dataset.train_data, dataset.val_data, dataset.test_data

    else:
        raise NotImplementedError()

    return train_data, val_data, test_data


@task('Making model')
def make_model(graph, params, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen):
    if params.model == 'DGSR':
        model = ctgraph.models.recommendation.dgsr.DGSR(graph, params, train_dataloader_gen, val_dataloader_gen)
    elif params.model == 'CTGR':
        model = ctgraph.models.recommendation.ctgr.CTGR(graph, params, train_dataloader_gen, val_dataloader_gen)
    elif params.model == 'DUMMY':
        model = ctgraph.models.recommendation.dummy.Dummy(graph, params, train_dataloader_gen, val_dataloader_gen)
    elif params.model == 'CKCONV':
        model = ctgraph.models.recommendation.ckconv_model.CKConvModel(graph, params, train_dataloader_gen, val_dataloader_gen)
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
            run = neptune.init(run=params.load_checkpoint)
            neptune = NeptuneLogger(run=run)

        for k, v in vars(params).items():
            neptune.experiment[f'global/params/{k}'] = str(v)
        neptune.experiment[f'global/info'] = params.info
        neptune.experiment[f'global/command'] = ' '.join(sys.argv)
        neptune.experiment[f'global/username'] = getpass.getuser()
    else:
        neptune = None
        if params.load_checkpoint is not None:
            raise NotImplementedError("Need to use logger (neptune) for checkpoints")

    return neptune


def load_best_model(neptune_logger, model):
    checkpoint_path = neptune_logger.experiment["training/model/best_model_path"].fetch()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"Loaded best model '{checkpoint_path}'")

# saves embedding layer (can be done after loading in a checkpoint)
def save_embed(model, name, location='./embeddings'):
    torch.save(model.user_embedding, path.join(location, f'{name}_user_embedding.torch'))
    torch.save(model.item_embedding, path.join(location, f'{name}_item_embedding.torch'))

# loads an embedding layer
def load_embed(model, name, location='./embeddings'):
    model.user_embedding = torch.load(path.join(location, f'{name}_user_embedding.torch'))
    model.item_embedding = torch.load(path.join(location, f'{name}_item_embedding.torch'))

# turns off the gradient on the embedding layers
def freeze_embed(model):
    model.user_embedding.weight.requires_grad = False
    model.item_embedding.weight.requires_grad = False

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

    if params.partial_save:
        data_name += "_partial_save"

    # load from disk
    if path.exists(path.join("./precomputed_data/", data_name)) and not params.dontloadfromdisk:

        graph, train_data, val_data, test_data = load_dataset(data_name, params)
    # or compute new data
    else:
        # parse entire dataset
        graph = make_graph(params)

        # sample and make loaders
        train_data, val_data, test_data = make_datasets(graph, data_name, params,
                                                        neptune_logger)

    train_dataloader_gen, val_dataloader_gen, test_dataloader_gen = make_dataloaders(train_data, val_data, test_data,
                                                                                     params)

    # initiate model
    model = make_model(graph, params, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen)

    if params.load_checkpoint is not None:
        load_best_model(neptune_logger, model)
    else:
        neptune_logger.log_model_summary(model=model, max_depth=-1)

    # save desired attributes if you want
    if params.save_embed is not None:
        save_embed(model, name=params.save_embed)

    # modify model
    if params.load_embed is not None:
        load_embed(model, name=params.load_embed)

    if params.freeze_embed:
        freeze_embed(model)

    # make trainer
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=2, save_last=True, monitor=params.monitor, mode='max')

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

    if not params.nologger:
        # Wait for everything to be uploaded before we try to grab the best model
        neptune_logger.experiment.wait()

    # test
    if not params.notest:
        task = Task('Testing best model').start()
        load_best_model(neptune_logger, model)
        trainer.test(model, test_dataloader_gen(model.current_epoch))
        task.done()

    # For interactive sessions/debugging
    return locals()


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
