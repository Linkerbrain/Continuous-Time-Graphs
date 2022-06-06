from cProfile import run
import torch
import argparse

import neptune.new as neptune
from pytorch_lightning.loggers import NeptuneLogger

from ctgraph import data, models, Task, task, logger
from ctgraph.train import make_model, load_best_model, load_dataset, make_dataloaders

import pytorch_lightning as pl

from main import parse_params

""" --testnormaltoo
python main.py random_test --load_checkpoint CTGRLOD-33
"""

# global arguments
def add_args(parser):
    parser.add_argument('--load_checkpoint', type=str)
    parser.add_argument('--testnormaltoo', action='store_true')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--preconditions', type=str, default=None)

# load in saved arguments
@task('Loading Neptune')
def init_neptune(checkpoint_name):
    run = neptune.init(run=checkpoint_name)
    neptune_logger = NeptuneLogger(run=run)

    return neptune_logger

@task('Loading Parameters')
def load_params(neptune_logger):
    param_dic = neptune_logger.experiment["global/params"].fetch()

    params = parse_params(args=[param_dic['task'], param_dic['model'], param_dic['sampler']])

    # hacky way to use saved parameters
    # parser = argparse.ArgumentParser()
    for k, v in param_dic.items():
        # Bools
        if v == 'True' or v =='False':
            # parser.add_argument('--' + k, default=v=='True', type=bool)
            setattr(params, k, v == 'True')
            continue

        # String that looks like int
        if k =='precision':
            # parser.add_argument('--' + k, default=v)
            setattr(params, k, v)
            continue

        # Ints
        try:
            if str(int(v)) == v:
                # parser.add_argument('--' + k, default=int(v), type=int)
                setattr(params, k, int(v))
                continue
        except:
            pass

        # Floats
        try:
            if str(float(v)) == v:
                # parser.add_argument('--' + k, default=float(v), type=float)
                setattr(params, k, float(v))
                continue
        except:
            pass

        # None's
        if v == 'None':
            # parser.add_argument('--' + k, default=None)
            setattr(params, k, None)
        # Strings
        else:
            # parser.add_argument('--' + k, default=v)
            setattr(params, k, v)
    # params = parser.parse_args(args=[])

    return params

# load in run
@task('Loading Data & Model')
def load_run(neptune_logger, params, newparams):
    # load data
    data_name = f"{params.dataset}_{params.sampler}_n{params.n_max_trans}_m{params.m_order}_numuser{params.num_users}"
    if params.newsampler:
        data_name += "_newsampled"
    if params.sample_all:
        data_name += "_sampleall"
    if params.partial_save:
        data_name += "_partial_save"
        
    graph, train_data, val_data, test_data = load_dataset(data_name, params)

    # make dataloaders
    params.batch_size = newparams.batch_size
    train_dataloader_gen, val_dataloader_gen, test_dataloader_gen = make_dataloaders(train_data, val_data, test_data,
                                                                                        params)
    # load model
    model = make_model(graph, params, train_dataloader_gen, val_dataloader_gen, test_dataloader_gen)
    load_best_model(neptune_logger, model)

    model = model.eval()

    return train_dataloader_gen, val_dataloader_gen, test_dataloader_gen, model


def main(newparams):
    # connect to neptune
    neptune_logger = init_neptune(newparams.load_checkpoint)

    preconditions = dict()
    for pair in newparams.preconditions.split(',') if newparams.preconditions is not None else []:
        k, v = pair.split(':')
        preconditions[k] = v

    # load params of run
    params = load_params(neptune_logger)

    for k, v in params.__dict__.items():
        if k in preconditions and str(v) != preconditions[k]:
            logger.info("Skipped run. Preconditions unsatisfied.")
            return locals()


    # make model
    train_dataloader_gen, val_dataloader_gen, test_dataloader_gen, model = load_run(neptune_logger, params, newparams)

    # --- set up testing ---
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

    # test model normal
    if newparams.testnormaltoo:
        model.randomize_time = False

        task = Task('Testing model with NORMAL TIME').start()
        trainer.test(model, test_dataloader_gen(model.current_epoch))
        task.done()


    # test model randomized
    model.randomize_time = True

    task = Task('Testing model with RANDOM TIME').start()
    trainer.test(model, test_dataloader_gen(model.current_epoch))
    task.done()

    # For interactive sessions/debugging
    return locals()