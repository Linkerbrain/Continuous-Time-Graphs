import argparse
import logging

from sgat import data, graphs


def make_dataset(params):
    if params.dataset in data.AMAZON_DATASETS:
        if params.days is not None:
            logging.warning("Amazon datasets do not support subsetting with --days")
        raise NotImplementedError()
    elif params.dataset == 'hm':
        hm_full = data.HMData()
        hm = hm_full.subset(params.days) if params.days is not None else hm_full
        graph, dates = hm.as_graph()
    else:
        raise NotImplementedError()

    # mutates graph
    graphs.add_transaction_order(graph, dates)

    return graph, dates

def make_datalaoders(graph, dates, params):
    if params.sampler == 'periodic':
        temporal_ds = graphs.TemporalDataset(graph, dates, chunk_days=params.chunk_days)

        train_dataloader = temporal_ds.train_dataloader()
        val_dataloader = temporal_ds.val_dataloader()
        test_dataloader = temporal_ds.test_dataloader()
    elif params.sampler == 'ordered':
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    return train_dataloader, val_dataloader, test_dataloader

def main(params):
    graph, dates = make_dataset(params)

    train_dataloader, val_dataloader, test_dataloader = make_datalaoders(graph, dates, params)

    # TODO: make_model

    # TODO: train

    # TODO: test

    # For interactive sessions/debugging
    return locals()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seasonal Graph Attention")
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--dataset', type=str, default='beauty', help='dataset name',
                        choices=['hm'] + data.AMAZON_DATASETS)
    parser.add_argument('--days', type=int, default=None, help='number of days')
    parser.add_argument('--submit', type=str, default='', help='whether to make a submission')

    subparsers = parser.add_subparsers(dest='sampler')

    parser_periodic = subparsers.add_parser('periodic')
    parser_periodic.add_argument('--chunk_days', type=int, default=7)
    parser_periodic.add_argument('--test_chunks', type=int, default=1)

    parser_ordered = subparsers.add_parser('ordered')
    # Add args for ordered sampling here

    params = parser.parse_args()

    logging.basicConfig(level=getattr(logging, params.logging_level))

    r = main(params)
