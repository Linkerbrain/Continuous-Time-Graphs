import argparse

from ctgraph import data, logger, train, stats, random_test
from ctgraph.cool_traceback import cool_traceback

import os


def parse_params(args=None):
    """
    Parse parameters
    """
    parser = argparse.ArgumentParser(description="Seasonal Graph Attention")
    parser.add_argument('--logging_level', type=str, default='INFO')
    parser.add_argument('--dataset', type=str, default='beauty', help='dataset name',
                        choices=['hm'] + data.AMAZON_DATASETS)
    parser.add_argument('--days', type=int, default=None, help='subset of the data to train and test with')
    parser.add_argument('--distribution', type=str, default=None)
    parser.add_argument('--info', type=str, default='', help='Random info you might want to log with the model')

    task_subparser = parser.add_subparsers(dest='task')

    parser_stats = task_subparser.add_parser('stats')
    stats.add_args(parser_stats)

    parser_train = task_subparser.add_parser('train')
    train.add_args(parser_train)

    parser_test = task_subparser.add_parser('random_test')
    random_test.add_args(parser_test)

    # Now parse the actual params
    params = parser.parse_args(args)

    return params


def main():
    # parse parameters
    params = parse_params()
    logger.info(params)

    if params.task == 'train':
        result_ = train.main(params)
    elif params.task == 'stats':
        result_ = stats.main(params)
    elif params.task == 'random_test':
        result_ = random_test.main(params)
    else:
        raise NotImplementedError()

    return params, result_


if __name__ == "__main__":

    if os.environ.get("NOCOOLTRACEBACK") == '1':
        params, result = main()
    else:
        params, result = cool_traceback(main)
