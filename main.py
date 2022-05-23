import argparse

from ctgraph import data, logger, train, stats
from ctgraph.cool_traceback import cool_traceback


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

    parser_stats = task_subparser.add_parser('stats')
    stats.add_args(parser_stats)

    parser_train = task_subparser.add_parser('train')
    train.add_args(parser_train)


    # Now parse the actual params
    params = parser.parse_args()

    return params

def main(params):
    if params.task == 'train':
        train.main(params)
    elif params.task == 'stats':
        stats.main(params)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    # parse parameters
    params = parse_params()
    logger.info(params)

    # start model
    cool_traceback(main, params)  # main(params)
