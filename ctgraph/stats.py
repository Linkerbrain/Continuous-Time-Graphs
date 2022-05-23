import torch
from torch import nn
from torch.distributions import Exponential

from ctgraph import train, logger

from scipy.stats import expon

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISTRIBUTIONS_PATH = 'distributions/'

DISTRIBUTIONS = ['expon']


class Distribution(nn.Module):
    def __init__(self, name):
        super(Distribution, self).__init__()
        self.name = name
        if name == 'expon':
            self.pre = lambda x: 1 - x
            self.fit = lambda x: expon.fit(self.pre(x))[1]
            # exponential = Exponential(torch.tensor())
            # self.forward = lambda x, p: expon.pdf(self.pre(x), *p)
            self.forward_function = lambda x, p: torch.exp(-1/p*self.pre(x))*1/p
        else:
            raise NotImplementedError()

    def forward(self, x, p):
        return self.forward_function(x, p)


class ParameterizedDistribution(Distribution):
    def __init__(self, name, p):
        super(ParameterizedDistribution, self).__init__(name)
        self.p = p

    def forward(self, x):
        return super(ParameterizedDistribution, self).forward(x, self.p)

    @staticmethod
    def load(params):
        p = torch.load(DISTRIBUTIONS_PATH + get_pars(params))
        return ParameterizedDistribution(params.distribution, p)

def add_args(parser):
    pass

def get_pars(params):
    pars = f"{params.dataset}_{params.days}_{params.distribution}"
    return pars

def compute_distribution(params, graph, name):
    pars = get_pars(params)

    f = Distribution(params.distribution)

    t = graph['u', 'b', 'i'].t

    p = f.fit(t)

    logger.info(f"Fitted {name} to {params.dataset}.t resulting in parameters {p}")

    torch.save(p, DISTRIBUTIONS_PATH + pars)

    fig = plt.figure()
    a1 = fig.add_axes([0, 0, 1, 1])
    a1.hist(t, bins=50, alpha=0.2)
    a2 = a1.twinx()

    x = np.linspace(0, 1, 100)
    a2.plot(x, f.forward(torch.tensor(x), p).numpy())

    plt.savefig(DISTRIBUTIONS_PATH + pars + '__figure')


def main(params):
    graph = train.make_graph(params)

    if params.distribution is None:
        dlist = DISTRIBUTIONS
    else:
        dlist = [params.distribution]

    for d in dlist:
        compute_distribution(params, graph, d)

    #
    # plt.hist(t, bins=50)
    #
    # expon.fit(t)
    #
    # plt.show()

    return graph