import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
from multiprocessing import Pool

from adaptivehmc import AdHMC
from pyhmc import PyHMC
import diagnostics as dg

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=2, type=int, help='Dimensions')
parser.add_argument('--nsamples',  default=1000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=100, type=int, help='Number of burning samples')
parser.add_argument('--lpath',  default=1, type=float, help='Nleapfrog*step_size')
parser.add_argument('--step_size', default=0.1, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--two_factor', default=2, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--nchains',  default=1, type=int, help='Number of chains')
parser.add_argument('--nparallel',  default=8, type=int, help='Number of parallel iterations for map')

parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
ndim = args.ndim
step_size = args.step_size
two_factor = args.two_factor
nsamples = args.nsamples
Lpath = args.lpath
suffix = args.suffix
burnin = args.burnin
nchains = args.nchains
nparallel = args.nparallel
Nleapfrog = int(Lpath / step_size)


##LOG PROB
def log_prob(p):
        return - 0.5 * (p**2).sum()

def grad_log_prob(p):
        return - p

#log_prob = samples.log_prob
#grad_log_prob = samples.grad_log_prob
hmc = AdHMC(log_prob, grad_log_prob)
hmc_vanilla = PyHMC(log_prob, grad_log_prob)
#def step(x):
#    return  hmc.hmc_step(x, Nleapfrog, smin=0.01, smax=1.0)
        

######
def do_hmc():
    
    samples = []
    #hmc = PyHMC(log_prob, grad_log_prob)
    initstate = np.random.uniform(-1., 1., size=nchains*ndim).reshape([nchains, ndim])
    q = initstate

    
    start = time.time()
    for i in range(nsamples):
        q =  hmc.hmc_step(q, Nleapfrog, smin=0.01, smax=1.0)[0]
        samples.append(q)
        #print(pf)
        #print(pb)
        #print(eps)
        #print(np.prod(pf), np.prod(pb), np.prod(pb)/np.prod(pf))

    qs = np.array(samples)[100:]
    print('Time taken : ', time.time() - start)
    
    start = time.time()
    samples = []
    for i in range(nsamples):
        q =  hmc_vanilla.hmc_step(q, Nleapfrog, step_size=np.random.uniform(0.01, 1.0))[0]
        samples.append(q)
    qs2 = np.array(samples)[100:]
    print('Time taken : ', time.time() - start)

    print(qs.mean(axis=0), qs.std(axis=0))
    print(qs2.mean(axis=0), qs2.std(axis=0))

    plt.hist(qs[:, 0, 0], alpha=0.5, bins='auto')
    plt.hist(qs2[:, 0, 0], alpha=0.5, bins='auto')
    plt.savefig('test.png')
    #

if __name__=="__main__":
    do_hmc()
