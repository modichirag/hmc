import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
import json, pickle
import pystan
import utils

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


from pyhmc import PyHMC_multistep, PyHMC
import diagnostics as dg

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nsamples',  default=100, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=100, type=int, help='Number of burning samples')
parser.add_argument('--lpath',  default=5, type=float, help='Nleapfrog*step_size')
parser.add_argument('--step_size', default=0.01, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--two_factor', default=2, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--nchains',  default=10, type=int, help='Number of chains')
parser.add_argument('--nparallel',  default=8, type=int, help='Number of parallel iterations for map')

parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')


args = parser.parse_args()
#ndim = args.ndim
step_size = args.step_size
two_factor = args.two_factor
nsamples = args.nsamples
Lpath = args.lpath
suffix = args.suffix
burnin = args.burnin
nchains = args.nchains
nparallel = args.nparallel
Nleapfrog = int(Lpath / step_size)

print("\nwith step size %0.3f and %d steps\n"%( step_size, Nleapfrog))
##
fpath= '/mnt/ceph/users/cmodi/hmc/outputs_long/stoch_voltality//'
try: os.makedirs(fpath)
except Exception as e: print(e)
fpathd = fpath + 'default/'
fpath = fpath + 'defaulthmc/'
try: os.makedirs(fpath)
except Exception as e: print(e)
fpath2 = fpath + 'metrics/'
try: os.makedirs(fpath2)
except Exception as e: print(e)


#######

fname = './models/stoch_voltality.pkl'
with open(fname, 'rb') as f:
    model = pickle.load(f)
with open('stoch_voltality.json', 'r') as fp:
    data = json.load(fp)


with open(fpathd + 'params.json', 'r') as fp:
    todump = json.load(fp)
    stansamples = model.sampling(data=data, chains=1, warmup=100, 
                                      iter=200, seed=rank, n_jobs=1,
                                      control={"metric":"diag_e",
                                                "stepsize_jitter":0,
                                                "adapt_delta":0.9 })



samplesy = []
for key in stansamples.extract().keys() : 
    y = stansamples.extract()[key]
    print(key, y.shape)
    if len(y.shape) == 1: y = np.expand_dims(y, 1)
    samplesy.append(y)
samplesy = np.concatenate(samplesy[:-1], axis=1)    
samplesy = np.expand_dims(samplesy, 1)
ndim = samplesy.shape[-1]

stepsizefid = comm.bcast(todump['stepsize'], root=0)
invmetricfid = comm.bcast(np.array(todump['invmetric']), root=0)
Tint = comm.bcast(todump['Tintegration'], root=0)
Nleapfrogfid = comm.bcast(todump['Nleapfrog'], root=0)

print(stepsizefid)
print(Nleapfrogfid)
#print(invmetricfid)
print(Tint)
comm.Barrier()    


#############################
###Do HMC


#Initistate
nchains = 1

tmp = stansamples.extract().copy()
ii = np.random.randint(0, stansamples.extract()['lp__'].size)
for ik, key in enumerate(tmp.keys()):
    tmp[key] = tmp[key][ii]
tmp.pop('lp__')
initstate = stansamples.unconstrain_pars(tmp).reshape([nchains, ndim])
print("initstate in rank ", rank, initstate)


niter = 50000
start = time.time()
samples = model.sampling(data=data, chains=1, warmup=100, algorithm="HMC",  #init=list(initstate[0]),
                                      iter=niter, seed=rank, n_jobs=1,
                                      control={"inv_metric":invmetricfid,
                                               "stepsize_jitter":0,
                                               "stepsize":stepsizefid,
                                               "adapt_engaged" : False,
                                               "adapt_gamma" : 0,
                                               "int_time" : Tint
                                                })
samplesy = samples.extract(permuted=False)[..., :-1]
#samplesy = []
#for key in samples.extract().keys() : 
#    y = samples.extract()[key]
#    print(key, y.shape)
#    if len(y.shape) == 1: y = np.expand_dims(y, 1)
#    samplesy.append(y)
#samplesy = np.concatenate(samplesy[:-1], axis=1)    
#samplesy = np.expand_dims(samplesy, 1)
#ndim = samplesy.shape[-1]
np.save(fpath + 'samples/%d'%rank, samplesy)

np.save(fpath2 + 'stepsize%02d'%rank, np.array([p['stepsize__'] for p in samples.get_sampler_params()][0]))
#np.save(fpath2 + 'invmetric%02d'%rank, samples.get_inv_metric()[0])
#np.save(fpath2 + 'nleaprfrog%02d'%rank,  np.array([p['n_leapfrog__'] for p in samples.get_sampler_params()][0]))


