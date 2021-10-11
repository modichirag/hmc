import numpy as np
#Splt.import matplotlib.pyplot as plt
import time
import sys, os
from mpi4py import MPI
import json, pickle
import utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pystan
import argparse
from setupstan import setup

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nsamples',  default=10000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')
parser.add_argument('--tnuts', default=0, type=int,
                    help='use nuts for tint')
parser.add_argument('--gather', default=1, type=int,
                    help='gather samples from different ranks')
parser.add_argument('--nssq',  default=0.9, type=float, help='Quantile of leapfrog step')
parser.add_argument('--adelta',  default=0.8, type=float, help='Adapt delta for Stan')

args = parser.parse_args()
nsamples = args.nsamples
suffix = args.suffix
if suffix != '': suffix = '-'+suffix


def save_model(obj, filename):
    """Save compiled models for reuse."""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    """Reload compiled models for reuse."""
    import pickle
    return pickle.load(open(filename, 'rb'))


##http://arxiv.org/abs/2003.03950
model_code="""
parameters {
  real y;
}
model {
  target += log_sum_exp(log(0.5) + normal_lpdf(y |0, 0.1),
                        log(0.5) + normal_lpdf(y |3, 1));
}
"""


modelname = 'gaussmixs0p1s1'
fname = './models/%s.pkl'%modelname
fpath = '/mnt/ceph/users/cmodi/hmc/outputs_long/%s%s/'%(modelname, suffix)
##os.makedirs(fpath, exist_ok=True)
##fpathparams = fpath
##fpath = fpath + 'default/'
##os.makedirs(fpath, exist_ok=True)
##os.makedirs(fpath + '/samples/', exist_ok=True)
##fpath2 = fpath + 'metrics/'
##os.makedirs(fpath2, exist_ok=True)
##print("output in : ", fpath)
##

start = time.time()
try:
    model = load_model(fname)
    print("Model loaded from %s"%fname, model)
except Exception as e:
    print(e)
    model = pystan.StanModel(model_code=model_code)
    save_model(model, fname)
    print("model saved in %s"%fname)
    
print("Time to make model : ", time.time()-start)


data = None

setup(model, data, args, fpath)

##
##np.random.seed(1234)
##
##niter = 10000
##samples = model.sampling(chains=1, warmup=niter, 
##                                      iter=niter*2, n_jobs=1,
##                                      control={"metric":"diag_e",
##                                                 })
##
##
###saves
##np.save(fpath2 + 'stepsize%02d'%rank, np.array([p['stepsize__'] for p in samples.get_sampler_params()][0]))
##np.save(fpath2 + 'invmetric%02d'%rank, samples.get_inv_metric()[0])
##np.save(fpath2 + 'nleaprfrog%02d'%rank,  np.array([p['n_leapfrog__'] for p in samples.get_sampler_params()][0]))
##
##
##stepsizes0 = comm.gather(samples.get_stepsize()[0], root=0)
##try: invmetrics = comm.gather(samples.get_inv_metric()[0], root=0)
##except: invmetrics  = [np.ones(ndim)]*size
##nleapfrogs = comm.gather([p['n_leapfrog__'][niter:] for p in samples.get_sampler_params()][0], root=0)
##stepsizes = comm.gather([p['stepsize__'][niter:] for p in samples.get_sampler_params()][0], root=0)
##divs = comm.gather([p['divergent__'][niter:] for p in samples.get_sampler_params()][0], root=0)
##
##samplesy = samples.extract(permuted=False)[..., :-1]
##ndim = samplesy.shape[-1]
##np.save(fpath + 'samples/%d'%rank, samplesy)
##
##refsamples = comm.gather(samplesy, root=0)
##Tint = None
##if rank ==0:
##   
##    print('In loop for rank 0') 
##    refsamples = np.concatenate(refsamples, axis=1)
##    np.save(fpath + 'samples', refsamples)
##    print("samples shape : ", refsamples.shape)                              
##
##    stepsizefid = np.array(stepsizes0).mean()
##    np.save(fpath2 + 'stepsizefid', np.array(stepsizes0))
##    invmetricfid = np.array(invmetrics).mean(axis=0)
##
##    nleapfrogs = np.concatenate(nleapfrogs)
##    stepsizes = np.concatenate(stepsizes)
##    divs = np.concatenate(divs)
##
##    #nss = nleapfrogs[np.where((abs(stepsizes - stepsizefid) < stepsizefid/10.) & (divs==0))]
##    nss = nleapfrogs.copy() #[np.where((abs(stepsizes - stepsizefid) < stepsizefid/10.) & (divs==0))]
##    Tint = stepsizefid*np.quantile(nss, 0.9)
##    Nleapfrogfid = int(Tint/stepsizefid)
##    Tint = comm.bcast(Tint, root=0)
##    
##
##    todump = {}
##    todump['stepsize'] = stepsizefid
##    todump['invmetric'] = list(invmetricfid)
##    todump['Tintegration'] = Tint
##    todump['Nleapfrog'] = Nleapfrogfid
##    todump['Ndim'] = ndim
##
##    print(todump)
##
##    with open(fpathparams + 'nutsparams.json', 'w') as fp: 
##        json.dump(todump, fp, sort_keys=True, indent=4)
##
##comm.Barrier()
########################################################
###HMC Adaptations
##samples = model.sampling(data=None, chains=1, warmup=niter, algorithm='HMC',
##                         iter=niter*2, n_jobs=4,
##                         control={"metric":"diag_e", 
##                                  })#"int_time":Tint})
##
##np.save(fpath2 + 'stepsizehmc%02d'%rank, np.array([p['stepsize__'] for p in samples.get_sampler_params()][0]))
##np.save(fpath2 + 'invmetrichmc%02d'%rank, samples.get_inv_metric()[0])
##np.save(fpath2 + 'nleaprfroghmc%02d'%rank,  np.array([p['int_time__']/p['stepsize__'] for p in samples.get_sampler_params()][0]))
##
##
##stepsizes0 = comm.gather(samples.get_stepsize()[0], root=0)
##try: invmetrics = comm.gather(samples.get_inv_metric()[0], root=0)
##except: invmetrics  = [np.ones(ndim)]*size
##inttime = comm.gather([p['int_time__'][niter:] for p in samples.get_sampler_params()][0], root=0)
##stepsizes = comm.gather([p['stepsize__'][niter:] for p in samples.get_sampler_params()][0], root=0)
##
##samplesy = samples.extract(permuted=False)[..., :-1]
##ndim = samplesy.shape[-1]
##np.save(fpath + 'samples/hmc%d'%rank, samplesy)
##
##refsamples = comm.gather(samplesy, root=0)
##if rank ==0:
##    print('In loop for rank 0') 
##    refsamples = np.concatenate(refsamples, axis=1)
##    np.save(fpath + 'sampleshmc', refsamples)
##    print("samples shape : ", refsamples.shape)
##    
##
##    stepsizefid = np.array(stepsizes0).mean()
##    np.save(fpath2 + 'stepsizefidhmc', np.array(stepsizes0))
##    invmetricfid = np.array(invmetrics).mean(axis=0)
##
##    inttime = np.concatenate(inttime)
##    stepsizes = np.concatenate(stepsizes)
##
##    Tint = inttime.mean()
##    Nleapfrogfid = int(Tint/stepsizefid)
##
##
##    todump = {}
##    todump['stepsize'] = stepsizefid
##    todump['invmetric'] = list(invmetricfid)
##    todump['Tintegration'] = Tint
##    todump['Nleapfrog'] = Nleapfrogfid
##    todump['Ndim'] = ndim
##
##    print(todump)
##
##    with open(fpathparams + 'params.json', 'w') as fp: 
##        json.dump(todump, fp, sort_keys=True, indent=4)
##
