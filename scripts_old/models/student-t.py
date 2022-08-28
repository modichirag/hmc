import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
import json
import pystan

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


from pyhmc import PyHMC
import diagnostics as dg

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=2, type=int, help='Dimensions')
parser.add_argument('--nsamples',  default=10000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
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

##
fpath= '/mnt/ceph/users/cmodi/hmc/outputs_long/student_t//'
try: os.makedirs(fpath)
except Exception as e: print(e)
fpath= '/mnt/ceph/users/cmodi/hmc/outputs_long/student_t/N%d/'%ndim
try: os.makedirs(fpath)
except Exception as e: print(e)
fpath = fpath + 'default/'
try: os.makedirs(fpath)
except Exception as e: print(e)
try: os.makedirs(fpath + '/samples/')
except Exception as e: print(e)
fpath2 = fpath + 'metrics/'
try: os.makedirs(fpath2)
except Exception as e: print(e)
print("output in : ", fpath)


#######


def save_model(obj, filename):
    """Save compiled models for reuse."""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    """Reload compiled models for reuse."""
    import pickle
    return pickle.load(open(filename, 'rb'))


model_code = """
data {
  int<lower = 0> N;
  real<lower = 1> nu;
}
parameters {
  vector[N] alpha;
}
model {
  alpha ~ student_t(nu, 0, 1);
}
generated quantities {
  //vector<lower = 0>[N]  alpha_sq = alpha^2;
}
"""

fname = './models/student_t_N%d.pkl'%ndim
data = {"N":ndim, "nu":2}

start = time.time()
try:
    model = load_model(fname)
    print("Model loaded from %s"%fname, model)
except Exception as e:
    print(e)
    model = pystan.StanModel(model_code=model_code)
    save_model(model, fname)
    print("model save5Cd in %s"%fname)
    
print("Time to make model : ", time.time()-start)

sys.exit()

niter = 50000
start = time.time()
samples = model.sampling(data=data, chains=1, warmup=niter, 
                                      iter=niter*2, seed=rank, n_jobs=1,
                                      control={"metric":"diag_e",
                                                "stepsize_jitter":0 })

#saves
np.save(fpath2 + 'stepsize%02d'%rank, np.array([p['stepsize__'] for p in samples.get_sampler_params()][0]))
np.save(fpath2 + 'invmetric%02d'%rank, samples.get_inv_metric()[0])
np.save(fpath2 + 'nleaprfrog%02d'%rank,  np.array([p['n_leapfrog__'] for p in samples.get_sampler_params()][0]))


stepsizes0 = comm.gather(samples.get_stepsize()[0], root=0)
try: invmetrics = comm.gather(samples.get_inv_metric()[0], root=0)
except: invmetrics  = [np.ones(ndim)]*size
nleapfrogs = comm.gather([p['n_leapfrog__'] for p in samples.get_sampler_params()][0], root=0)
stepsizes = comm.gather([p['stepsize__'] for p in samples.get_sampler_params()][0], root=0)
divs = comm.gather([p['divergent__'] for p in samples.get_sampler_params()][0], root=0)

samplesy = samples.extract(permuted=False)[..., :-1]
#samplesy = []
#for key in samples.extract().keys() : 
#    y = samples.extract()[key]
#    print(key, y.shape)
#    if len(y.shape) == 1: y = np.expand_dims(y, 1)
#    samplesy.append(y)
#samplesy = np.concatenate(samplesy[:-1], axis=1)    
#samplesy = np.expand_dims(samplesy, 1)
ndim = samplesy.shape[-1]
np.save(fpath + 'samples/%d'%rank, samplesy)

refsamples = comm.gather(samplesy, root=0)


if rank ==0:
    print('In loop for rank 0') 

    stepsizefid = np.array(stepsizes0).mean()
    np.save(fpath2 + 'stepsizefid', np.array(stepsizes0))
    invmetricfid = np.array(invmetrics).mean(axis=0)

    nleapfrogs = np.concatenate(nleapfrogs)
    stepsizes = np.concatenate(stepsizes)
    divs = np.concatenate(divs)

    nss = nleapfrogs[np.where((abs(stepsizes - stepsizefid) < stepsizefid/10.) & (divs==0))]
    Tint = stepsizefid*np.quantile(nss, 0.9)
    Nleapfrogfid = int(Tint/stepsizefid)

    todump = {}
    todump['stepsize'] = stepsizefid
    todump['invmetric'] = list(invmetricfid)
    todump['Tintegration'] = Tint
    todump['Nleapfrog'] = Nleapfrogfid
    todump['Ndim'] = ndim

    print(todump)

    with open(fpath + 'params.json', 'w') as fp: 
        json.dump(todump, fp, sort_keys=True, indent=4)
    print(len(refsamples))

    refsamples = np.concatenate(refsamples, axis=1)
    np.save(fpath + 'samples', refsamples)
    print("samples shape : ", refsamples.shape)
    
comm.Barrier()
