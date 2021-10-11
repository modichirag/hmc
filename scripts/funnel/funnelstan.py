import numpy as np
import pystan
import time
import sys, os
import diagnostics as dg
from mpi4py import MPI
import json, pickle
import utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=2, type=int, help='Dimensions')
parser.add_argument('--nsamples',  default=10000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
parser.add_argument('--lpath',  default=5, type=float, help='Nleapfrog*step_size')
parser.add_argument('--step_size', default=0.1, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--adelta', default=0.8, type=float,
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



def save_model(obj, filename):
    """Save compiled models for reuse."""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(filename):
    """Reload compiled models for reuse."""
    import pickle
    return pickle.load(open(filename, 'rb'))



model_code_funnel = """
parameters {
  real v; 
  vector[%s] theta;
}


model {
v ~ normal(0, 3);
theta ~ normal(0, exp(v/2));
}
"""%(ndim-1)


fname = './models/funnel_%sdim.pkl'%ndim
print(fname)

start = time.time()
try:
    sm_funnel = load_model(fname)
    print("Model loaded from %s"%fname, sm_funnel)
except Exception as e:
    print(e)
    sm_funnel = pystan.StanModel(model_code=model_code_funnel)
    save_model(sm_funnel, fname)
    print("model saved in %s"%fname)
    
print("Time to make model : ", time.time()-start)


#start = time.time()
#samples = sm_funnel.sampling(iter=2*nsamples, chains=1, algorithm="HMC", seed=100, n_jobs=1,
#                     control={"stepsize":0.1, 
#                                "adapt_t0":False,
#                                "adapt_delta":False,
#                                "adapt_kappa":False,
#                                "metric":"unit_e",
#                              "int_time":5
#                                })
#end = time.time()
#ss = samples.extract()
#ss = np.hstack([ss['v'].reshape(-1, 1), ss['theta']])
##ss = np.array([ss[i] for i in ss.keys()])
#print(ss)
#print("nsamples : ", len(ss[0]))
#print("One chain, 1 job : ", end - start)
#
##start = time.time()
##samples = sm_funnel.sampling(iter=1*nsamples+burnin, chains=nchains, algorithm="HMC", seed=rank, n_jobs=1, init=1, warmup=burnin,
##                     control={"stepsize":step_size, 
##                                "adapt_t0":False,
##                                "adapt_delta":False,
##                                "adapt_kappa":False,
##                                "metric":"unit_e",
##                              "int_time":Lpath
##                                })
##end = time.time()
##ss = samples.extract()
##try: mysamples = np.hstack([ss['v'].reshape(-1, 1), ss['theta']])
##except: mysamples = np.array([ss['v'], ss['theta']]).T
##print(mysamples.shape)
##mysamples = np.expand_dims(mysamples, 1)
##
##fpath = './outputs_stan/Ndim%02d/'%ndim
##try: os.makedirs(fpath)
##except Exception as e: print(e)
##fpath = fpath + 'step%03d_nleap%02d/'%(step_size*100, Nleapfrog)
##try: os.makedirs(fpath)
##except Exception as e: print(e)
##print("output in : ", fpath)
##np.save(fpath + '/samples', mysamples)
###
##dg.plot_hist(mysamples, fpath)
##dg.plot_trace(mysamples, fpath)
##dg.plot_scatter(mysamples, fpath)
##dg.plot_autorcc(mysamples, fpath)
##print("4 chain, 1 job : ", end - start)
##

#fdir= '/mnt/ceph/users/cmodi/hmc/outputs_unit/%s//'%posname
fdir= '/mnt/ceph/users/cmodi/hmc/outputs_long/funnel/Ndim%02d/'%ndim
fpath = fdir + 'default_adelta%02d/'%(args.adelta*100)
try: os.makedirs(fpath)
except Exception as e: print(e)
try: os.makedirs(fpath + '/samples/')
except Exception as e: print(e)
fpath2 = fpath + 'metrics/'
try: os.makedirs(fpath2)
except Exception as e: print(e)


niter = args.nsamples
stansamples = sm_funnel.sampling(chains=1, warmup=niter, 
                              iter=2*niter, seed=rank, n_jobs=1,
                              control={"metric":"diag_e",
                                        "stepsize_jitter":0,
                                       "adapt_delta":args.adelta
                                         })

print(rank, stansamples.get_adaptation_info())
print(stansamples.get_inv_metric())

#saves
print('saving metrics')
np.save(fpath2 + 'stepsize%02d'%rank, np.array([p['stepsize__'] for p in stansamples.get_sampler_params()][0]))
np.save(fpath2 + 'invmetric%02d'%rank, stansamples.get_inv_metric()[0])
np.save(fpath2 + 'nleaprfrog%02d'%rank,  np.array([p['n_leapfrog__'] for p in stansamples.get_sampler_params()][0]))


stepsizes0 = comm.gather(stansamples.get_stepsize()[0], root=0)
try: invmetrics = comm.gather(stansamples.get_inv_metric()[0], root=0)
except: invmetrics  = [np.ones(ndim)]*size
nleapfrogs = comm.gather([p['n_leapfrog__'] for p in stansamples.get_sampler_params()][0], root=0)
stepsizes = comm.gather([p['stepsize__'] for p in stansamples.get_sampler_params()][0], root=0)
divs = comm.gather([p['divergent__'] for p in stansamples.get_sampler_params()][0], root=0)
samplesy = stansamples.extract(permuted=False)[..., :-1]
np.save(fpath + 'samples/%d'%rank, samplesy)

#print(rank, nleapfrogs)

if rank ==0:
    stepsizefid = np.array(stepsizes0).mean()
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

#except:
#    pass
