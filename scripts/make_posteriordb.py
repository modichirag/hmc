##Script to setup compiled posteriordb model as well optimal HMC configuration

import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
from mpi4py import MPI
import json, pickle
import utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


from pyhmc import PyHMC_multistep, PyHMC
import diagnostics as dg
import pystan

import argparse
from setupstan import setup


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-posnumber',   type=int, help='Posterior Number')
parser.add_argument('--nsamples',  default=10000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
parser.add_argument('--suffix', default='hmctnuts', type=str,
                    help='sum the integers (default: find the max)')
parser.add_argument('--tnuts', default=1, type=int,
                    help='use nuts for tint')
parser.add_argument('--gather', default=1, type=int,
                    help='gather samples from different ranks')
parser.add_argument('--nssq',  default=0.9, type=float, help='Quantile of leapfrog step')
parser.add_argument('--adelta',  default=0.8, type=float, help='Adapt delta for Stan')
parser.add_argument('--ndata', default=2, type=int,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
nsamples = args.nsamples
suffix = args.suffix
if suffix != '': suffix = '-'+suffix



#
from posteriordb import PosteriorDatabase
import os
pdb_path = os.path.join('../../posteriordb/posterior_database/')
my_pdb = PosteriorDatabase(pdb_path)

pos = my_pdb.posterior_names()
posnumber = args.posnumber
posname = pos[posnumber]

##fname = './models/%s.pkl'%posname
posterior = my_pdb.posterior(posname)
model, data = posterior.model, posterior.data
print(model.code('stan'))
##smodel = pystan.StanModel(model_code=model.code('stan'))
##print("smodel : ", smodel)
##with open(fname, 'wb') as f:
##    pickle.dump(smodel, f, protocol=pickle.HIGHEST_PROTOCOL)
##print("Save model in %s"%fname)
##
##
try:
    posnumber = args.posnumber
    posname = pos[posnumber]
    #posname = "eight_schools-eight_schools_centered"
    print('Name of the posterior is ', posname)

    posterior = my_pdb.posterior(posname)
    model, data = posterior.model, posterior.data
    refinfo = posterior.reference_draws_info()
    refdrawsdict = posterior.reference_draws()
    refdraws = []
    for i in range(len(refdrawsdict)):
        refdraws.append(np.array([refdrawsdict[i][key] for key in refdrawsdict[i].keys()]).T)
    refdraws = np.stack(refdraws, 1)
    refdraws2d = refdraws.reshape(-1, refdraws.shape[-1])
    ndim = refdraws.shape[-1]
    print('ndim  : ', ndim)

except: pass

##Compile model
fname = './models/%s.pkl'%posname
try:
    with open(fname, 'rb') as f:
        smodel = pickle.load(f)
except Exception as e:
    print(e)
    print("Model not found")
    try:
        smodel = pystan.StanModel(model_code=model.code('stan'))
        print("smodel : ", smodel)
        with open(fname, 'wb') as f:
            pickle.dump(smodel, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Save model in %s"%fname)
    except: pass

#fdir= '/mnt/ceph/users/cmodi/hmc/outputs_unit/%s//'%posname
modelname = posname
fpath = '/mnt/ceph/users/cmodi/hmc/outputs_long/%s%s/'%(modelname, suffix)
#fdir= '/mnt/ceph/users/cmodi/hmc/outputs_long/%s-def//'%posname
os.makedirs(fpath, exist_ok=True)
##except Exception as e: print(e)
##fpath = fdir + 'default/'
##try: os.makedirs(fpath)
##except Exception as e: print(e)
##try: os.makedirs(fpath + '/samples2/')
##except Exception as e: print(e)
##fpath2 = fpath + 'metrics/'
##try: os.makedirs(fpath2)
##except Exception as e: print(e)
##

#############################
###setup parameters from Stan

try:
    with open(fpath + 'params.json', 'r') as fp:
        todump = json.load(fp)
    #stansamples = smodel.sampling(data=data.values(), chains=1, iter=100, n_jobs=1, seed=100, verbose=False)
    raise Exception("trying again to save metrics")
except Exception as e:
    #print(e)
    #try:
        niter = 10000
        ndim = sum([posterior.posterior_info['dimensions'][key] for key in posterior.posterior_info['dimensions'].keys()])
        print('ndim  : ', ndim)
        #model, data = posterior.model, posterior.data
        print(model.code)

        model, data = smodel, posterior.data.values()
        setup(model, data, args, fpath)

##        stansamples = smodel.sampling(data=data.values(), chains=1, warmup=niter, 
##                                      iter=2*niter, seed=rank, n_jobs=1,
##                                      control={"metric":"diag_e",
##                                                "stepsize_jitter":0
##                                                 })
##
##        print(rank, stansamples.get_adaptation_info())
##        print(stansamples.get_inv_metric())
##
##        #saves
##        print('saving metrics')
##        np.save(fpath2 + 'stepsize%02d'%rank, np.array([p['stepsize__'] for p in stansamples.get_sampler_params()][0]))
##        np.save(fpath2 + 'invmetric%02d'%rank, stansamples.get_inv_metric()[0])
##        np.save(fpath2 + 'nleaprfrog%02d'%rank,  np.array([p['n_leapfrog__'] for p in stansamples.get_sampler_params()][0]))
##
##
##        stepsizes0 = comm.gather(stansamples.get_stepsize()[0], root=0)
##        try: invmetrics = comm.gather(stansamples.get_inv_metric()[0], root=0)
##        except: invmetrics  = [np.ones(ndim)]*size
##        nleapfrogs = comm.gather([p['n_leapfrog__'] for p in stansamples.get_sampler_params()][0], root=0)
##        stepsizes = comm.gather([p['stepsize__'] for p in stansamples.get_sampler_params()][0], root=0)
##        divs = comm.gather([p['divergent__'] for p in stansamples.get_sampler_params()][0], root=0)
##        samplesy = stansamples.extract(permuted=False)[..., :-1]
##        np.save(fpath + 'samples2/%d'%rank, samplesy)
##
##        #print(rank, nleapfrogs)
##
##        if rank ==0:
##            stepsizefid = np.array(stepsizes0).mean()
##            invmetricfid = np.array(invmetrics).mean(axis=0)
##
##            nleapfrogs = np.concatenate(nleapfrogs)
##            stepsizes = np.concatenate(stepsizes)
##            divs = np.concatenate(divs)
##
##            nss = nleapfrogs[np.where((abs(stepsizes - stepsizefid) < stepsizefid/10.) & (divs==0))]
##            Tint = stepsizefid*np.quantile(nss, 0.9)
##            Nleapfrogfid = int(Tint/stepsizefid)
##
##            todump = {}
##            todump['stepsize'] = stepsizefid
##            todump['invmetric'] = list(invmetricfid)
##            todump['Tintegration'] = Tint
##            todump['Nleapfrog'] = Nleapfrogfid
##            todump['Ndim'] = ndim
##
##            print(todump)
##
##            with open(fdir + 'params2.json', 'w') as fp: 
##                json.dump(todump, fp, sort_keys=True, indent=4)
##
##    #except:
##    #    pass
