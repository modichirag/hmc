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


from pyhmc import PyHMC_multistep, PyHMC_multistep_tries3, PyHMC
import diagnostics as dg
import pystan

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nsamples',  default=10000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
parser.add_argument('--nchains',  default=1, type=int, help='Number of chains')
parser.add_argument('--nparallel',  default=1, type=int, help='Number of parallel iterations for map')
parser.add_argument('--posnumber',  default=0, type=int, help='Posterior Number')
parser.add_argument('--posname',  default='', type=str, help='Posterior Name')
parser.add_argument('--olong',  default=1, type=int, help='prob or no prob')
parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
nsamples = args.nsamples
suffix = args.suffix
if suffix != '': suffix = '-'+suffix
burnin = args.burnin

#

from posteriordb import PosteriorDatabase
import os
pdb_path = os.path.join('../../posteriordb/posterior_database/')
my_pdb = PosteriorDatabase(pdb_path)

pos = my_pdb.posterior_names()
if args.posname == '':
    posname = pos[args.posnumber]
else: posname = args.posname
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


##Compile model
fname = './models/%s.pkl'%posname
try:
    with open(fname, 'rb') as f:
        smodel = pickle.load(f)
except Exception as e:
    print(e)
    print("Model not found")
    smodel = pystan.StanModel(model_code=model.code('stan'))
    with open(fname, 'wb') as f:
        pickle.dump(smodel, f, protocol=pickle.HIGHEST_PROTOCOL)

if args.olong == 1:  fdir = '/mnt/ceph/users/cmodi/hmc/outputs_long/%s%s//'%(posname, suffix)
if args.olong == 3:  fdir = '/mnt/ceph/users/cmodi/hmc/outputs_long3/%s%s//'%(posname, suffix)
try: os.makedirs(fdir)
except Exception as e: print(e)


#############################
###setup parameters from Stan

try:
    with open(fdir + 'params.json', 'r') as fp:
        todump = json.load(fp)
    stansamples = smodel.sampling(data=data.values(), chains=1, iter=1000, n_jobs=1, seed=100, verbose=False)

except Exception as e:
    print(e)
    if rank == 0: niter = 100000
    else: niter = 100
    stansamples, todump = utils.getstanparams(smodel, posterior, niter=niter) 
    if rank == 0:
        with open(fdir + 'params.json', 'w') as fp: 
            json.dump(todump, fp, sort_keys=True, indent=4)


comm.Barrier()    
stepsizefid = comm.bcast(todump['stepsize'], root=0)
invmetricfid = comm.bcast(np.array(todump['invmetric']), root=0)
Tint = comm.bcast(todump['Tintegration'], root=0)
Nleapfrogfid = comm.bcast(todump['Nleapfrog'], root=0)

#
comm.Barrier()


#############################
###Do HMC

def samplesaves(fpath, mysamples, accepted, probs, counts):

    try:
        mysamples = comm.gather(mysamples, root=0)
        accepted = comm.gather(accepted, root=0)
        probs = comm.gather(probs, root=0)
        counts = comm.gather(counts, root=0)
    except Exception as e:
        print(rank, e)
        comm.Abort(1)    #    sys.exit(-1)
    
    if rank == 0:
        try:  os.makedirs(fpath)
        except Exception as e: print(e)
        mysamples = np.concatenate(mysamples, axis=1)
        accepted = np.concatenate(accepted, axis=1)
        probs = np.concatenate(probs, axis=1)
        counts  = np.concatenate(counts, axis=1)
 
        np.save(fpath + '/samples', mysamples)
        np.save(fpath + '/accepted', accepted)
        np.save(fpath + '/probs', probs)
        np.save(fpath + '/counts', counts)
    
        print('Saved in %s'%fpath)
        start = time.time()
        dg.plot_hist(mysamples[::10], fpath)
        #print(time.time() - start)
    #   

#############################
###Do HMC

nchains = 1
log_prob = stansamples.log_prob
grad_log_prob = stansamples.grad_log_prob

hmc = PyHMC(log_prob, grad_log_prob, invmetricfid)
if args.olong == 1: hmc_multi = PyHMC_multistep(log_prob, grad_log_prob, invmetricfid)
if args.olong == 3: hmc_multi = PyHMC_multistep_tries3(log_prob, grad_log_prob, invmetricfid)
#hmc_multi = PyHMC_multistep(log_prob, grad_log_prob, invmetricfid)
#initstate = np.random.uniform(-1., 1., size=nchains*ndim).reshape([nchains, ndim])

samplesy = stansamples.extract(permuted=False)[..., ]
os.makedirs(fdir + 'step10/initsamples', exist_ok=True)
np.save(fdir + 'step10/initsamples/%02d'%rank, samplesy)
tmp = stansamples.extract().copy()
ii = np.random.randint(0, stansamples.extract()['lp__'].size)
counter = 0 
print("samples shape : ", samplesy.shape)
for ik, key in enumerate(tmp.keys()):
    size = tmp[key][ii].size
    print(key, size)
    #tmp[key] = tmp[key][ii]
    if size > 1: 
        tmp[key] = samplesy[-args.burnin:, 0, counter: counter + size].mean(axis=0)
    else: 
        tmp[key] = samplesy[-args.burnin:, 0, counter].mean(axis=0)
    counter  = counter + size
tmp.pop('lp__')
initstate = stansamples.unconstrain_pars(tmp).reshape([nchains, ndim])
print("initstate in rank ", rank, initstate)


for ss in [1, 2, 5, 0.5]:

    step_size, Nleapfrog =  stepsizefid*ss, max(5, int(Nleapfrogfid/ss))
    stepfunc =  lambda x:  hmc.hmc_step(x, Nleapfrog, step_size)

    print("\nFor step size %0.3f and %d leapfrog steps\n"%(step_size, Nleapfrog))

    mysamplesu, accepted, probs, counts = utils.do_hmc(stepfunc, initstate, nsamples=args.nsamples)
    mysamples = np.array([[stansamples.constrain_pars(mysamplesu[i, j]) \
                           for i in range(mysamplesu.shape[0])] for j in range(mysamplesu.shape[1])])
    mysamples = mysamples.transpose(1, 0, 2)
    fpath = fdir + 'step%02d/'%(ss*10)
    samplesaves(fpath, mysamples, accepted, probs, counts)

    for nsub in [2, 3, 4]:
        for two_factor in [2, 5, 10]:
            if two_factor**(nsub-1) > 200: continue
            print("\nFor step size %0.3f and %d leapfrog steps\n"%(step_size, Nleapfrog))
            print("\nSubsize with %d upto %d times\n"%(two_factor, nsub))
            stepfunc_multi =   lambda x: hmc_multi.multi_step(nsub, x, Nleapfrog, step_size, two_factor)
            mysamplesu, accepted, probs, counts = utils.do_hmc(stepfunc_multi, initstate, nsamples=args.nsamples)
            mysamples = np.array([[stansamples.constrain_pars(mysamplesu[i, j]) \
                                   for i in range(mysamplesu.shape[0])] for j in range(mysamplesu.shape[1])])
            mysamples = mysamples.transpose(1, 0, 2)
            fpath = fdir + 'step%02d_fac%02d_nsub%d/'%(ss*10, two_factor, nsub)
            samplesaves(fpath, mysamples, accepted, probs, counts)
            
