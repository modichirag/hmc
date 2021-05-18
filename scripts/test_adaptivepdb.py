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


from adaptivehmc import AdHMC
from pyhmc import PyHMC,  PyHMC_multistep
import diagnostics as dg
import pystan

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nsamples',  default=10000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
parser.add_argument('--nchains',  default=1, type=int, help='Number of chains')
parser.add_argument('--nparallel',  default=1, type=int, help='Number of parallel iterations for map')
parser.add_argument('--posnumber',  default=0, type=int, help='Posterior Number')
parser.add_argument('--nleap',  default=10, type=int, help='Posterior Number')
parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')
parser.add_argument('--vanilla', default=1, type=int,
                    help='vanilla run')

args = parser.parse_args()
nsamples = args.nsamples
suffix = args.suffix
burnin = args.burnin
Nleapfrog = args.nleap
nchains = 1
#

from posteriordb import PosteriorDatabase
import os
pdb_path = os.path.join('../../posteriordb/posterior_database/')
my_pdb = PosteriorDatabase(pdb_path)

pos = my_pdb.posterior_names()
posname = pos[args.posnumber]
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


###setup parameters from Stan
with open('/mnt/ceph/users/cmodi/hmc/outputs_long/%s-def/params.json'%posname, 'r') as fp:
    todump = json.load(fp)
stansamples = smodel.sampling(data=data.values(), chains=1, iter=100, n_jobs=1, seed=100, verbose=False)


comm.Barrier()    
stepsizefid = comm.bcast(todump['stepsize'], root=0)
invmetricfid = comm.bcast(np.array(todump['invmetric']), root=0)
Tint = comm.bcast(todump['Tintegration'], root=0)
Nleapfrogfid = comm.bcast(todump['Nleapfrog'], root=0)

#
comm.Barrier()

#################################
#
fdir= '/mnt/ceph/users/cmodi/hmc/outputs_adaptive/%s//'%posname
try: os.makedirs(fdir)
except Exception as e: print(e)


##
smin, smax = 0.01, 0.1
if args.vanilla:
    fpath = fdir + 'nleap%02d_vanilla/'%(Nleapfrog)
    ratios = [1.0, 1.0, 1.0]
else:
    fpath = fdir + 'nleap%02d/'%(Nleapfrog)
    ratios = [0.8, 1.0, 1/0.8]
try: os.makedirs(fpath)
except Exception as e: print(e)
print("output in : ", fpath)

try:
    checksamples =  np.load(fpath + '/samples.npy')
    print(checksamples.shape)
    if checksamples.shape[1] == 50: 
        if checksamples.shape[0] == 10000:
            import sys
            print("\nAlready did this run \n Exit")
            #sys.exit()
except Exception as e:
    print(e)
    print("Carry on")




        

#############################

##Setup HMC
log_prob = stansamples.log_prob
grad_log_prob = stansamples.grad_log_prob

hmc = AdHMC(log_prob, grad_log_prob, invmetricfid)
def step(x):
    if args.vanilla == 1:
        return  hmc.hmc_step_vanilla(x, Nleapfrog, smin=smin, smax=smax)
    else:
        return  hmc.hmc_step(x, Nleapfrog, smin=smin, smax=smax, ratios=ratios)



tmp = stansamples.extract().copy()
ii = np.random.randint(0, stansamples.extract()['lp__'].size)
for ik, key in enumerate(tmp.keys()):
    tmp[key] = tmp[key][ii]
tmp.pop('lp__')
initstate = stansamples.unconstrain_pars(tmp).reshape([nchains, ndim])
print("initstate in rank ", rank, initstate)



def do_hmc():
    
    samples = []
    accepts = []
    probs = []
    diags = []
    start = time.time()
    q = initstate

    for i in range(nsamples + burnin):
        #out = pool.map(step, q)
        out = list(map(step, q))
        q = [i[0] for i in out] 
        acc = [i[2] for i in out]
        prob = [i[3] for i in out] 
        diag = [i[4] for i in out] 
        samples.append(q)
        accepts.append(acc)
        probs.append(prob)
        diags.append(diag)

    end = time.time()
    print(rank, end - start)
    mysamplesu = np.array(samples)[burnin:]
    mysamples = np.array([[stansamples.constrain_pars(mysamplesu[i, j]) \
                           for i in range(mysamplesu.shape[0])] for j in range(mysamplesu.shape[1])])
    mysamples = mysamples.transpose(1, 0, 2)
    accepted = np.array(accepts)[burnin:]
    probs = np.array(probs)[burnin:]
    diags = np.array(diags)[burnin:]
    print('daigs shape : ', mysamples.shape, diags.shape)
    return mysamples, accepted, probs, diags
     
if __name__=="__main__":
    mysamples, accepted, probs, diags = do_hmc()
    print(rank, mysamples.shape)
    #print(rank, mysamples)

    #sys.exit(-1)
    mysamples = comm.gather(mysamples, root=0)
    accepted = comm.gather(accepted, root=0)
    probs = comm.gather(probs, root=0)
    diags = comm.gather(diags, root=0)

    if rank == 0:
        mysamples = np.concatenate(mysamples, axis=1)
        accepted = np.concatenate(accepted, axis=1)
        probs = np.concatenate(probs, axis=1)
        diags  = np.concatenate(diags, axis=1)
        print(mysamples.shape)

        np.save(fpath + '/samples', mysamples)
        np.save(fpath + '/accepted', accepted)
        np.save(fpath + '/probs', probs)
        np.save(fpath + '/diags', diags)
    
        print('Saved in %s'%fpath)
        start = time.time()
        dg.plot_hist(mysamples, fpath, sub=1)
        print(time.time() - start)
        start = time.time()
        #dg.plot_trace(mysamples, fpath)
        print(time.time() - start)
        start = time.time()
        dg.plot_scatter(mysamples, fpath, sub=1)
        print(time.time() - start)
        start = time.time()
        #dg.plot_autorcc(mysamples, fpath)
        print(time.time() - start)
        start = time.time()
    #   
