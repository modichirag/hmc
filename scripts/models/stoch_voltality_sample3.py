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


from pyhmc import PyHMC_multistep_tries3, PyHMC
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
fpath= '/mnt/ceph/users/cmodi/hmc/outputs_long3/stoch_voltality//'
try: os.makedirs(fpath)
except Exception as e: print(e)
fpathd =  '/mnt/ceph/users/cmodi/hmc/outputs_long/stoch_voltality-def/default/'


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


comm.Barrier()    

#############################
###Do HMC

def samplesaves(fpathd, mysamples, accepted, probs, counts):

    for ff in [fpathd, fpathd + '/samples/', fpathd + '/accepted/', fpathd + '/probs/', fpathd + '/counts/']:
        try:  
            os.makedirs(ff)
        except Exception as e: print(e)

    np.save(fpathd + '/samples/%d'%rank, mysamples)
    np.save(fpathd + '/accepted/%d'%rank, accepted)
    np.save(fpathd + '/probs/%d'%rank, probs)
    np.save(fpathd + '/counts/%d'%rank, counts)
    #try:
    #    np.save(fpathd + '/samples/%d'%rank, mysamples)
    #    np.save(fpathd + '/accepted/%d'%rank, accepted)
    #    np.save(fpathd + '/probs/%d'%rank, probs)
    #    np.save(fpathd + '/counts/%d'%rank, counts)
    #    mysamples = comm.gather(mysamples, root=0)
    #    accepted = comm.gather(accepted, root=0)
    #    probs = comm.gather(probs, root=0)
    #    counts = comm.gather(counts, root=0)
    #except Exception as e:
    #    print(rank, e)
    #    comm.Abort(1)    #    sys.exit(-1)
    #
    #if rank == 0:
    #    #mysamples = np.concatenate(mysamples, axis=1)
    #    accepted = np.concatenate(accepted, axis=1)
    #    probs = np.concatenate(probs, axis=1)
    #    counts  = np.concatenate(counts, axis=1)
    #
    #    #np.save(fpathd + '/samples', mysamples)
    #    np.save(fpathd + '/accepted', accepted)
    #    np.save(fpathd + '/probs', probs)
    #    np.save(fpathd + '/counts', counts)
    #
    #    print('Saved in %s'%fpathd)
    #    start = time.time()
    #    #dg.plot_hist(mysamples[::10], fpathd)
    #    #print(time.time() - start)
    #   


###LOG PROB
log_prob = stansamples.log_prob
grad_log_prob = stansamples.grad_log_prob
nchains = 1



#############################
###Do HMC

nchains = 1
log_prob = stansamples.log_prob
grad_log_prob = stansamples.grad_log_prob

hmc = PyHMC(log_prob, grad_log_prob, invmetricfid)
hmc_multi = PyHMC_multistep_tries3(log_prob, grad_log_prob, invmetricfid)

#Initistate
tmp = stansamples.extract().copy()
ii = np.random.randint(0, stansamples.extract()['lp__'].size)
for ik, key in enumerate(tmp.keys()):
    tmp[key] = tmp[key][ii]
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
    fpathd = fpath + 'step%02d/'%(ss*10)
    samplesaves(fpathd, mysamples, accepted, probs, counts)

    for nsub in [2, 3, 4, 5]:
        for two_factor in [2, 5, 10]:
            if two_factor**(nsub-1) > 200: continue
            print("\nFor step size %0.3f and %d leapfrog steps\n"%(step_size, Nleapfrog))
            print("\nSubsize with %d upto %d times\n"%(two_factor, nsub))
            try:
                stepfunc_multi =   lambda x: hmc_multi.multi_step(nsub, x, Nleapfrog, step_size, two_factor)
                mysamplesu, accepted, probs, counts = utils.do_hmc(stepfunc_multi, initstate, nsamples=args.nsamples)
                mysamples = np.array([[stansamples.constrain_pars(mysamplesu[i, j]) \
                                       for i in range(mysamplesu.shape[0])] for j in range(mysamplesu.shape[1])])
                mysamples = mysamples.transpose(1, 0, 2)
                fpathd = fpath + 'step%02d_fac%02d_nsub%d/'%(ss*10, two_factor, nsub)
                samplesaves(fpathd, mysamples, accepted, probs, counts)
            except: pass
