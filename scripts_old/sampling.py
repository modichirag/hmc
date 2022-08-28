import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
import json, pickle
import pystan

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


sys.path.append('../src/')
import utils
from pyhmc import PyHMC_multistep, PyHMC_multistep_tries3, PyHMC
import diagnostics as dg

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-modelname',   type=str, help='Name of the model')
parser.add_argument('--dataname',   type=str, help='Name of the model')
parser.add_argument('--nsamples',  default=1000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
parser.add_argument('--Tint',  default=5, type=float, help='Nleapfrog*step_size')
parser.add_argument('--step_size', default=0.01, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--two_factor', default=2, type=float,
                    help='sum the integers (default: find the max)')
#parser.add_argument('--nchains',  default=10, type=int, help='Number of chains')
parser.add_argument('--olong',  default=1, type=int, help='prob or no prob')
parser.add_argument('--gather',  default=1, type=int, help='prob or no prob')
parser.add_argument('--vanilla',  default=1, type=int, help='if 1, do vanilla HMC as well')
parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')
parser.add_argument('--nutsparams', default=1, type=int,
                    help='Use fit params')



args = parser.parse_args()
modelname = args.modelname
print("Model name : ", modelname)
#ndim = args.ndim
#step_size = args.step_size
two_factor = args.two_factor
nsamples = args.nsamples
suffix = args.suffix
if suffix != '': suffix = '-'+suffix
#burnin = args.burnin
nchains = 1 #args.nchains


##

if args.olong == 1:  fpath= '/mnt/ceph/users/cmodi/hmc/outputs_long/%s%s//'%(modelname, suffix)
if args.olong == 3:  fpath= '/mnt/ceph/users/cmodi/hmc/outputs_long3/%s%s//'%(modelname, suffix)
#try: os.makedirs(fpath)
#except Exception as e: print(e)
print('Output folder : ', fpath)


#######

#model
fname = './models/%s.pkl'%modelname
with open(fname, 'rb') as f:
    model = pickle.load(f)

#data
try:
    with open('modeldata/%s.json'%args.dataname, 'r') as fp:
        data = json.load(fp)
except: 
    try: 
        with open('modeldata/%s.json'%modelname, 'r') as fp:
            data = json.load(fp)
    except: data = None

#params
if args.nutsparams:
    print('Reading parameter file at %s'%(fpath + 'params.json'))
    with open(fpath + 'params.json', 'r') as fp:
        todump = json.load(fp)
    stepsizefid = comm.bcast(todump['stepsize'], root=0)
    invmetricfid = comm.bcast(np.array(todump['invmetric']), root=0)
    Tint = comm.bcast(todump['Tintegration'], root=0)
    Nleapfrogfid = comm.bcast(todump['Nleapfrog'], root=0)
    tnamepath = False
    print('Using fid params from parameter file at %s'%(fpath + 'params.json'))
else:
    print('Using arguments for step size and Tint')
    stepsizefid = args.step_size
    invmetricfid = 1 #np.ones(ndim)
    Tint = args.Tint
    Nleapfrogfid = Tint/stepsizefid
    tnamepath = True

#############################
##Stansampling

stansamples = model.sampling(data=data, chains=1, warmup=1, algorithm='HMC',
                                      iter=2*args.burnin, seed=rank, n_jobs=1,
                                      control={"metric":"diag_e", 
                                               "stepsize":stepsizefid,
                                               "int_time":Tint,
                                               "inv_metric":invmetricfid, 
                                               "stepsize_jitter":0
                                           })

log_prob = stansamples.log_prob
grad_log_prob = stansamples.grad_log_prob


samplesy = []
for key in stansamples.extract().keys() : 
    y = stansamples.extract()[key]
    print(key, y.shape)
    if len(y.shape) == 1: y = np.expand_dims(y, 1)
    samplesy.append(y)
samplesy = np.concatenate(samplesy[:-1], axis=1)    
samplesy = np.expand_dims(samplesy, 1)
ndim = samplesy.shape[-1]

#Initistate
samplesy = stansamples.extract(permuted=False)[..., ]
os.makedirs(fpath + 'step10/initsamples', exist_ok=True)
np.save(fpath + 'step10/initsamples/%02d'%rank, samplesy)

tmp = stansamples.extract().copy() #[args.burnin:]
ii = np.random.randint(0, stansamples.extract()['lp__'].size-args.burnin)
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
#initstate = np.random.uniform(0.1, 1., size=nchains*ndim).reshape([nchains, ndim])
#print("initstate in rank ", rank, initstate)

#sys.exit()


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
    
    if args.gather == 1: 
        try:
            mysamples = comm.gather(mysamples, root=0)
            accepted = comm.gather(accepted, root=0)
            probs = comm.gather(probs, root=0)
            counts = comm.gather(counts, root=0)
        except Exception as e:
            print(rank, e)
            comm.Abort(1)        ##sys.exit(-1)

        if rank == 0:
            mysamples = np.concatenate(mysamples, axis=1)
            accepted = np.concatenate(accepted, axis=1)
            probs = np.concatenate(probs, axis=1)
            counts  = np.concatenate(counts, axis=1)

            np.save(fpathd + '/samples', mysamples)
            np.save(fpathd + '/accepted', accepted)
            np.save(fpathd + '/probs', probs)
            np.save(fpathd + '/counts', counts)

            print('Saved in %s'%fpathd)
            start = time.time()
            #dg.plot_hist(mysamples[::10], fpathd)
            print(time.time() - start)



#############################
###Do HMC


print('Step size fid : ', stepsizefid)
#print('invemtric fid : ', invmetricfid)
print('Tint fid : ', Tint)
print('Nleapfrog fid : ', Nleapfrogfid)


hmc = PyHMC(log_prob, grad_log_prob, invmetricfid)
if args.olong == 1: hmc_multi = PyHMC_multistep(log_prob, grad_log_prob, invmetricfid)
if args.olong == 3: hmc_multi = PyHMC_multistep_tries3(log_prob, grad_log_prob, invmetricfid)


stepsf = [1, 2, 5, 0.5]
nsubs = [2, 3, 4]
factors = [2, 5,10]
costthresh = 200 

for ss in stepsf:

    step_size =  stepsizefid*ss
    Nleapfrog = max(1, int(Tint/step_size)) #+ 1
    #if Nleapfrog <= 2: 
    #    print("Only %d steps for this step size \eps=%0.3f, so skipping"%(Nleapfrog, step_size))
    #    continue


    print("\nFor step size %0.3f and %d leapfrog steps\n"%(step_size, Nleapfrog))

    if args.vanilla == 1:
        stepfunc =  lambda x:  hmc.hmc_step(x, Nleapfrog, step_size)
        mysamplesu, accepted, probs, counts = utils.do_hmc(stepfunc, initstate, nsamples=args.nsamples, burnin=args.burnin)
        mysamples = np.array([[stansamples.constrain_pars(mysamplesu[i, j]) \
                           for i in range(mysamplesu.shape[0])] for j in range(mysamplesu.shape[1])])
        mysamples = mysamples.transpose(1, 0, 2)
        if tnamepath : 
            fpathd = fpath + 'step%02d_tint%03d/'%(ss*10, Tint*10)
        else: fpathd = fpath + 'step%02d/'%(ss*10)
        samplesaves(fpathd, mysamples, accepted, probs, counts)

    for nsub in nsubs:
        for two_factor in factors:
            if two_factor**(nsub-1) > costthresh: continue
            print("\nFor step size %0.3f and %d leapfrog steps\n"%(step_size, Nleapfrog))
            print("\nSubsize with %d upto %d times\n"%(two_factor, nsub))
            try:
                stepfunc_multi =   lambda x: hmc_multi.multi_step(nsub, x, Nleapfrog, step_size, two_factor)
                mysamplesu, accepted, probs, counts = utils.do_hmc(stepfunc_multi, initstate, nsamples=args.nsamples, burnin=args.burnin)
                mysamples = np.array([[stansamples.constrain_pars(mysamplesu[i, j]) \
                                       for i in range(mysamplesu.shape[0])] for j in range(mysamplesu.shape[1])])
                mysamples = mysamples.transpose(1, 0, 2)
                if tnamepath: 
                    fpathd = fpath + 'step%02d_tint%03d_fac%02d_nsub%d/'%(ss*10, Tint*10, two_factor, nsub)
                else: fpathd = fpath + 'step%02d_fac%02d_nsub%d/'%(ss*10, two_factor, nsub)
                samplesaves(fpathd, mysamples, accepted, probs, counts)
            except: pass
