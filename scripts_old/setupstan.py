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



def setup(model, data, args, fpath):

 
    fpathparams = fpath
    fpath = fpath + 'default/'
    os.makedirs(fpath, exist_ok=True)
    os.makedirs(fpath + '/samples/', exist_ok=True)
    fpath2 = fpath + 'metrics/'
    os.makedirs(fpath2, exist_ok=True)
    print("output in : ", fpath)
    
    niter = args.nsamples
    ######################################################
    #NUTS Adaptations
    samples = model.sampling(data=data, chains=1, warmup=niter, 
                                          iter=niter*2, n_jobs=1,
                                          control={"metric":"diag_e", "adapt_delta":args.adelta})

    samplesy = samples.extract(permuted=False)[..., :-1]
    ndim = samplesy.shape[-1]
    np.save(fpath + 'samples/%d'%rank, samplesy)

    np.save(fpath2 + 'stepsize%02d'%rank, np.array([p['stepsize__'] for p in samples.get_sampler_params()][0]))
    np.save(fpath2 + 'invmetric%02d'%rank, samples.get_inv_metric()[0])
    np.save(fpath2 + 'nleaprfrog%02d'%rank,  np.array([p['n_leapfrog__'] for p in samples.get_sampler_params()][0]))


    stepsizes0 = comm.gather(samples.get_stepsize()[0], root=0)
    try: invmetrics = comm.gather(samples.get_inv_metric()[0], root=0)
    except: invmetrics  = [np.ones(ndim)]*size
    nleapfrogs = comm.gather([p['n_leapfrog__'][niter:] for p in samples.get_sampler_params()][0], root=0)
    stepsizes = comm.gather([p['stepsize__'][niter:] for p in samples.get_sampler_params()][0], root=0)
    divs = comm.gather([p['divergent__'][niter:] for p in samples.get_sampler_params()][0], root=0)

    if args.gather: refsamples = comm.gather(samplesy, root=0)
    Tint = None
    if rank ==0:

        print('In loop for rank 0') 
        if args.gather: 
            refsamples = np.concatenate(refsamples, axis=1)
            np.save(fpath + 'samples', refsamples)
            print("samples shape : ", refsamples.shape)                              

        stepsizefid = np.array(stepsizes0).mean()
        np.save(fpath2 + 'stepsizefid', np.array(stepsizes0))
        invmetricfid = np.array(invmetrics).mean(axis=0)

        nleapfrogs = np.concatenate(nleapfrogs)
        stepsizes = np.concatenate(stepsizes)
        divs = np.concatenate(divs)

        #nss = nleapfrogs[np.where((abs(stepsizes - stepsizefid) < stepsizefid/10.) & (divs==0))]
        nss = nleapfrogs.copy() #[np.where((abs(stepsizes - stepsizefid) < stepsizefid/10.) & (divs==0))]
        Tint = stepsizefid*np.quantile(nss, args.nssq)
        Nleapfrogfid = int(Tint/stepsizefid)
        Tint = comm.bcast(Tint, root=0)


        todump = {}
        todump['stepsize'] = stepsizefid
        todump['invmetric'] = list(invmetricfid)
        todump['Tintegration'] = Tint
        todump['Nleapfrog'] = Nleapfrogfid
        todump['Ndim'] = ndim

        print(todump)

        with open(fpathparams + 'nutsparams.json', 'w') as fp: 
            json.dump(todump, fp, sort_keys=True, indent=4)

    comm.Barrier()
    print(rank, Tint)
    Tint = comm.bcast(Tint, root=0)
    comm.Barrier()
    print(rank, Tint)
    comm.Barrier()
    ######################################################
    #HMC Adaptations
    if args.tnuts == 1:
        samples = model.sampling(data=data, chains=1, warmup=niter, algorithm='HMC',
                             iter=niter*2, n_jobs=1,
                             control={"metric":"diag_e", 
                                      "int_time":Tint, 
                                      "stepsize_jitter":0, "adapt_delta":args.adelta})
    else:
        samples = model.sampling(data=data, chains=1, warmup=niter, algorithm='HMC',
                             iter=niter*2, n_jobs=1,
                             control={"metric":"diag_e", 
                                      "stepsize_jitter":0, "adapt_delta":args.adelta})
        Tint = np.pi*2

    np.save(fpath2 + 'stepsizehmc%02d'%rank, np.array([p['stepsize__'] for p in samples.get_sampler_params()][0]))
    np.save(fpath2 + 'invmetrichmc%02d'%rank, samples.get_inv_metric()[0])
    np.save(fpath2 + 'nleaprfroghmc%02d'%rank,  np.array([p['int_time__']/p['stepsize__'] for p in samples.get_sampler_params()][0]))


    stepsizes0 = comm.gather(samples.get_stepsize()[0], root=0)
    try: invmetrics = comm.gather(samples.get_inv_metric()[0], root=0)
    except: invmetrics  = [np.ones(ndim)]*size
    inttime = comm.gather([p['int_time__'][niter:] for p in samples.get_sampler_params()][0], root=0)
    stepsizes = comm.gather([p['stepsize__'][niter:] for p in samples.get_sampler_params()][0], root=0)

    samplesy = samples.extract(permuted=False)[..., :-1]
    ndim = samplesy.shape[-1]
    np.save(fpath + 'samples/hmc%d'%rank, samplesy)

    if args.gather:
        refsamples = comm.gather(samplesy, root=0)
    if rank ==0:
        print('In loop for rank 0') 
        if args.gather:
            refsamples = np.concatenate(refsamples, axis=1)
            np.save(fpath + 'sampleshmc', refsamples)
            print("samples shape : ", refsamples.shape)


        stepsizefid = np.array(stepsizes0).mean()
        np.save(fpath2 + 'stepsizefidhmc', np.array(stepsizes0))
        print(stepsizefid)
        invmetricfid = np.array(invmetrics).mean(axis=0)
        print(invmetricfid)

        #inttime = np.concatenate(inttime)
        #stepsizes = np.concatenate(stepsizes)
        #Tint = inttime.mean()
        Nleapfrogfid = int(Tint/stepsizefid)

        todump = {}
        todump['stepsize'] = stepsizefid
        todump['invmetric'] = list(invmetricfid)
        todump['Tintegration'] = Tint
        todump['Nleapfrog'] = Nleapfrogfid
        todump['Ndim'] = ndim

        print(todump)
        print(fpathparams)
        print(fpath)

        with open(fpathparams + 'params.json', 'w') as fp: 
            json.dump(todump, fp, sort_keys=True, indent=4)

    comm.Barrier()
