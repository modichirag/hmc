import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
from mpi4py import MPI
import argparse
import arviz as az
import pystan
import contextlib
import json

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import os, sys
sys.path.append('../../src/')
from adaptivehmc import AdHMC, AdHMC_eps0, AdHMC_tint
from pyhmc import PyHMC,  PyHMC_multistep
import diagnostics as dg
import utils


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=5, type=int, help='Dimensions')
parser.add_argument('--nsamples',  default=1000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=100, type=int, help='Number of burning samples')
parser.add_argument('--step_size', default=0.1, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--mode',  default="nleap:100", type=str, help='Give nleap or tint')
#parser.add_argument('--nleap',  default=0, type=int, help='Nleapfrog')
#parser.add_argument('--Tint', default=0, type=float,
#                    help='Integration Time')
parser.add_argument('--nchains',  default=1, type=int, help='Number of chains')
parser.add_argument('--nparallel',  default=1, type=int, help='Number of parallel iterations for map')
parser.add_argument('--suffix', default='', type=str,
                    help='suffix for output folder')
parser.add_argument('--prefix', default='', type=str,
                    help='prefix for parent of output folder')
parser.add_argument('--vanilla', default=1, type=int,
                    help='vanilla run')
parser.add_argument('--normal', default=0, type=int,
                    help='vanilla run')
parser.add_argument('--smin', default=0.01, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--smax', default=1.0, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--ratio',  default=1/1.414, type=float, help='Ratio  of step size change')
parser.add_argument('--nleap',  default=10, type=int, help='Number of leapfrogs before changing step size')
parser.add_argument('--biasdn',  default=1., type=float, help='Bias weight to decreasing step size')
parser.add_argument('--biasup',  default=1., type=float, help='Bias weight to increasing step size')
parser.add_argument('--nstepseps',  default=1, type=int, help='Number of steps to get first step size')



def modeparse():
    mode, f = args.mode.split(":")
    if mode == "tint": 
        Tint = float(f)
        Nleapfrog = 0 
    elif mode == "nleap": 
        Tint = 0 
        Nleapfrog = int(f)        
    return mode, Tint, Nleapfrog


args = parser.parse_args()
ndim = args.ndim
step_size = args.step_size
nsamples = args.nsamples
mode, Tint, Nleapfrog = modeparse()
suffix = args.suffix
prefix = args.prefix
burnin = args.burnin
nchains = args.nchains
nparallel = args.nparallel
ratio = args.ratio
nleap = args.nleap
nstepseps = args.nstepseps
pwts0 = [args.biasdn, args.biasup]

if (Nleapfrog == 0) & (Tint == 0.):
    print('Either Tint or Nleap has to be non-zero')
    comm.Abort(1)    #    sys.exit(-1)

if (Nleapfrog == 0) & (args.vanilla == 0):
    print('Time of integration is being set for adaptive method')
    #comm.Abort(1)    #    sys.exit(-1)

nchains = 1
 

##
smin, smax = args.smin, args.smax
fdir = '/mnt/ceph/users/cmodi/hmc/outputs_adaptive//'
for ss in [args.prefix, 'funnel/', 'Ndim%02d/'%ndim]:
    fdir = fdir + ss
    os.makedirs(fdir, exist_ok=True)

if Tint != 0:
    foldername = 'tint%02d'%Tint
else: 
    foldername = 'nleap%02d'%Nleapfrog

if args.vanilla:
    foldername = foldername + '_vanilla'
    ratios = [1.0, 1.0, 1.0]
else:
    foldername = foldername + '_r%d'%(ratio*100)
    if args.normal : foldername = foldername + "_normal"
    ratios = [ratio, 1/ratio]
if args.suffix != "": foldername = foldername + "_" + args.suffix

print(args.ratio, ratio, foldername)
fpath = fdir + '/%s/'%foldername
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



#######

def get_logprob():
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


    fname = '../compiled_models/funnel_%sdim.pkl'%ndim

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




    start = time.time()
    samples = sm_funnel.sampling(iter=1, chains=1, algorithm="HMC", seed=100, n_jobs=1, verbose=False,
                         control={"stepsize":step_size, 
                                    "adapt_t0":False,
                                    "adapt_delta":False,
                                    "adapt_kappa":False,
                                    "metric":"unit_e",
                                  "int_time":10
                                    })
    end = time.time()

    ##LOG PROB
    log_prob = samples.log_prob
    grad_log_prob = samples.grad_log_prob

    return log_prob, grad_log_prob
        

#############################

log_prob, grad_log_prob = get_logprob()    
#hmc = PyHMC_multistep(log_prob, grad_log_prob)
if args.vanilla: hmc = AdHMC_eps0(log_prob, grad_log_prob)
elif mode == 'nleap': hmc = AdHMC(log_prob, grad_log_prob)
elif mode == 'tint': hmc = AdHMC_tint(log_prob, grad_log_prob)
def step(x):
    if args.vanilla == 1:
        return  hmc.hmc_step(x, Nleapfrog, smin=smin, smax=smax, Tint=Tint, nsteps_eps0=nstepseps)
    elif mode == 'nleap':
        return  hmc.hmc_step(x, Nleapfrog, nleap, smin=smin, smax=smax, ratios=ratios, pwts0=pwts0, nsteps_eps0=nstepseps)
    elif mode == 'tint':
        return  hmc.hmc_step(x, Tint, nleap, smin=smin, smax=smax, ratios=ratios, pwts0=pwts0, nsteps_eps0=nstepseps)
        

def do_hmc():
    
    samples = []
    accepts = []
    probs = []
    checks = []
    counts = []
    start = time.time()
    initstate = np.random.uniform(-1., 1., size=nchains*ndim).reshape([nchains, ndim])
    q = initstate

    for i in range(nsamples + burnin):
        if i%1000 == 0: print("In rank %d, iteration #%d"%(rank, i))
        #with contextlib.redirect_stderr(io.StringIO()):
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            out = list(map(step, q))
        q = [i[0] for i in out] 
        acc = [i[2] for i in out]
        prob = [i[3] for i in out] 
        check = [i[4] for i in out] 
        count = [i[5] for i in out] 
        samples.append(q)
        accepts.append(acc)
        probs.append(prob)
        checks.append(check)
        counts.append(count)

    end = time.time()
    print(rank, end - start)
    mysamples = np.array(samples)[burnin:]
    accepted = np.array(accepts)[burnin:]
    probs = np.array(probs)[burnin:]
    checks = np.array(checks)[burnin:]
    counts = np.array(counts)[burnin:]
    

    try: os.makedirs(fpath +'/samples/')
    except: pass
    try: os.makedirs(fpath +'/probs/')
    except: pass
    try: os.makedirs(fpath +'/accepted/')
    except: pass
    try: os.makedirs(fpath +'/checks/')
    except: pass
    try: os.makedirs(fpath +'/counts/')
    except: pass
    np.save(fpath + '/samples/%02d'%rank, mysamples)
    np.save(fpath + '/probs/%02d'%rank, probs)
    np.save(fpath + '/accepted/%02d'%rank, accepted)
    np.save(fpath + '/checks/%02d'%rank, checks)
    np.save(fpath + '/counts/%02d'%rank, counts)
    return mysamples, accepted, probs, checks, counts

     
if __name__=="__main__":
    mysamples, accepted, probs, checks, counts = do_hmc()
    print(rank, mysamples.shape)
    #print(rank, mysamples)

    #sys.exit(-1)
    try:
        mysamples = comm.gather(mysamples, root=0)
        accepted = comm.gather(accepted, root=0)
        probs = comm.gather(probs, root=0)
        counts = comm.gather(counts, root=0)
        checks = comm.gather(checks, root=0)
    except Exception as e:
        print(rank, e)
        comm.Abort(1) 

    print('Saved in %s'%fpath)
    if rank == 0:
        mysamples = np.concatenate(mysamples, axis=1)
        accepted = np.concatenate(accepted, axis=1)
        probs = np.concatenate(probs, axis=1)
        checks  = np.concatenate(checks, axis=1)
        counts  = np.concatenate(counts, axis=1)
        print(mysamples.shape)

        np.save(fpath + '/samples', mysamples)
        np.save(fpath + '/accepted', accepted)
        np.save(fpath + '/probs', probs)
        np.save(fpath + '/checks', checks)
        np.save(fpath + '/counts', counts)

        ###Deleting rank saves to prevent double saving
        print('Deleting rank folders')
        import shutil
        shutil.rmtree(fpath + '/samples')
        shutil.rmtree(fpath + '/accepted')
        shutil.rmtree(fpath + '/probs')
        shutil.rmtree(fpath + '/checks')
        shutil.rmtree(fpath + '/counts')
    
        print('Saved in %s'%fpath)


        ##Metrics
        sigmarep = np.random.normal(0,3,10000000)
        yvrep = np.random.normal(0,1,10000000)
        alpharep = np.exp(sigmarep/2.)*yvrep
        smean, sstd = 0, 3.
        amean, astd = 0, alpharep.std()
        s2mean, s2std = (sigmarep**2).mean(), (sigmarep**2).std()
        a2mean, a2std = (alpharep**2).mean(), (alpharep**2).std()

        means = np.array([smean] + [amean]*(ndim-1))
        stds = np.array([sstd] + [astd]*(ndim-1))
        ess, esschain = utils.get_ess_true(mysamples, means, stds)
        print("ESS : ", ess)
        print(counts.shape)
        print("Counts : ", counts.sum(axis=(0, 1)))
    
        
        checkdict = {}
        checkdict['ess'] = ess
        checkdict['mean'] = mysamples.mean(axis=(0, 1))
        checkdict['std'] = mysamples.std(axis=(0, 1))
        checkdict['counts'] = counts.sum(axis=(0, 1))
        checkdict['acceptance'] = (accepted == 1).sum()/accepted.size
        checkdict['costV_g'] = checkdict['counts'][1]/ess
        checkdict['cost'] = checkdict['counts'][:2].sum()/ess
        #az ess
        azdata = az.convert_to_inference_data(np.transpose(mysamples, (1, 0, 2)))
        azess = np.array(list(az.ess(azdata).values()))[0]
        checkdict['azess'] = azess

        #Dump
        for key in checkdict.keys():
            try: checkdict[key] = list(checkdict[key].astype(np.float64))
            except Exception as e: print(e)
        with open(fpath + 'checks.json', 'w') as fp:
            json.dump(checkdict, fp, sort_keys=True, indent=4)
 
        
        rcc = np.zeros_like(mysamples)
        tcc = np.zeros([rcc.shape[0], rcc.shape[1]])
        for i in range(rcc.shape[1]):
            for j in range(rcc.shape[2]):
                rcc[:, i, j] = az.autocorr(mysamples[:, i, j])
                for m in range(500):
                    if m > 5*(1 + 2*rcc[:m, i, j].sum()): break
                tcc[i, j] = 1 + 2*rcc[:m, i, j].sum()
        np.save(fpath + 'azrcc', rcc)
        np.save(fpath + 'azrclenght', tcc)

        ###Funnel specific plots
        ###
        nbins = 30
        if nsamples > 5000: sub = nsamples//5000
        else: sub = 1
        print('Subsample for figure with : ', sub)
        lbl = '%0.2f (%0.2f)'%(checkdict['mean'][0], checkdict['std'][0])
        plt.hist(mysamples[::sub, ..., 0].flatten(), bins=nbins, range=(-10, 10), alpha=0.8, density=True, label=lbl)
        plt.hist(sigmarep[:10000].flatten(), bins=nbins, range=(-10, 10), color='k', lw=2, histtype='step', density=True)
        plt.grid(which='both')
        plt.legend()
        plt.savefig(fpath + '/hist_sigma.png')
        plt.close()

        nplot = ndim-1
        if nplot > 4: nplot = 4
        fig, ax = plt.subplots(1, nplot, figsize = (nplot*4-3, 4), sharex=True, sharey=True)
        if nplot == 2: ax = [ax]
        rrange = (-50, 50)
        nbins = 100
        for i in range(1, min(5, 1+nplot)):
            lbl = '%0.2f (%0.2f)'%(checkdict['mean'][i], checkdict['std'][i])
            ax[i-1].hist(mysamples[::sub, ..., i].flatten(), bins=nbins, alpha=0.8, label=lbl, density=True, range=rrange)
            ax[i-1].hist(alpharep[:100000].flatten(), bins=nbins, color='k', lw=2, histtype='step', density=True, range=rrange)
            ax[i-1].grid(which='both')
            ax[i-1].set_yscale('log')
            ax[i-1].set_ylim(1e-3, 1)
            plt.tight_layout()
            ax[i-1].legend()
        plt.savefig(fpath + '/hist_alpha.png')
        plt.close()


        
        
        
