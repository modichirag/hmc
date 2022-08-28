import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pystan
import arviz as az
import argparse
import json

import os, sys
sys.path.append('../../src/')
from pyhmc import PyHMC_multistep
import diagnostics as dg
import utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=5, type=int, help='Dimensions')
parser.add_argument('--nsamples',  default=10000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
parser.add_argument('--lpath',  default=10, type=float, help='Nleapfrog*step_size')
parser.add_argument('--step_size', default=0.1, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--nsub', default=2, type=int,
                    help='Number of adaptations for HMC')
parser.add_argument('--two_factor', default=5, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--nchains',  default=1, type=int, help='Number of chains')
parser.add_argument('--nparallel',  default=1, type=int, help='Number of parallel iterations for map')

parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
ndim = args.ndim
step_size = args.step_size
nsub = args.nsub
two_factor = args.two_factor
nsamples = args.nsamples
Lpath = args.lpath
suffix = args.suffix
burnin = args.burnin
nchains = args.nchains
nparallel = args.nparallel
Nleapfrog = int(Lpath / step_size)



#np.random.seed(100)
def mute():
    sys.stdout = open(os.devnull, 'w')    


if ndim < 2:
    print('atleast 2 dimnesions')
    import sys
    sys.exit()
#

print("\nFor %d dimensions with step size %0.3f and %d steps\n"%(ndim, step_size, Nleapfrog))
##
#fpath = './outputs_long/Ndim%02d//'%ndim
fpath = '/mnt/ceph/users/cmodi/hmc/outputs_adaptive/funnel/Ndim%02d//'%ndim
try: os.makedirs(fpath)
except Exception as e: print(e)
fpath = fpath + 'step%03d_nleap%02d_fac%02d_nsub%d/'%(step_size*100, Nleapfrog, two_factor, nsub)
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
            sys.exit()
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
                                  "int_time":Lpath
                                    })
    end = time.time()

    ##LOG PROB
    log_prob = samples.log_prob
    grad_log_prob = samples.grad_log_prob

    return log_prob, grad_log_prob
        

#############################

log_prob, grad_log_prob = get_logprob()    
hmc = PyHMC_multistep(log_prob, grad_log_prob)
def step(x):
    return  hmc.multi_step(nsub, x, Nleapfrog, step_size, two_factor)


def do_hmc():
    
    samples = []
    accepts = []
    probs = []
    counts = []
    start = time.time()
    initstate = np.random.uniform(-1., 1., size=nchains*ndim).reshape([nchains, ndim])
    q = initstate

    for i in range(nsamples + burnin):
        #out = pool.map(step, q)
        out = list(map(step, q))
        q = [i[0] for i in out] 
        acc = [i[2] for i in out]
        prob = [i[3] for i in out] 
        count = [i[4] for i in out] 
        samples.append(q)
        accepts.append(acc)
        probs.append(prob)
        counts.append(count)

    end = time.time()
    print(rank, end - start)
    mysamples = np.array(samples)[burnin:]
    accepted = np.array(accepts)[burnin:]
    probs = np.array(probs)[burnin:]
    counts = np.array(counts)[burnin:]

    try: os.makedirs(fpath +'/samples/')
    except: pass
    try: os.makedirs(fpath +'/probs/')
    except: pass
    try: os.makedirs(fpath +'/accepted/')
    except: pass
    try: os.makedirs(fpath +'/counts/')
    except: pass
    np.save(fpath + '/samples/%02d'%rank, mysamples)
    np.save(fpath + '/probs/%02d'%rank, probs)
    np.save(fpath + '/accepted/%02d'%rank, accepted)
    np.save(fpath + '/counts/%02d'%rank, counts)

    return mysamples, accepted, probs, counts
     
if __name__=="__main__":
    mysamples, accepted, probs, counts = do_hmc()
    print(rank, mysamples.shape)
    #print(rank, mysamples)
    
    try:
        mysamples = comm.gather(mysamples, root=0)
        accepted = comm.gather(accepted, root=0)
        probs = comm.gather(probs, root=0)
        counts = comm.gather(counts, root=0)
    except Exception as e:
        print(rank, e)
        comm.Abort(1)    #    sys.exit(-1)
    
    if rank == 0:
        mysamples = np.concatenate(mysamples, axis=1)
        accepted = np.concatenate(accepted, axis=1)
        probs = np.concatenate(probs, axis=1)
        counts  = np.concatenate(counts, axis=1)
        print(mysamples.shape)

        np.save(fpath + '/samples', mysamples)
        np.save(fpath + '/accepted', accepted)
        np.save(fpath + '/probs', probs)
        np.save(fpath + '/counts', counts)
    

        ###Deleting rank saves to prevent double saving
        print('Deleting rank folders')
        import shutil
        shutil.rmtree(fpath + '/samples')
        shutil.rmtree(fpath + '/accepted')
        shutil.rmtree(fpath + '/probs')
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


        
        
        
