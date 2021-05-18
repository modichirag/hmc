import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


from adaptivehmc import AdHMC
from pyhmc import PyHMC,  PyHMC_multistep
import diagnostics as dg
import pystan

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=2, type=int, help='Dimensions')
parser.add_argument('--nsamples',  default=1000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=100, type=int, help='Number of burning samples')
parser.add_argument('--nleap',  default=10, type=int, help='Nleapfrog')
parser.add_argument('--step_size', default=0.1, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--Tint', default=6.28, type=float,
                    help='Integration Time')
parser.add_argument('--nchains',  default=1, type=int, help='Number of chains')
parser.add_argument('--nparallel',  default=1, type=int, help='Number of parallel iterations for map')
parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')
parser.add_argument('--vanilla', default=1, type=int,
                    help='vanilla run')

args = parser.parse_args()
ndim = args.ndim
step_size = args.step_size
nsamples = args.nsamples
Tint = args.Tint
suffix = args.suffix
burnin = args.burnin
nchains = args.nchains
nparallel = args.nparallel
Nleapfrog = args.nleap



#np.random.seed(100)
def mute():
    sys.stdout = open(os.devnull, 'w')    


if ndim < 2:
    print('atleast 2 dimnesions')
    import sys
    sys.exit()
#

##
smin, smax = 0.01, 1.0
fdir = '/mnt/ceph/users/cmodi/hmc/outputs_adaptive/funnel//'
fpath = fdir + '/Ndim%02d//'%ndim
try: os.makedirs(fpath)
except Exception as e: print(e)
if args.vanilla:
    fpath = fpath + 'nleap%02d_vanilla/'%(Nleapfrog)
    ratios = [1.0, 1.0, 1.0]
else:
    fpath = fpath + 'nleap%02d/'%(Nleapfrog)
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


    fname = './models/funnel_%sdim.pkl'%ndim

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
hmc = AdHMC(log_prob, grad_log_prob)
def step(x):
    if args.vanilla == 1:
        return  hmc.hmc_step_vanilla(x, Nleapfrog, smin=smin, smax=smax)
        #return  hmc.hmc_step(x, Nleapfrog, smin=smin, smax=smax, ratios=ratios)
    else:
        return  hmc.hmc_step(x, Nleapfrog, smin=smin, smax=smax, ratios=ratios)


def do_hmc():
    
    samples = []
    accepts = []
    probs = []
    diags = []
    start = time.time()
    initstate = np.random.uniform(-1., 1., size=nchains*ndim).reshape([nchains, ndim])
    #initstate = np.random.uniform(1., 3., size=nchains*ndim).reshape([nchains, ndim])
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
    mysamples = np.array(samples)[burnin:]
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
