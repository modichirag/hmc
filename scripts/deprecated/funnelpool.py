import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
import  multiprocessing as mp
import multiprocessing as mp

from pyhmc import PyHMC
import diagnostics as dg

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=2, type=int, help='Dimensions')
parser.add_argument('--nsamples',  default=10000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
parser.add_argument('--lpath',  default=5, type=float, help='Nleapfrog*step_size')
parser.add_argument('--step_size', default=0.1, type=float,
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



def mute():
    sys.stdout = open(os.devnull, 'w')    


if ndim < 2:
    print('atleast 2 dimnesions')
    import sys
    sys.exit()
#

print("\nFor %d dimensions with step size %0.3f and %d steps\n"%(ndim, step_size, Nleapfrog))
##
fpath = './outputs/Ndim%02d/'%ndim
try: os.makedirs(fpath)
except Exception as e: print(e)
fpath = fpath + 'step%03d_nleap%02d%s/'%(step_size*100, Nleapfrog, args.suffix)
try: os.makedirs(fpath)
except Exception as e: print(e)
print("output in : ", fpath)


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
    samples = sm_funnel.sampling(iter=1, chains=1, algorithm="HMC",  n_jobs=1, verbose=False,
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
hmc = PyHMC(log_prob, grad_log_prob)
def step(x):
    return  hmc.hmc_step(x, Nleapfrog, step_size)


def do_hmc(pool):
    
    samples = []
    accepts = []
    probs = []
    start = time.time()
    initstate = np.random.uniform(-1., 1., size=nchains*ndim).reshape([nchains, ndim])
    q = initstate

    for i in range(nsamples + burnin):
        out = pool.map(step, q)
        #out = list(map(step, q))
        q = [i[0] for i in out] 
        acc = [i[2] for i in out] 
        prob = [i[3] for i in out] 
        samples.append(q)
        accepts.append(acc)
        probs.append(prob)
        
    end = time.time()
    print(end - start)
    mysamples = np.array(samples)[burnin:]
    accepted = np.array(accepts)[burnin:]
    probs = np.array(probs)[burnin:]
    print(mysamples.shape)
    np.save(fpath + '/samples', mysamples)
    np.save(fpath + '/accepted', accepted)
    np.save(fpath + '/probs', probs)
#
    dg.plot_hist(mysamples, fpath)
    dg.plot_trace(mysamples, fpath)
    dg.plot_scatter(mysamples, fpath)
    dg.plot_autorcc(mysamples, fpath)
#    
if __name__=="__main__":
    pool = mp.Pool(nparallel, initializer=mute)
    do_hmc(pool)
