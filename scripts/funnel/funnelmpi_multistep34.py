import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


from pyhmc import PyHMC_multistep_tries3
import diagnostics as dg
import pystan

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=2, type=int, help='Dimensions')
parser.add_argument('--nsamples',  default=10000, type=int, help='Number of ssmples')
parser.add_argument('--burnin',  default=1000, type=int, help='Number of burning samples')
parser.add_argument('--lpath',  default=5, type=float, help='Nleapfrog*step_size')
parser.add_argument('--step_size', default=0.1, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--nsub', default=2, type=int,
                    help='Number of adaptations for HMC')
parser.add_argument('--two_factor', default=2, type=float,
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

nskip = 1

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
fpath = '/mnt/ceph/users/cmodi/hmc/outputs_long3/funnel4/Ndim%02d//'%ndim
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
    v ~ normal(0, 4);
    theta ~ normal(0, exp(v/2));
    }
    """%(ndim-1)


    fname = './models/funnel4_%sdim.pkl'%ndim

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
hmc = PyHMC_multistep_tries3(log_prob, grad_log_prob)
def step(x):
    return  hmc.multi_step(nsub, x, Nleapfrog, step_size, two_factor)


def do_hmc():
    
    samples = []
    accepts = []
    probs = []
    counts = []
    ptries = []
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
        ptries = [i[5] for i in out] 
        samples.append(q)
        accepts.append(acc)
        probs.append(prob)
        counts.append(count)
        ptries.append(ptries)

    end = time.time()
    print(rank, end - start)
    mysamples = np.array(samples)[burnin::nskip]
    accepted = np.array(accepts)[burnin::nskip]
    probs = np.array(probs)[burnin::nskip]
    counts = np.array(counts)[burnin::nskip]
    ptries = np.array(ptries)[burnin::nskip]

    try: os.makedirs(fpath +'/samples/')
    except: pass
    try: os.makedirs(fpath +'/probs/')
    except: pass
    try: os.makedirs(fpath +'/accepts/')
    except: pass
    try: os.makedirs(fpath +'/counts/')
    except: pass
    try: os.makedirs(fpath +'/ptries/')
    except: pass
    np.save(fpath + '/samples/%02d'%rank, mysamples)
    np.save(fpath + '/probs/%02d'%rank, probs)
    np.save(fpath + '/accepts/%02d'%rank, accepted)
    np.save(fpath + '/counts/%02d'%rank, counts)
    np.save(fpath + '/ptries/%02d'%rank, ptries)

    return mysamples, accepted, probs, counts, ptries
     
if __name__=="__main__":

    try: 
        ss = np.load(fpath +'/samples/00.npy')
        if ss.shape[0] == nsamples:
            print('\nAlready done correctly with all samples\n')
            comm.Abort(1)    #    sys.exit(-1)
            sys.exit()
    except Exception as e: 
        print(e)
                 
    mysamples, accepted, probs, counts, ptries = do_hmc()
    print(rank, mysamples.shape)
    #print(rank, mysamples)
    
    try:
        mysamples = comm.gather(mysamples, root=0)
        accepted = comm.gather(accepted, root=0)
        probs = comm.gather(probs, root=0)
        counts = comm.gather(counts, root=0)
        ptries = comm.gather(ptries, root=0)
    except Exception as e:
        print(rank, e)
        comm.Abort(1)    #    sys.exit(-1)
    
    if rank == 0:
        mysamples = np.concatenate(mysamples, axis=1)
        accepted = np.concatenate(accepted, axis=1)
        probs = np.concatenate(probs, axis=1)
        counts  = np.concatenate(counts, axis=1)
        ptries  = np.concatenate(ptries, axis=1)
        print(mysamples.shape)

        np.save(fpath + '/samples', mysamples)
        np.save(fpath + '/accepted', accepted)
        np.save(fpath + '/probs', probs)
        np.save(fpath + '/counts', counts)
        np.save(fpath + '/ptries', ptries)
    
        print('Saved in %s'%fpath)
        print("\nFor %d dimensions with step size %0.3f and %d leapfrog steps\n"%(ndim, step_size, Nleapfrog))
        print("\nSubsize with %d upto %d times\n"%(two_factor, nsub))
        start = time.time()
        dg.plot_hist(mysamples, fpath, sub=1000)
        print(time.time() - start)
        start = time.time()
        #dg.plot_trace(mysamples, fpath)
        print(time.time() - start)
        start = time.time()
        dg.plot_scatter(mysamples, fpath)
        print(time.time() - start)
        start = time.time()
        #dg.plot_autorcc(mysamples, fpath)
        print(time.time() - start)
        start = time.time()
    #   
