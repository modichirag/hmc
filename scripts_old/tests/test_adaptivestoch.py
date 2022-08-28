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
parser.add_argument('--vanilla', default=1, type=int,
                    help='vanilla run')
parser.add_argument('--normal', default=0, type=int,
                    help='vanilla run')
parser.add_argument('--smin', default=0.01, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--smax', default=1.0, type=float,
                    help='sum the integers (default: find the max)')
parser.add_argument('--ratio',  default=0.9, type=float, help='Ratio  of step size change')


def modeparse():
    mode, f = args.mode.split(":")
    if mode == "tint": 
        Tint = float(f)
        Nleapfrog = 0 
    elif mode == "nleap": 
        Tint = 0 
        Nleapfrog = int(f)        
    return Tint, Nleapfrog


args = parser.parse_args()
ndim = args.ndim
step_size = args.step_size
nsamples = args.nsamples
Tint, Nleapfrog = modeparse()
suffix = args.suffix
burnin = args.burnin
nchains = args.nchains
nparallel = args.nparallel
ratio = args.ratio

if (Nleapfrog == 0) & (Tint == 0.):
    print('Either Tint or Nleap has to be non-zero')
    comm.Abort(1)    #    sys.exit(-1)

if (Nleapfrog == 0) & (args.vanilla == 0):
    print('Time of integration can only be set by vanilla method')
    comm.Abort(1)    #    sys.exit(-1)

nchains = 1
 

##
smin, smax = args.smin, args.smax
fdir = '/mnt/ceph/users/cmodi/hmc/outputs_adaptive//'
for ss in [args.suffix, 'funnel/', 'Ndim%02d/'%ndim]:
    fdir = fdir + ss
    try: 
        os.makedirs(fdir)
    except Exception as e: print(e)
#fdir = fdir + '/funnel/'
#fdir = fdir + 'Ndim%02d//'%ndim
#try: os.makedirs(fdir)
#except Exception as e: print(e)
if Tint != 0:
    pref = 'tint%02d'%Tint
else: 
    pref = 'nleap%02d'%Nleapfrog

if args.vanilla:
    pref = pref + '_vanilla'
    ratios = [1.0, 1.0, 1.0]
else:
    pref = pref + '_r%d'%(ratio*10)
    if args.normal : pref = pref + "_normal"
    ratios = [ratio, 1., 1/ratio]

#if args.vanilla:
#    if Nleapfrog == 0: fpath = fdir + 'tint%02d_vanilla/'%(Tint)
#    else: fpath = fdir + 'nleap%02d_vanilla/'%(Nleapfrog)
#    ratios = [1.0, 1.0, 1.0]
#else:
#    pref = pref + '_r%d'%(ratio*10)
#    if Nleapfrog == 0: fpath = fdir + 'tint%02d/'%(Tint)
#    else: fpath = fdir + 'nleap%02d/'%(Nleapfrog)
#    if args.normal:
#        if Nleapfrog == 0: 
#            fpath = fdir + 'tint%02d_normal/'%(Tint)
#        else: fpath = fdir + 'nleap%02d_normal/'%(Nleapfrog)
#    ratios = [ratio, 1.0, 1/ratio]

fpath = fdir + '/%s'%pref
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
        return  hmc.hmc_step_vanilla(x, Nleapfrog, smin=smin, smax=smax, Tint=Tint)
    else:
        #if args.normal:
        #    return  hmc.hmc_step_normal(x, Nleapfrog, smin=smin, smax=smax, ratios=ratios, Tint=Tint)
        return  hmc.hmc_step(x, Nleapfrog, smin=smin, smax=smax, ratios=ratios, Tint=Tint, normprob=args.normal)
        

def do_hmc():
    
    samples = []
    accepts = []
    probs = []
    diags = []
    counts = []
    start = time.time()
    initstate = np.random.uniform(-1., 1., size=nchains*ndim).reshape([nchains, ndim])
    #initstate = np.random.uniform(1., 3., size=nchains*ndim).reshape([nchains, ndim])
    q = initstate

    for i in range(nsamples + burnin):
        if i%100 == 0: print("In rank %d, iteration #%d"%(rank, i))
        #out = pool.map(step, q)
        out = list(map(step, q))
        q = [i[0] for i in out] 
        acc = [i[2] for i in out]
        prob = [i[3] for i in out] 
        diag = [i[4] for i in out] 
        count = [i[5] for i in out] 
        samples.append(q)
        accepts.append(acc)
        probs.append(prob)
        diags.append(diag)
        counts.append(count)

    end = time.time()
    print(rank, end - start)
    mysamples = np.array(samples)[burnin:]
    accepted = np.array(accepts)[burnin:]
    probs = np.array(probs)[burnin:]
    diags = np.array(diags)[burnin:]
    counts = np.array(counts)[burnin:]

    try: os.makedirs(fpath +'/samples/')
    except: pass
    try: os.makedirs(fpath +'/probs/')
    except: pass
    try: os.makedirs(fpath +'/accepted/')
    except: pass
    try: os.makedirs(fpath +'/diags/')
    except: pass
    try: os.makedirs(fpath +'/counts/')
    except: pass
    np.save(fpath + '/samples/%02d'%rank, mysamples)
    np.save(fpath + '/probs/%02d'%rank, probs)
    np.save(fpath + '/accepted/%02d'%rank, accepted)
    np.save(fpath + '/diags/%02d'%rank, diags)
    np.save(fpath + '/counts/%02d'%rank, counts)
    return mysamples, accepted, probs, diags, counts

     
if __name__=="__main__":
    mysamples, accepted, probs, diags, counts = do_hmc()
    print(rank, mysamples.shape)
    #print(rank, mysamples)

    #sys.exit(-1)
    try:
        mysamples = comm.gather(mysamples, root=0)
        accepted = comm.gather(accepted, root=0)
        probs = comm.gather(probs, root=0)
        counts = comm.gather(counts, root=0)
        diags = comm.gather(diags, root=0)
    except Exception as e:
        print(rank, e)
        comm.Abort(1)    #    sys.exit(-1)

    print('Saved in %s'%fpath)
    if rank == 0:
        mysamples = np.concatenate(mysamples, axis=1)
        accepted = np.concatenate(accepted, axis=1)
        probs = np.concatenate(probs, axis=1)
        diags  = np.concatenate(diags, axis=1)
        counts  = np.concatenate(counts, axis=1)
        print(mysamples.shape)

        np.save(fpath + '/samples', mysamples)
        np.save(fpath + '/accepted', accepted)
        np.save(fpath + '/probs', probs)
        np.save(fpath + '/diags', diags)
        np.save(fpath + '/counts', counts)
    
        print('Saved in %s'%fpath)

##        start = time.time()
##        dg.plot_hist(mysamples, fpath, sub=1)
##        print(time.time() - start)
##        start = time.time()
##        #dg.plot_trace(mysamples, fpath)
##        print(time.time() - start)
##        start = time.time()
##        dg.plot_scatter(mysamples, fpath, sub=1)
##        print(time.time() - start)
##        start = time.time()
##        #dg.plot_autorcc(mysamples, fpath)
##        print(time.time() - start)
##        start = time.time()
##    #   
