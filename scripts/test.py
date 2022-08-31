import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


sys.path.append('../src/')
from algorithms import alg_dict, HMC, GHMC, Persistent_GHMC
from stan_models import get_bridgestan_model, constrain_samples

sys.path.append(os.getcwd() + '/..')
import PythonClient as pbs

MODELDIR = '../compiled_models/'

def setup_model():    

    #modelname = 'gaussian'
    #modelname = 'normal_02'
    modelname = 'funnel_02'

    lib = MODELDIR + "/%s/%s_model.so"%(modelname, modelname)
    data = MODELDIR + "/%s/%s.data.json"%(modelname, modelname)

    model = get_bridgestan_model(lib, data)
    print(model)
    D = model.dims()
    log_density = model.log_density
    grad_log_density = lambda x: model.log_density_gradient(x)[1]

    refsamples = None
    if "normal" in lib: refsamples = np.random.normal(0, 1, 10000*D).reshape(-1, D)
    return model, D, log_density, grad_log_density, refsamples


###
def plot_samples(samples, fname='tmp', refsamples=None):

    ndim = samples.shape[-1]
    fig, ax = plt.subplots(1, ndim, figsize = (ndim*3, 3))
    for i in range(ndim):
        ax[i].hist(samples[:, i].flatten(), bins=100, alpha=0.5, density=True)
        if refsamples is not None:
            ax[i].hist(refsamples[:, i].flatten(), bins='auto', alpha=1, color='k', histtype='step', density=True)
        ax[i].grid(which='both')
        #ax[i].set_yscale('log')
    plt.tight_layout()
    plt.savefig('./tmp_figs/'+fname)
    plt.close()

###
def callback(state, nsamples, nprint):
    if (nprint*state.i)%nsamples == 0: 
        accepts = np.array(state.accepts)
        print("Iteration %d"%state.i)
        acc_fraction = (accepts==1).sum()/accepts.size
        print("Accepted = %0.3f"%acc_fraction)


###
def test_alg(alg, nleap=10, stepsize=0.1, nsamples=1000, burnin=10, skip=1, nprint=10, **kwargs):
    np.random.seed(0)
    print('\nalgorithm : ', alg)
    model, D, log_density, grad_log_density, refsamples = setup_model()
    kernel = alg_dict[alg](log_density, log_prob_and_grad=model.log_density_gradient)
    
    #
    q, p = np.random.normal(0, 1, D), np.random.normal(0, 1, D)
    callback_fn = lambda x: callback(x, nsamples, nprint)
    start = time.time()
    state = kernel.sample(q=q, nsamples=nsamples, burnin=burnin, 
                          nleap=nleap, step_size=stepsize,
                          callback=callback_fn, **kwargs)
    print("Time taken : ", time.time()-start)

    mysamples = constrain_samples(model, state.samples)
    print("mean : ", mysamples.mean(axis=0))
    print("std : ", mysamples.std(axis=0)) 
    print(state.counts.sum(axis=0))
    plot_samples(mysamples, '%s.png'%alg, refsamples)



stepsize = 0.1
drstepsize = 0.1
nsamples = 1000
nsamplesg = 5000
alpha = 0.8
delta = 0.1
nleap = 100 

#test_alg('hmc', nleap=nleap, nsamples=nsamples, stepsize=stepsize, skip=1)

#test_alg('ghmc', nleap=1, nsamples=nsamplesg, stepsize=stepsize, alpha=alpha, skip=10)

#test_alg('pghmc', nleap=1, nsamples=nsamplesg, stepsize=stepsize, alpha=alpha, delta=delta, skip=10, )

#test_alg('drghmc', nleap=1, nsamples=nsamplesg, stepsize=drstepsize, alpha=alpha, delta=delta, nprops=2, stepfactor=5, skip=10)

test_alg('drhmc', nleap=nleap, nsamples=nsamples, stepsize=drstepsize, nprops=2, stepfactor=5, skip=1)
