import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os

import tensorflow as tf
#tf.config.threading.set_inter_op_parallelism_threads(8)
#tf.compat.v1.disable_eager_execution()
import tensorflow_probability as tfp

from myhmc import KE, get_grads, leapfrog, metropolis
import diagnostics as dg

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=2, type=int, help='Dimensions')
parser.add_argument('--nparallel',  default=8, type=int, help='Dimensions')
parser.add_argument('--step_size', default=0.1, type=float,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
ndim = args.ndim
step_size = args.step_size
nparallel = args.nparallel
#nsamples, burnin = 100, 100
#nchains = 2
nsamples, burnin = 1000, 100
nchains = 10
if ndim < 2:
    print('atleast 2 dimnesions')
    import sys
    sys.exit()
initstate = tf.random.uniform([nchains, ndim], -1., 1. )
#
Lpath = 1
Nleapfrog = int(Lpath / step_size)
Nleapfrog = max(10, Nleapfrog)



# Initialize the HMC transition kernel.
num_results = nsamples
num_burnin_steps = burnin

print("\nFor %d dimensions with step size %0.3f and %d steps\n"%(ndim, step_size, Nleapfrog))
##
fpath = './outputs/Ndim%02d/'%ndim
try: os.makedirs(fpath)
except Exception as e: print(e)
fpath = fpath + 'step%03d_nleap%02d_test/'%(step_size*100, Nleapfrog)
try: os.makedirs(fpath)
except Exception as e: print(e)
print("output in : ", fpath)


#######

normal = tfp.distributions.Normal

def log_prob(x):
    if len(x.shape) > 1:
        yp = normal(0, 3).log_prob(x[:, 0])
        xp = tf.reduce_sum(tf.stack([normal(0, tf.exp(x[:, 0]/2.)).log_prob(x[:, i]) for i in range(1, x.shape[1])], 1), 1)
    else:
        yp = normal(0, 3).log_prob(x[0])
        xp = tf.reduce_sum([normal(0, tf.exp(x[0]/2.)).log_prob(x[i]) for i in range(1, x.shape[0])])
    return yp + xp


#######
V = lambda x: -1* log_prob(x)
@tf.function
def H(q,p):
    return V(q) + KE(p)

#######


######
hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob,
        num_leapfrog_steps=Nleapfrog,
        step_size=step_size)


# Run the chain (with burn-in).
@tf.function
def run_chain():
    # Run the chain (with burn-in).
    samples, is_accepted = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state= initstate,
      kernel=hmc,
        trace_fn=lambda _, pkr: pkr.is_accepted,
        parallel_iterations=nparallel)

    return samples, is_accepted


#########
@tf.function
def hmc_step(q):
    p = tf.random.normal(shape=q.shape)
    q1, p1 = leapfrog(Nleapfrog, q, p, step_size, V, KE)
    q, p, accepted = metropolis([q, p], [q1, p1], H)
    return q


def cond(i, q, samples):
    return tf.less(i, tf.constant(nsamples+burnin))

    
@tf.function
def hmc_body(q):
    #q = tf.vectorized_map(hmc_step, q)
    q = tf.map_fn(hmc_step, q)
    return q
#

def do_hmc():
    
    samples = tf.TensorArray(dtype=tf.float32, size=nsamples+burnin)
    i = tf.constant(0)

    q = initstate
    start = time.time()
    q = hmc_body(q)
    print('Time to set up : ', time.time() - start)
    print('Starting')
    start = time.time()    

    q = initstate
    for i in range(nsamples+burnin):
        q = hmc_body(q)
        samples = samples.write(i, q)

    end = time.time()

    print("Time taken : ", end - start)

    mysamples = samples.stack().numpy()[burnin:]
    print(mysamples.shape)
    np.save(fpath + '/samples', mysamples)
#
    dg.plot_hist(mysamples, fpath)
    dg.plot_trace(mysamples, fpath)
    dg.plot_scatter(mysamples, fpath)
    dg.plot_autorcc(mysamples, fpath)
#    
if __name__=="__main__":
    do_hmc()
