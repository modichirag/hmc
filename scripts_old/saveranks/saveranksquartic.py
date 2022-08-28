import numpy as np
import os, sys
from scipy import signal
from scipy.stats import kstest
from scipy.stats import ks_2samp
import arviz as az
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
wsize = comm.Get_size()
from utils import get_cdf, clean_samples

import json
import argparse


stepsf = [1, 2, 5, 0.5]
#stepsf = [0.5]
nsubs = [2, 3, 4]
factors = [2, 5,10]
adel = 99
olong = 1
suffix = 'adelta99-hmctnuts'


def bsess(s):
    ess0, ess1 = [], []
    for i in range(20):
        ii = np.random.randint(0, 50, 50)
        ess0.append(az.ess(s[:, ii, 0].T))
        ess1.append(az.ess(s[:, ii, 1].T))
    ess0 = np.array(ess0)
    ess1 = np.array(ess1)
    return ess0.std(), ess1.std()



ess, nevals = {}, {}


q0 = np.concatenate([np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/quartic_N10-%s//default/samples/%d.npy'%(suffix, i))[..., :2] for i in range(50)], axis=1)
n0 = np.concatenate([np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/quartic_N10-%s//default/metrics/nleaprfrog%02d.npy'%(suffix, i))[q0.shape[0]:].reshape(-1, 1) for i in range(50)], axis=1)
ess00 = az.ess(q0[..., 0].T, relative=False)
ess01 = az.ess(q0[..., 1].T, relative=False)
err0, err1 = bsess(q0)
cost00, cost01 = n0.sum()/ess00, n0.sum()/ess01
keyy = 'nuts'
ess[keyy] = [ess00, ess01, err0, err1]
nevals[keyy] = np.float64(n0.sum())

q0 = np.concatenate([np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/quartic_N10-%s//default/samples/hmc%d.npy'%(suffix, i))[..., :2] for i in range(50)], axis=1)
n0 = np.concatenate([np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/quartic_N10-%s//default/metrics/nleaprfroghmc%02d.npy'%(suffix, i))[q0.shape[0]:].reshape(-1, 1) for i in range(50)], axis=1)
ess00 = az.ess(q0[..., 0].T, relative=False)
ess01 = az.ess(q0[..., 1].T, relative=False)
err0, err1 = bsess(q0)
cost00, cost01 = n0.sum()/ess00, n0.sum()/ess01
keyy = 'hmc'
ess[keyy] = [ess00, ess01, err0, err1]
nevals[keyy] = np.float64(n0.sum())



for ss in stepsf:
    try:
        if olong == 1:
            s = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/quartic_N10-%s/step%02d//samples.npy'%(suffix, ss*10,))
            c = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/quartic_N10-%s/step%02d//counts.npy'%(suffix, ss*10,))
        if olong == 3:
            s = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long3/quartic_N10-%s/step%02d//samples.npy'%(suffix, ss*10,))
            c = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long3/quartic_N10-%s/step%02d//counts.npy'%(suffix, ss*10,))
        ess00 = az.ess(s[..., 0].T, relative=False)
        ess01 = az.ess(s[..., 1].T, relative=False)
        cost0, cost1 = c[..., :2].sum()/ess00, c[..., :2].sum()/ess01
        err0, err1 = bsess(s)
        cost0err, cost1err = c[..., :2].sum()*err0/ess00**2, c[..., :2].sum()*err1/ess01**2
        keyy = 'step%02d'%(ss*10)
        ess[keyy] = [ess00, ess01, err0, err1]
        nevals[keyy] = np.float64(c[..., :2].sum())
    except: pass

    for iff, ff in enumerate(factors):
        for insub, nsub in enumerate(nsubs):
            keyy = 'step%02d//%d-%d'%(ss*10, ff, nsub)
            print(keyy)
            try:
                if olong == 1:
                    s = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/quartic_N10-%s/step%02d_fac%02d_nsub%d//samples.npy'%(suffix, ss*10, ff, nsub))
                    c = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/quartic_N10-%s/step%02d_fac%02d_nsub%d//counts.npy'%(suffix, ss*10, ff, nsub))
                if olong == 3:
                    s = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long3/quartic_N10-%s/step%02d_fac%02d_nsub%d//samples.npy'%(suffix, ss*10, ff, nsub))
                    c = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long3/quartic_N10-%s/step%02d_fac%02d_nsub%d//counts.npy'%(suffix, ss*10, ff, nsub))

                ess00 = az.ess(s[..., 0].T, relative=False)
                ess01 = az.ess(s[..., 1].T, relative=False)
                cost0, cost1 = c[..., :2].sum()/ess00, c[..., :2].sum()/ess01
                err0, err1 = bsess(s)
                ess[keyy] = [ess00, ess01, err0, err1]
                nevals[keyy] = np.float64(c[..., :2].sum())
                print(nevals)
            except: pass
print(nevals)            
                
                
fpname = './data2/%s/nbins%d/olong%d/ess%s.json'%('quartic',  20, olong, suffix)
with open(fpname, 'w') as fp:
    json.dump(ess, fp, sort_keys=True, indent=4)

fpname = './data2/%s/nbins%d/olong%d/evals%s.json'%('quartic',  20, olong, suffix)
with open(fpname, 'w') as fp:
    json.dump(nevals, fp, sort_keys=True, indent=4)
    
