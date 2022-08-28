import numpy as np
import os, sys
from scipy import signal
from scipy.stats import kstest
from scipy.stats import ks_2samp
import arviz as az

from utils import get_cdf, clean_samples

import json
import argparse


stepsf = [1, 2, 5, 0.5]
nsubs = [2, 3, 4]
factors = [2, 5,10]
adel = 99
olong = 1
#stepsf = [1]
#nsubs = [5]
#factors = [2]
nbs = 50

mu, std = 2.1000915, 2.1680835703048293
refsamples = np.concatenate([np.random.normal(3, 1, 100000), np.random.normal(0, 0.1, 100000)])
mu, std = refsamples.mean(), refsamples.std()
xref = refsamples.flatten()

def get_ess(x, nbs=nbs):
      
    err = (x.mean(axis=0) - xref.mean()).flatten()
    rms = (err**2).mean(axis=0)**0.5
    ess = (xref.std(axis=0)/rms)**2
    esser = []
    for i in range(nbs):
        ii = np.random.randint(0, 50, 50)
        errbs = err[ii].flatten()
        rms = (errbs**2).mean(axis=0)**0.5
        esser.append((xref.std(axis=0)/rms)**2)
    esser = np.array(esser).std()

    x2, xref2 = x**2, xref**2
    err2 = (x2.mean(axis=0) - xref2.mean()).flatten()
    rms2 = (err2**2).mean(axis=0)**0.5
    ess2 = (xref2.std(axis=0)/rms2)**2
    esser2 = []
    for i in range(nbs):
        ii = np.random.randint(0, 50, 50)
        errbs = err2[ii].flatten()
        rms2 = (errbs**2).mean(axis=0)**0.5
        esser2.append((xref2.std(axis=0)/rms2)**2)
    esser2 = np.array(esser2).std()
    return ess*x.shape[1], ess2*x.shape[1], esser, esser2


def bsess(s, nbs=nbs):
    ess0 = []
    for i in range(nbs):
        ii = np.random.randint(0, 50, 50)
        ess0.append(az.ess(s[:, ii, 0].T))
    ess0 = np.array(ess0)
    ess2 = []
    for i in range(20):
        ii = np.random.randint(0, 50, 50)
        ess2.append(az.ess((s[:, ii, 0]**2).T))
    ess2 = np.array(ess2)
    return ess0.std(), ess2.std()


ess, nevals = {}, {}
essaz = {}
err = {}

posname = 'gaussmixs0p1s1-hmc'
print('loading nuts')
#q0 = np.concatenate([np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s//default/samples/%d.npy'%(posname, i))[..., :2] for i in range(50)], axis=1)
q0 = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s//default/samples.npy'%(posname))
n0 = np.concatenate([np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s//default/metrics/nleaprfrog%02d.npy'%(posname, i))[q0.shape[0]:].reshape(-1, 1) for i in range(50)], axis=1)
keyy = 'nuts'
ess00, ess02  =  az.ess(q0[..., 0].T, relative=False), az.ess((q0[..., 0]**2).T, relative=False)
err0, err2 = bsess(q0)
essaz[keyy] = [ess00, ess02, err0, err2]
ess00, ess02, err0, err2 = get_ess(q0)
ess[keyy] = [ess00, ess02, err0, err2] #np.array([ess00, ess02, err0, err2]).flatten().tolist()
print(ess)
nevals[keyy] = np.float64(n0.sum())

print('loading hmc')
#q0 = np.concatenate([np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s//default/samples/hmc%d.npy'%(posname, i))[..., :2] for i in range(50)], axis=1)
q0 = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s//default/sampleshmc.npy'%(posname))
n0 = np.concatenate([np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s//default/metrics/nleaprfroghmc%02d.npy'%(posname, i))[q0.shape[0]:].reshape(-1, 1) for i in range(50)], axis=1)
keyy = 'hmc'
ess00, ess02  =  az.ess(q0[..., 0].T, relative=False), az.ess((q0[..., 0]**2).T, relative=False)
err0, err2 = bsess(q0)
essaz[keyy] = [ess00, ess02, err0, err2]
ess00, ess02, err0, err2 = get_ess(q0)
ess[keyy] = [ess00, ess02, err0, err2]
#err[keyy] = [err0, err2]
nevals[keyy] = np.float64(n0.sum())

print(ess)

for ss in stepsf:
    try:
        if olong == 1:
            s = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s/step%02d//samples.npy'%(posname,  ss*10,))
            c = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s/step%02d//counts.npy'%(posname,  ss*10,))
        if olong == 3:
            s = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long3/%s/step%02d//samples.npy'%(posname,  ss*10,))
            c = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long3/%s/step%02d//counts.npy'%(posname,  ss*10,))
        keyy = 'step%02d'%(ss*10)
        nevals[keyy] = np.float64(c[..., :2].sum())
        ess00 = az.ess(s[..., 0].T, relative=False)
        ess02 = az.ess((s[..., 0]**2).T, relative=False)
        err0, err2 = bsess(s)
        essaz[keyy] = [ess00, ess02, err0, err2]
        ess00, ess02, err0, err2 = get_ess(s)
        ess[keyy] = [ess00, ess02, err0, err2]
        
    except Exception as e: 
        print(e)

    for iff, ff in enumerate(factors):
        for insub, nsub in enumerate(nsubs):
            keyy = 'step%02d//%d-%d'%(ss*10, ff, nsub)
            print(keyy)
            try:
                if olong == 1:
                    s = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s/step%02d_fac%02d_nsub%d//samples.npy'%(posname,  ss*10, ff, nsub))
                    c = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long/%s/step%02d_fac%02d_nsub%d//counts.npy'%(posname,  ss*10, ff, nsub))
                if olong == 3:
                    s = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long3/%s/step%02d_fac%02d_nsub%d//samples.npy'%(posname,  ss*10, ff, nsub))
                    c = np.load('/mnt/ceph/users/cmodi/hmc/outputs_long3/%s/step%02d_fac%02d_nsub%d//counts.npy'%(posname,  ss*10, ff, nsub))

                nevals[keyy] = np.float64(c[..., :2].sum())
                ess00 = az.ess(s[..., 0].T, relative=False)
                ess02 = az.ess((s[..., 0]**2).T, relative=False)
                err0, err2 = bsess(s)
                essaz[keyy] = [ess00, ess02, err0, err2]
                ess00, ess02, err0, err2 = get_ess(s)
                ess[keyy] = [ess00, ess02, err0, err2]
                
            except Exception as e: 
                print(e)


                
fpname = './data2/%s/nbins%d/olong%d/ess.json'%(posname,  20, olong)
print('saving in ', fpname)
with open(fpname, 'w') as fp:
    json.dump(ess, fp, sort_keys=True, indent=4)

fpname = './data2/%s/nbins%d/olong%d/essaz.json'%(posname,  20, olong)
print('saving in ', fpname)
with open(fpname, 'w') as fp:
    json.dump(essaz, fp, sort_keys=True, indent=4)

fpname = './data2/%s/nbins%d/olong%d/err.json'%(posname,  20, olong)
print('saving in ', fpname)
with open(fpname, 'w') as fp:
    json.dump(err, fp, sort_keys=True, indent=4)

fpname = './data2/%s/nbins%d/olong%d/evals.json'%(posname,  20, olong)
print('saving in ', fpname)
with open(fpname, 'w') as fp:
    json.dump(nevals, fp, sort_keys=True, indent=4)
    
