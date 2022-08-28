import numpy as np
import os, sys
from scipy import signal
from scipy.stats import kstest
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import matplotlib.ticker as mticker
f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%.2e' % x))
fmt = mticker.FuncFormatter(g)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
wsize = comm.Get_size()
from utils import get_cdf, clean_samples

import json
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ndim',  default=5, type=int, help='Dimensions')
parser.add_argument('--nbins',  default=20, type=int, help='# Bins')
parser.add_argument('--subsample',  default=1, type=int, help='Subsample')
parser.add_argument('--lpath',  default=10, type=float, help='Nleapfrog*step_size')
parser.add_argument('--olong',  default=3, type=float, help='Nleapfrog*step_size')
parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
ndim = args.ndim
lpath = args.lpath
nbins = args.nbins
suffix = args.suffix
olong = args.olong

sigmarep = np.random.normal(0,3,10000000)
yvrep = np.random.normal(0,1,10000000)
alpharep = np.exp(sigmarep/2.)*yvrep
sigquantile = np.quantile(sigmarep, np.linspace(0, 1, nbins+1))
alpquantile = np.quantile(alpharep, np.linspace(0, 1, nbins+1))

smean, sstd = 0, 3.
amean, astd = 0, alpharep.std()
s2mean, s2std = (sigmarep**2).mean(), (sigmarep**2).std()
a2mean, a2std = (alpharep**2).mean(), (alpharep**2).std()
sigquantile[0] = -10000
sigquantile[-1] = 10000
alpquantile[0] = -10000
alpquantile[-1] = 10000

fquantiles = np.stack([sigquantile] + [alpquantile]*ndim)
refsamples  = np.stack([sigmarep] + [alpharep]*ndim).T


########################################## 
steps = [0.5, 0.2, 0.1, 0.05, 0.04, 0.02,  0.01]
#steps = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
#steps = [0.2]#, 1.0]
facs = [2, 5, 10]
subs = [2, 3, 4, 5, 6]
nsteps = len(steps)
nfacs = len(facs)
nsubs = len(subs)
samples = {}
clsamples = {}
acc = {}
probs = {}
vcounts = {}

keystoplot = []
alldiff = 1000
ranksamples, rankacc, rankprobs, rankcounts = None, None, None, None
readkey = None
samplekeys, samplevals = [], []    
bincounts = {}
ess = {}
esschain = {}
evals = {}
clsize = {}
ks = {}
kp = {}

def get_ess(x, mean, std):
    err = x.mean(axis=0) - mean
    rms = (err**2).mean()**0.5
    ess = (std/rms)**2
    esschain = (std/err)**2
    return ess, esschain


def get_ks(x, xref):
    #Get KS statistics for the total samples and the tails
    mu, std = xref.mean(axis=0), xref.std(axis=0)
    #print("means and std from reference : ", mu, std)
    stats, pvals = [],[]
    for i in range(x.shape[1]):
        ss, pp = [], []
        ks = ks_2samp(x[:, i], xref[:, i])
        ss.append(ks.statistic)
        pp.append(ks.pvalue)
        for j in [-2, -1, 1, 2]:
            tt = mu[i] + j*std[i]
            a, b = x[:, i].copy(), xref[:, i].copy()
            if j < 0: 
                a, b = a[a<tt], b[b<tt]
            else: 
                a, b = a[a>tt], b[b>tt] 
            ks = ks_2samp(a, b)
            ss.append(ks.statistic)
            pp.append(ks.pvalue)
        stats.append(ss)
        pvals.append(pp)
    return stats, pvals



def combine(fpath):
    print("Combining")
    for ftype in ['samples', 'counts', 'accepts']:
        ss = []
        for i in range(50):
            ss.append(np.load(fpath + '/%s/%02d.npy'%(ftype, i)))
        ss = np.squeeze(np.array(ss))
        if len(ss.shape) == 3: 
            ss = np.transpose(ss, (1, 0, 2))
        else: ss = np.transpose(ss, (1, 0))
        print(ftype, ss.shape)
        if ftype != 'accepts': np.save(fpath + '/%s.npy'%ftype, ss)
        else:  np.save(fpath + '/%s.npy'%'accepted', ss)



for istep in range(nsteps):    
    counter = 1
    key, fpath = None, None
    comm.Barrier()
    Nleapfrog = int(lpath / steps[istep])
    Nleapfrog = max(10, Nleapfrog)
    if olong == 1: fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long/funnel//Ndim%02d/'%ndim
    else: fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long%d/funnel//Ndim%02d/'%(olong, ndim)
    fpathold = '/mnt/ceph/users/cmodi/hmc/outputs_long/funnel//Ndim%02d/'%ndim
    if rank == 0 :
        fpath = fpathold + 'step%03d_nleap%02d/'%(steps[istep]*100, Nleapfrog)
        key = 'step %0.3f'%(steps[istep])
    
    for fac in facs:            
        for isub, sub in enumerate(subs):
            if ((counter) %wsize  == rank) & (fpath is None):
                fpath = fpath0 + 'step%03d_nleap%02d_fac%02d_nsub%d/'%(steps[istep]*100, Nleapfrog, fac, sub)
                key = 'step %0.3f//%d-%d'%(steps[istep],  fac, sub);            
            counter += 1
    print(steps[istep], rank, key, fpath)            
    #Got key for every rank, now load file and analyze
    try: 
        if os.path.isfile(fpath + '/samples.npy'): pass
        elif os.path.isfile(fpath + '/samples/00.npy'): 
                combine(fpath)
        ranksamples = np.load(fpath + '/samples.npy')[::args.subsample]
        print(rank, ranksamples.shape)
        rankcsamples = clean_samples(ranksamples)[0]
        rankclshape = rankcsamples.shape[0]
        readkey = key
        if '//' in key: 
            rankevals = np.load(fpath + '/counts.npy')
            nevals = rankevals.sum(axis=(0, 1))[:2].sum()
        else: nevals = (Nleapfrog + 3)*50*1e5 #np.load(fpath + '/counts.npy')

        rankcdf, rankbincounts = get_cdf(rankcsamples, quantiles=fquantiles)
        rankks, rankkp = get_ks(rankcsamples, refsamples)
        print(key, rankks)
        esss0, esss0chain = get_ess(ranksamples[..., 0], smean, sstd)
        esss2, esss2chain = get_ess(ranksamples[..., 0]**2, s2mean, s2std)
        essa0 = np.array([get_ess(ranksamples[..., j], amean, astd)[0] for j in range(1, ndim)]).mean()
        essa2 = np.array([get_ess(ranksamples[..., j]**2, a2mean, a2std)[0] for j in range(1, ndim)]).mean()
        essa0chain = np.array([get_ess(ranksamples[..., j], amean, astd)[1] for j in range(1, ndim)]).mean(axis=0)
        essa2chain = np.array([get_ess(ranksamples[..., j]**2, a2mean, a2std)[1] for j in range(1, ndim)]).mean(axis=0)
        rankess = np.array([esss0, esss2, essa0, essa2]) * 50.
        rankesschain = np.stack([esss0chain, esss2chain, essa0chain, essa2chain])
        

        print(rank, key, rankess)
    except Exception as e:  
        readkey = None
        ranksamples, nevals, rankbincounts, rankess, rankclshape, rankesschain, rankkp, rankks = None, None, None, None, None, None, None, None
        print("Exception in rank %d with key %s"%(rank, readkey), e)            
    
    comm.Barrier()
    gkeys = comm.gather(readkey, root=0)
    print(gkeys)
    #gsamples = comm.gather(ranksamples, root=0)
    gnevals  = comm.gather(nevals, root=0)
    gbincounts = comm.gather(rankbincounts, root=0)
    gclsize = comm.gather(rankclshape, root=0)
    gess = comm.gather(rankess, root=0)
    gesschain = comm.gather(rankesschain, root=0)
    gks = comm.gather(rankks, root=0)
    gkp = comm.gather(rankkp, root=0)
    print(gkeys)

    if rank == 0:
        for ik in range(len(gkeys)):
            if gkeys[ik] is not None:
                print(gkeys[ik])
                #print("gathered array : ",  ik, samplekeys[ik], samplevals[ik].shape)
                #samples[gkeys[ik]] = gsamples[ik]
                evals[gkeys[ik]] = [gnevals[ik]*1.0]
                bincounts[gkeys[ik]] = gbincounts[ik]
                clsize[gkeys[ik]] = [gclsize[ik]*1.0]
                ess[gkeys[ik]] = list(gess[ik])
                esschain[gkeys[ik]] = gesschain[ik].tolist()
                ks[gkeys[ik]] = gks[ik]
                kp[gkeys[ik]] = gkp[ik]
    print("Move onto next step")
    comm.Barrier()


    
if rank == 0:

    
    print("Dump")
    print(evals)
    try: os.makedirs('./data2/funnel/nbins%d/'%(nbins))
    except Exception as e: print(e)
    try: os.makedirs('./data2/funnel/nbins%d/olong%d/'%(nbins, olong))
    except Exception as e: print(e)
    try: os.makedirs('./data2/funnel/nbins%d/olong%d/Ndim%02d/'%(nbins, olong, ndim))
    except Exception as e: print(e)

    print('saving in data2/funnel/nbins%d//olong%d/Ndim%02d/l%02d_evals.json'%(nbins, olong, ndim, lpath))
    with open('./data2/funnel/nbins%d//olong%d/Ndim%02d/l%02d_evals.json'%(nbins, olong, ndim, lpath), 'w') as fp:
        json.dump(evals, fp, sort_keys=True, indent=4)
    print('evals saved')
    with open('./data2/funnel/nbins%d//olong%d/Ndim%02d/l%02d_clsize.json'%(nbins, olong, ndim, lpath), 'w') as fp:
        json.dump(clsize, fp, sort_keys=True, indent=4)
    print('clsize saved')
    with open('./data2/funnel/nbins%d//olong%d/Ndim%02d/l%02d_ess.json'%(nbins, olong, ndim, lpath), 'w') as fp:
        json.dump(ess, fp, sort_keys=True, indent=4)
    with open('./data2/funnel/nbins%d//olong%d/Ndim%02d/l%02d_ksstat.json'%(nbins, olong, ndim, lpath), 'w') as fp:
        json.dump(ks, fp, sort_keys=True, indent=4)
    with open('./data2/funnel/nbins%d//olong%d/Ndim%02d/l%02d_kspval.json'%(nbins, olong, ndim, lpath), 'w') as fp:
        json.dump(kp, fp, sort_keys=True, indent=4)
    with open('./data2/funnel/nbins%d//olong%d/Ndim%02d/l%02d_esschain.json'%(nbins, olong, ndim, lpath), 'w') as fp:
        json.dump(esschain, fp, sort_keys=True, indent=4)
    print('ess saved')
    for ic in range(ndim):
        todump = {}
        for key in bincounts.keys(): 
            todump[key] = list(bincounts[key][ic])
            todump[key] = bincounts[key][ic].tolist()
        if ic == 0: 
            fpname = './data2/funnel/nbins%d/olong%d/Ndim%02d/l%02d_bincounts_sigma.json'%(nbins, olong, ndim, lpath)
        else: fpname = './data2/funnel/nbins%d/olong%d/Ndim%02d/l%02d_bincounts_alpha%d.json'%(nbins, olong, ndim, lpath, ic)
        with open(fpname, 'w') as fp:
            json.dump(todump, fp, sort_keys=True, indent=4)


