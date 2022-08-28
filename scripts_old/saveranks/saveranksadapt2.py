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

fquantiles = np.stack([sigquantile] + [alpquantile]*50)
refsamples  = np.stack([sigmarep] + [alpharep]*50).T


########################################## 
steps = [0.5, 0.2, 0.1, 0.05, 0.04, 0.02,  0.01]
#steps = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
#steps = [0.04, 0.02, 0.01]#, 1.0]
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
    for ftype in ['samples', 'counts', 'accepted', 'diags']:
        ss = []
        for i in range(50):
            ss.append(np.load(fpath + '/%s/%02d.npy'%(ftype, i)))
        ss = np.squeeze(np.array(ss))
        if len(ss.shape) == 3: 
            ss = np.transpose(ss, (1, 0, 2))
        else: ss = np.transpose(ss, (1, 0))
        print(ftype, ss.shape)
        np.save(fpath + '/%s.npy'%ftype, ss)
        #else:  np.save(fpath + '/%s.npy'%'accepted', ss)





ratio = 0.9
for ndim in [20]:    
    fpathold = '/mnt/ceph/users/cmodi/hmc/outputs_long/funnel//Ndim%02d/'%ndim
    fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long/funnel//Ndim%02d/step001_nleap1000/'%ndim
    fpath1 = '/mnt/ceph/users/cmodi/hmc/outputs_long/funnel//Ndim%02d/step010_nleap100_fac10_nsub2/'%ndim
    #else: fpath = None
    fpaths = []
    for mode in ['tint', 'nleap']:
        for nl in [10, 20, 50, 100]:
            for suff in ['_r9', '_r9_normal', '_vanilla']:
                fname = mode + '%d'%nl + suff
                fpath = '/mnt/ceph/users/cmodi/hmc/outputs_adaptive/funnel//Ndim%02d/'%ndim + fname + '/'
                if os.path.isfile(fpath + '/samples.npy'): 
                    f = np.load(fpath + "/samples.npy")
                    if f.shape[0] == 20000: print(f.shape)
                    fpaths.append(fpath)
                    #raise Exception('idk sth weird is happening so combine again')
                else:
                    try:
                        combine(fpath)
                        f = np.load(fpath + "/samples.npy")
                        fpaths.append(fpath)
                    except Exception as e:
                        print(rank, e)
    
    fpaths = [fpath0, fpath1] + fpaths
    print(fpaths)

    for fpath in fpaths:
        #try: 
        #if os.path.isfile(fpath + '/samples.npy'): pass
            #elif os.path.isfile(fpath + '/samples/00.npy'): 
            #    combine(fpath)
            ranksamples = np.load(fpath + '/samples.npy')[::args.subsample]
            print(rank, ranksamples.shape)
            rankcsamples = clean_samples(ranksamples)[0]
            rankclshape = rankcsamples.shape[0]
            readkey = fpath.split('/')[-2]
            key = readkey
            print(rank, fpath, readkey, rankcsamples.shape)
            try: 
                rankevals = np.load(fpath + '/counts.npy')
                nevals = rankevals.sum(axis=(0, 1))[:2].sum()
            except Exception as e: 
                print(readkey, e)
                if readkey == 'step001_nleap1000': nevals = ranksamples.shape[0]*ranksamples.shape[1]*(1000+2)
            
            print(fquantiles.shape)    
            rankcdf, rankbincounts = get_cdf(rankcsamples, quantiles=fquantiles)
            rankks, rankkp = get_ks(rankcsamples, refsamples)
            esss0, esss0chain = get_ess(ranksamples[..., 0], smean, sstd)
            esss2, esss2chain = get_ess(ranksamples[..., 0]**2, s2mean, s2std)
            essa0 = np.array([get_ess(ranksamples[..., j], amean, astd)[0] for j in range(1, ndim)]).mean()
            essa2 = np.array([get_ess(ranksamples[..., j]**2, a2mean, a2std)[0] for j in range(1, ndim)]).mean()
            essa0chain = np.array([get_ess(ranksamples[..., j], amean, astd)[1] for j in range(1, ndim)]).mean(axis=0)
            essa2chain = np.array([get_ess(ranksamples[..., j]**2, a2mean, a2std)[1] for j in range(1, ndim)]).mean(axis=0)
            rankess = np.array([esss0, esss2, essa0, essa2]) / ranksamples.shape[0]
            rankesschain = np.stack([esss0chain, esss2chain, essa0chain, essa2chain]) / ranksamples.shape[0]

            evals[readkey] = [nevals*1.0]
            bincounts[readkey] = rankbincounts
            clsize[readkey] = [rankclshape*1.0]
            ess[readkey] = list(rankess)
            esschain[readkey] = rankesschain.tolist()
            ks[readkey] = rankks
            kp[readkey] = rankkp

        #except Exception as e:  
        #    readkey = None
        #    ranksamples, nevals, rankbincounts, rankess, rankclshape, rankesschain, rankkp, rankks = None, None, None, None, None, None, None, None
        #    print("Exception in rank %d with key %s"%(rank, readkey), e)            


    

    
    print("Dump")
    print(evals)
    try: os.makedirs('./data2/funnel/nbins%d/'%(nbins))
    except Exception as e: print(e)
    try: os.makedirs('./data2/funnel/nbins%d/adaptive/'%(nbins))
    except Exception as e: print(e)
    try: os.makedirs('./data2/funnel/nbins%d/adaptive/Ndim%02d/'%(nbins,  ndim))
    except Exception as e: print(e)

    print('saving in data2/funnel/nbins%d//adaptive/Ndim%02d/l%02d_evals.json'%(nbins,  ndim, lpath))
    with open('./data2/funnel/nbins%d//adaptive/Ndim%02d/l%02d_evals.json'%(nbins,  ndim, lpath), 'w') as fp:
        json.dump(evals, fp, sort_keys=True, indent=4)
    print('evals saved')
    with open('./data2/funnel/nbins%d//adaptive/Ndim%02d/l%02d_clsize.json'%(nbins,  ndim, lpath), 'w') as fp:
        json.dump(clsize, fp, sort_keys=True, indent=4)
    print('clsize saved')
    with open('./data2/funnel/nbins%d//adaptive/Ndim%02d/l%02d_ess.json'%(nbins,  ndim, lpath), 'w') as fp:
        json.dump(ess, fp, sort_keys=True, indent=4)
    with open('./data2/funnel/nbins%d//adaptive/Ndim%02d/l%02d_ksstat.json'%(nbins,  ndim, lpath), 'w') as fp:
        json.dump(ks, fp, sort_keys=True, indent=4)
    with open('./data2/funnel/nbins%d//adaptive/Ndim%02d/l%02d_kspval.json'%(nbins,  ndim, lpath), 'w') as fp:
        json.dump(kp, fp, sort_keys=True, indent=4)
    with open('./data2/funnel/nbins%d//adaptive/Ndim%02d/l%02d_esschain.json'%(nbins,  ndim, lpath), 'w') as fp:
        json.dump(esschain, fp, sort_keys=True, indent=4)
    print('ess saved')
    for ic in range(ndim):
        todump = {}
        for key in bincounts.keys(): 
            todump[key] = list(bincounts[key][ic])
            todump[key] = bincounts[key][ic].tolist()
        if ic == 0: 
            fpname = './data2/funnel/nbins%d/adaptive/Ndim%02d/l%02d_bincounts_sigma.json'%(nbins,  ndim, lpath)
        else: fpname = './data2/funnel/nbins%d/adaptive/Ndim%02d/l%02d_bincounts_alpha%d.json'%(nbins,  ndim, lpath, ic)
        with open(fpname, 'w') as fp:
            json.dump(todump, fp, sort_keys=True, indent=4)


