import numpy as np
import os, sys
from scipy import signal
from scipy.stats import kstest
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import matplotlib
import arviz as az
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

from posteriordb import PosteriorDatabase
import os
pdb_path = os.path.join('../../posteriordb/posterior_database/')
my_pdb = PosteriorDatabase(pdb_path)
pos = my_pdb.posterior_names()
mn = my_pdb.model_names()
dn = my_pdb.dataset_names()


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--posnum',  default=0, type=int, help='Dimensions')
parser.add_argument('--nbins',  default=20, type=int, help='# Bins')
parser.add_argument('--subsample',  default=1, type=int, help='Subsample')
parser.add_argument('--lpath',  default=10, type=float, help='Nleapfrog*step_size')
parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')


args = parser.parse_args()
posnum = args.posnum
lpath = args.lpath
nbins = args.nbins
suffix = args.suffix


posname = pos[posnum]
print(posname)
fdir = '/mnt/ceph/users/cmodi/hmc/outputs_long/%s-def/'%(posname)
posterior = my_pdb.posterior(posname)
names = posterior.reference_draws_info()['diagnostics']['diagnostic_information']['names']
print(names)
refdrawsdict = posterior.reference_draws()
refdraws = []
for i in range(len(refdrawsdict)):
    refdraws.append(np.array([refdrawsdict[i][key] for key in refdrawsdict[i].keys()]).T)
refdraws = np.stack(refdraws, 1)
refdraws2d = refdraws.reshape(-1, refdraws.shape[-1])
refsamples = refdraws2d
ndim = len(names)
print(ndim)
nchains = 50
nsamples = 100000#samples[fidkey].shape[0]
if rank == 0:
    with open('/mnt/ceph/users/cmodi/hmc/outputs_long/%s-def/params.json'%posname) as ff:
        for ll in ff.readlines(): print(ll)

with open(fdir + '/params.json', 'r') as fp:
    fidparams = json.load(fp)
stepsizefid = comm.bcast(fidparams['stepsize'], root=0)
invmetricfid = comm.bcast(np.array(fidparams['invmetric']), root=0)
Tint = comm.bcast(fidparams['Tintegration'], root=0)
Nleapfrogfid = comm.bcast(fidparams['Nleapfrog'], root=0)
if Nleapfrogfid < 10: sys.exit()

########################################## 
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
ess, ess2 = {}, {}
errchain, errchain2 = {}, {}
essaz, rhat = {}, {}
evals = {}
clsize = {}
ssize = {}
ks = {}
kp = {}

def get_ess(x, xref):
       
    err = x.mean(axis=0) - xref.mean(axis=0)
    rms = (err**2).mean(axis=0)**0.5
    ess = (xref.std(axis=0)/rms)**2
    errchain = (xref.std(axis=0)/err)**2
    x2, xref2 = x**2, xref**2
    err2 = x2.mean(axis=0) - xref2.mean(axis=0)
    rms2 = (err2**2).mean(axis=0)**0.5
    ess2 = (xref2.std(axis=0)/rms2)**2
    errchain2 = (xref2.std(axis=0)/err2)**2
    return ess*x.shape[1], ess2*x.shape[1], err, err2


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
#fpathold = '/mnt/ceph/users/cmodi/hmc/outputs_long/%s-def//step10/'%posname
fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long/%s-def//step10/'%posname
#fpath1 = '/mnt/ceph/users/cmodi/hmc/outputs_long/%s-def//step10/'%posname
#else: fpath = None
fnames = ['nleapfid_vanilla', 'tintfid_vanilla', 'nleapfid_r9_normal', 'nleapfid_r9']
fpaths = []
for fname in fnames:
    fpath = '/mnt/ceph/users/cmodi/hmc/outputs_adaptive/%s//'%posname + fname + '/'
    if os.path.isfile(fpath + '/samples.npy'): 
        f = np.load(fpath + "/samples.npy")
        if f.shape[0] == 20000: 
            print(f.shape)
        else:
            raise Exception('size did not match')
        fpaths.append(fpath)
    else:
        try:
            combine(fpath)
            f = np.load(fpath + "/samples.npy")
            fpaths.append(fpath)
        except Exception as e:
            print(rank, e)

fpaths = [fpath0] + fpaths
print(fpaths)

for fpath in fpaths:
    try: 
    #if os.path.isfile(fpath + '/samples.npy'): pass
        #elif os.path.isfile(fpath + '/samples/00.npy'): 
        #    combine(fpath)
        ranksamples = np.load(fpath + '/samples.npy')[::args.subsample]
        print(rank, ranksamples.shape)
        rankcsamples = clean_samples(ranksamples)[0]
        rankclshape = rankcsamples.shape[0]
        ranksize = ranksamples.shape[0]*ranksamples.shape[1]
        readkey = fpath.split('/')[-2]
        key = readkey
        print(rank, fpath, readkey)
        rankevals = np.load(fpath + '/counts.npy')
        nevals = rankevals.sum(axis=(0, 1))[:2].sum()
        #except Exception as e: 
        #    print(readkey, e)
        #    if readkey == 'step001_nleap1000': nevals = ranksamples.shape[0]*ranksamples.shape[1]*(1000+2)

        rankcdf, rankbincounts = get_cdf(rankcsamples, xref=refdraws2d, nbins=args.nbins, xmin=-1e10, xmax=1e10)
        rankks, rankkp = get_ks(rankcsamples, refdraws2d)
        rankess, rankess2, rankerrchain, rankerrchain2 = get_ess(ranksamples, refdraws2d)
        rankessaz = [az.ess(ranksamples[..., i].T, relative=False) for i in range(ndim)]
        rankrhat = [az.rhat(ranksamples[..., i].T) for i in range(ndim)]

        evals[readkey] = [nevals*1.0]
        bincounts[readkey] = rankbincounts
        clsize[readkey] = [rankclshape*1.0]
        ssize[readkey] = [ranksize*1.0]
        ess[readkey] = list(rankess)
        errchain[readkey] = rankerrchain.tolist()
        ess2[readkey] = list(rankess2)
        errchain2[readkey] = rankerrchain2.tolist()
        ks[readkey] = rankks
        kp[readkey] = rankkp
        essaz[readkey] = rankessaz
        rhat[readkey] = rankrhat

    except Exception as e:  
        readkey = None
        ranksamples, nevals, rankbincounts, rankess, rankclshape, rankerrchain, rankkp, rankks = None, None, None, None, None, None, None, None
        rankessaz, rankrhat = None, None
        print("Exception in rank %d with key %s"%(rank, readkey), e)            





print("Dump")
print(essaz)
try: os.makedirs('./data2/%s-def/nbins%d/'%(posname, nbins))
except Exception as e: print(e)
try: os.makedirs('./data2/%s-def/nbins%d/adaptive/'%(posname, nbins))
except Exception as e: print(e)

print('saving in data2/%s-def/nbins%d//adaptive/evals.json'%(posname, nbins,   ))
with open('./data2/%s-def/nbins%d//adaptive/evals.json'%(posname, nbins,   ), 'w') as fp:
    json.dump(evals, fp, sort_keys=True, indent=4)
print('evals saved')
with open('./data2/%s-def/nbins%d//adaptive/clsize.json'%(posname, nbins,   ), 'w') as fp:
    json.dump(clsize, fp, sort_keys=True, indent=4)
print('clsize saved')
with open('./data2/%s-def/nbins%d//adaptive/ssize.json'%(posname, nbins,   ), 'w') as fp:
    json.dump(ssize, fp, sort_keys=True, indent=4)
with open('./data2/%s-def/nbins%d//adaptive/ess.json'%(posname, nbins,   ), 'w') as fp:
    json.dump(ess, fp, sort_keys=True, indent=4)
with open('./data2/%s-def/nbins%d//adaptive/ess2.json'%(posname, nbins,   ), 'w') as fp:
    json.dump(ess2, fp, sort_keys=True, indent=4)
with open('./data2/%s-def/nbins%d//adaptive/ksstat.json'%(posname, nbins,   ), 'w') as fp:
    json.dump(ks, fp, sort_keys=True, indent=4)
with open('./data2/%s-def/nbins%d//adaptive/kspval.json'%(posname, nbins,   ), 'w') as fp:
    json.dump(kp, fp, sort_keys=True, indent=4)
with open('./data2/%s-def/nbins%d//adaptive/essaz.json'%(posname, nbins,   ), 'w') as fp:
    json.dump(essaz, fp, sort_keys=True, indent=4)
with open('./data2/%s-def/nbins%d//adaptive/rhat.json'%(posname, nbins,   ), 'w') as fp:
    json.dump(rhat, fp, sort_keys=True, indent=4)
print('ess saved')


##    for ic in range(ndim):
##        todump = {}
##        for key in bincounts.keys(): 
##            todump[key] = list(bincounts[key][ic])
##            todump[key] = bincounts[key][ic].tolist()
##        fpname = './data2/%s-def/nbins%d/adaptive/bincounts_%s.json'%(posname, nbins,  names[ic])
##        with open(fpname, 'w') as fp:
##            json.dump(todump, fp, sort_keys=True, indent=4)
##
##        todump = {}
##        for key in errchain.keys(): 
##            #todump[key] = list(errchain[key][:, ic])
##            todump[key] = errchain[key][:, ic].tolist()
##        fpname = './data2/%s-def/nbins%d/adaptive/errchain_%s.json'%(posname, nbins,  names[ic])
##        with open(fpname, 'w') as fp:
##            json.dump(todump, fp, sort_keys=True, indent=4)
##
##        todump = {}
##        for key in errchain2.keys(): 
##            #todump[key] = list(errchain[key][:, ic])
##            todump[key] = errchain2[key][:, ic].tolist()
##        fpname = './data2/%s-def/nbins%d/adaptive/errchain2_%s.json'%(posname, nbins, names[ic])
##        with open(fpname, 'w') as fp:
##            json.dump(todump, fp, sort_keys=True, indent=4)
##
##        print(errchain[key].shape)
