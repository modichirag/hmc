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
parser.add_argument('--olong',  default=1, type=float, help='Nleapfrog*step_size')
parser.add_argument('--suffix', default='', type=str,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
posnum = args.posnum
lpath = args.lpath
nbins = args.nbins
suffix = args.suffix
olong = args.olong


posname = pos[posnum]
print(posname)
if olong != 1:fdir = '/mnt/ceph/users/cmodi/hmc/outputs_long%d/%s-%s/'%(olong, posname, suffx)
else: fdir = '/mnt/ceph/users/cmodi/hmc/outputs_long/%s-%s/'%(posname, suffix)
print('reading from path : %s'%fdir)
posterior = my_pdb.posterior(posname)
names = posterior.reference_draws_info()['diagnostics']['diagnostic_information']['names']
print(names)
refdrawsdict = posterior.reference_draws()
refdraws = []
for i in range(len(refdrawsdict)):
    refdraws.append(np.array([refdrawsdict[i][key] for key in refdrawsdict[i].keys()]).T)
refdraws = np.stack(refdraws, 1)
refdraws2d = refdraws.reshape(-1, refdraws.shape[-1])
ndim = len(names)
print(ndim)
nchains = 50
nsamples = 100000#samples[fidkey].shape[0]
if rank == 0:
    with open('/mnt/ceph/users/cmodi/hmc/outputs_long/%s-def/params.json'%posname) as ff:
        for ll in ff.readlines(): print(ll)

with open('/mnt/ceph/users/cmodi/hmc/outputs_long/%s-def/params.json'%posname) as fp:
    fidparams = json.load(fp)
stepsizefid = comm.bcast(fidparams['stepsize'], root=0)
invmetricfid = comm.bcast(np.array(fidparams['invmetric']), root=0)
Tint = comm.bcast(fidparams['Tintegration'], root=0)
Nleapfrogfid = comm.bcast(fidparams['Nleapfrog'], root=0)



########################################## 
steps = [1, 2, 5, 0.5]
#steps = [1]

facs = [2, 5, 10]
subs = [2, 3, 4, 5]

nsteps = len(steps)
nfacs = len(facs)
nsubs = len(subs)
samples, accepts, vcounts = {}, {}, {}
clsamples = {}
acc = {}
probs = {}
vcounts = {}

keystoplot = []
ranksamples, rankacc, rankprobs, rankcounts = None, None, None, None
readkey = None
samplekeys, samplevals = [], []    
bincounts = {}
ess, ess2 = {}, {}
errchain, errchain2 = {}, {}
evals = {}
clsize = {}
ks = {}
kp = {}


def get_ess(x, xref):
       
    err = x.mean(axis=0) - xref.mean(axis=0)
    rms = (err**2).mean(axis=0)**0.5
    ess = (xref.std(axis=0)/rms)**2
    esschain = (xref.std(axis=0)/err)**2
    x2, xref2 = x**2, xref**2
    err2 = x2.mean(axis=0) - xref2.mean(axis=0)
    rms2 = (err2**2).mean(axis=0)**0.5
    ess2 = (xref2.std(axis=0)/rms2)**2
    esschain2 = (xref2.std(axis=0)/err2)**2
    return ess*x.shape[1], ess2*x.shape[1], err, err2


def get_ks(x, xref):
    print('shape in ks : ', x.shape, xref.shape)
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
            try: 
                ks = ks_2samp(a, b)
                ss.append(ks.statistic)
                pp.append(ks.pvalue)
            except: 
                ss.append(0)
                pp.append(1e-99)
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
    ss = steps[istep]
    key, fpath = None, None
    comm.Barrier()
    #Nleapfrog = int(lpath / steps[istep])
    #Nleapfrog = max(10, Nleapfrog)
    if rank == 0:
        key = 'step%02d'%(ss*10)
        fpath = fdir + key
    
    step_size, Nleapfrog =  stepsizefid*ss, max(5, int(Nleapfrogfid/ss))
    for fac in facs:            
        for isub, sub in enumerate(subs):
            if ((counter) %wsize  == rank) & (fpath is None):
                fpath = fdir + 'step%02d_fac%02d_nsub%d/'%(ss*10, fac, sub)
                key = 'step%02d//%d-%d'%(ss*10, fac, sub)
            else: pass
            counter += 1
    print(steps[istep], rank, key, fpath)            
    try: 
        #check if exists, else try to make by combine
        if os.path.isfile(fpath + '/samples.npy'): pass
        elif os.path.isfile(fpath + '/samples/00.npy'): 
                combine(fpath)
        #read
        ranksamples = np.load(fpath + '/samples.npy')[::args.subsample]
        print(rank, ranksamples.shape)
        rankcsamples = clean_samples(ranksamples)[0]
        rankclshape = rankcsamples.shape[0]
        readkey = key
        #if '//' in key: 
        try:
            rankevals = np.load(fpath + '/counts.npy')
            nevals = rankevals.sum(axis=(0, 1))[:2].sum()
        except:
            nevals = (Nleapfrog + 3)*50*1e5 #np.load(fpath + '/counts.npy')
        rankcdf, rankbincounts = get_cdf(rankcsamples, xref=refdraws2d, nbins=args.nbins, xmin=-1e10, xmax=1e10)
        rankks, rankkp = get_ks(rankcsamples, refdraws2d)
        rankess, rankess2, rankerrchain, rankerrchain2 = get_ess(ranksamples, refdraws2d)
        print(rank, key, rankess, rankess2)
        print(rank, rankerrchain.shape, rankerrchain2.shape)
    except Exception as e:  
        readkey = None
        ranksamples, nevals, rankbincounts, rankess, rankess2, rankclshape, rankerrchain, rankerrchain2 = None, None, None, None, None, None, None, None
        rankks, rankkp = None, None
        print("Exception in rank %d with key %s"%(rank, readkey), e)            
    
    comm.Barrier()
    gkeys = comm.gather(readkey, root=0)
    print(gkeys)
    #gsamples = comm.gather(ranksamples, root=0)
    gnevals  = comm.gather(nevals, root=0)
    gbincounts = comm.gather(rankbincounts, root=0)
    gclsize = comm.gather(rankclshape, root=0)
    gess = comm.gather(rankess, root=0)
    gess2 = comm.gather(rankess2, root=0)
    gerrchain = comm.gather(rankerrchain, root=0)
    gerrchain2 = comm.gather(rankerrchain2, root=0)
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
                ess2[gkeys[ik]] = list(gess2[ik])
                errchain[gkeys[ik]] = gerrchain[ik]
                errchain2[gkeys[ik]] = gerrchain2[ik]
                ks[gkeys[ik]] = gks[ik]
                kp[gkeys[ik]] = gkp[ik]
                    
    print("Move onto next step")
    comm.Barrier()

    
if rank == 0:

    posname = posname + '-' + suffix
    print("Dump")
    print(evals)
    try: os.makedirs('./data2/%s/nbins%d/'%(posname, nbins))
    except Exception as e: print(e)
    try: os.makedirs('./data2/%s/nbins%d/olong%d/'%(posname, nbins, olong))
    except Exception as e: print(e)


    with open('./data2/%s/nbins%d//olong%d/evals.json'%(posname, nbins, olong), 'w') as fp:
        json.dump(evals, fp, sort_keys=True, indent=4)
    print('evals saved')
    with open('./data2/%s/nbins%d//olong%d/clsize.json'%(posname, nbins, olong), 'w') as fp:
        json.dump(clsize, fp, sort_keys=True, indent=4)
    print('clsize saved')
    with open('./data2/%s/nbins%d//olong%d/ess.json'%(posname, nbins, olong), 'w') as fp:
        json.dump(ess, fp, sort_keys=True, indent=4)
    with open('./data2/%s/nbins%d//olong%d/ess2.json'%(posname, nbins, olong), 'w') as fp:
        json.dump(ess2, fp, sort_keys=True, indent=4)
    print('ess saved')
    with open('./data2/%s/nbins%d//olong%d/ksstat.json'%(posname, nbins, olong), 'w') as fp:
        json.dump(ks, fp, sort_keys=True, indent=4)
    print('ksstat saved')
    with open('./data2/%s/nbins%d//olong%d/kspval.json'%(posname, nbins, olong), 'w') as fp:
        json.dump(kp, fp, sort_keys=True, indent=4)
    print('kspval saved')

    for ic in range(ndim):
        
        todump = {}
        for key in errchain.keys(): 
            #todump[key] = list(errchain[key][:, ic])
            todump[key] = errchain[key][:, ic].tolist()
        fpname = './data2/%s/nbins%d/olong%d/errchain_%s.json'%(posname, nbins, olong, names[ic])
        with open(fpname, 'w') as fp:
            json.dump(todump, fp, sort_keys=True, indent=4)

        todump = {}
        for key in errchain2.keys(): 
            #todump[key] = list(errchain[key][:, ic])
            todump[key] = errchain2[key][:, ic].tolist()
        fpname = './data2/%s/nbins%d/olong%d/errchain2_%s.json'%(posname, nbins, olong, names[ic])
        with open(fpname, 'w') as fp:
            json.dump(todump, fp, sort_keys=True, indent=4)

        todump = {}
        for key in bincounts.keys(): 
            todump[key] = list(bincounts[key][ic])
            todump[key] = bincounts[key][ic].tolist()
        fpname = './data2/%s/nbins%d/olong%d/bincounts_%s.json'%(posname, nbins, olong, names[ic])
        with open(fpname, 'w') as fp:
            json.dump(todump, fp, sort_keys=True, indent=4)

        print(errchain[key].shape)
