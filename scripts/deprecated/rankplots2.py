import numpy as np
import os
from scipy import signal
from scipy.stats import kstest
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
size = comm.Get_size()

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

sigmarep = np.random.normal(0,3,1000000)
yvrep = np.random.normal(0,1,1000000)
alpharep = np.exp(sigmarep/2.)*yvrep
sigquantile = np.quantile(sigmarep, np.linspace(0, 1, nbins+1))
alpquantile = np.quantile(alpharep, np.linspace(0, 1, nbins+1))
sigquantile[0] = -10000
sigquantile[-1] = 10000
alpquantile[0] = -10000
alpquantile[-1] = 10000

unicdf = []
#for i in range(10000):
for i in range(5000):
    uranks = np.random.randint(0, nbins, 100000)
    cdfuranks = np.cumsum(np.unique(uranks, return_counts=True)[1])
    cdfuranks = cdfuranks/cdfuranks[-1]
    unicdf.append(cdfuranks)
unicdf = np.array(unicdf)


def get_rcc(x, c=5, threshold=0.2):
    tmp = x.copy()
    tmp = (tmp - tmp.mean(axis=0))
    autocorr2 = signal.fftconvolve(tmp, tmp[::-1], mode='full', axes=0)
    rcc = autocorr2[autocorr2.shape[0]//2:]
    rcc /= rcc[0]
    idx = np.arange(rcc.shape[0])    
    taus = 2*np.cumsum(rcc, axis=0)+1# -1
    idx2 = (idx.reshape(-1, 1, 1) < c *taus)
    window = np.argmin(idx2, axis=0)
    #window2 = np.argmin(rcc>0.2, axis=0)
    tc = np.array([taus[window[ii, jj], ii, jj] for ii in range(window.shape[0]) for jj in range(window.shape[1])])
    tc = tc.reshape(window.shape).astype(int)
#     tc2 = np.array([taus[window2[ii, jj], ii, jj] for ii in range(window.shape[0]) for jj in range(window.shape[1])])
#     tc = np.max(np.stack([tc, tc2]), 0)
#     tc2 = tc2.reshape(window.shape).astype(int)
#rcc /= rcc[0]
#tcc = []
#        for j in range(x.shape[1]):
#            for m in range(500):
#                if m > 5*(1 + 2*rcc[:m, j].sum()): break
#            tcc.append((1 + 2*rcc[:m, j].sum()))
#        tcc = np.array(tcc)
#        return rcc, tcc
#
    return rcc, tc


def get_cdf(x):
    test = x.copy()
    quantiles = np.vstack([sigquantile, [alpquantile]*(x.shape[-1]-1)]).T
    ranks = np.array([np.searchsorted(quantiles[:, i], test[:, i]) for i in range(x.shape[1])])-1
    cdfranks = np.zeros((nbins, x.shape[1]))
    countsrank = np.zeros((nbins, x.shape[1]))
    for i in range(x.shape[1]):
        x, counts = np.unique(ranks[i], return_counts=True)
        if x[-1] == nbins:
            print("ranks = ", x, counts)
            if counts[-1] > 10: 
                sys.exit()
            counts[-2] += counts[-1]
            x = x[:-1]
            counts = counts[:-1]
        countsrank[x, i] = counts
        cdfranks[x, i] = np.cumsum(counts)/np.sum(counts)
    return cdfranks.T, countsrank.T


def clean_samples(x, maxv=None):

    rc, tc = get_rcc(x)
    subs = tc.max(axis=1)
    #print("subs from rcc : ", subs)
    for i in range(subs.size):
        #pass
        if subs[i] == 1: 
            subs[i] = x.shape[0]
            print('Something wrong for correlation length to be 1, setting it to be size of x', i, subs[i])
    if maxv is not None: 
        print('Correlated samples : %d - %0.2f\n'%(maxv, subs.mean()), subs)
        for i in range(subs.size): 
            if (subs[i] > maxv) & (subs[i] != x.shape[0]): subs[i] = maxv
        print('Updated subs :  %d\n'%maxv, subs)    
    #print("Subs implemented : ", subs)
    toret = np.concatenate([x[::subs[i], i] for i in range(subs.size)])
    return x, toret



########################################## 
steps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.04, 0.02,  0.01]
#steps = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
#steps = [1.0]#, 1.0]
facs = [2, 5, 10]
subs = [2, 3, 4, 5, 6, 7, 8]
#subs = [2]
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
cdfs, counts = {}, {}
costs = {}
kss = {}
chisqs = {}


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
    comm.Barrier()
    Nleapfrog = int(lpath / steps[istep])
    Nleapfrog = max(10, Nleapfrog)
    if olong == 1: fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long/funnel//Ndim%02d/'%ndim
    else: fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long%d/funnel//Ndim%02d/'%(olong, ndim)
    fpathold = '/mnt/ceph/users/cmodi/hmc/outputs_long/funnel//Ndim%02d/'%ndim
    fpath = fpathold + 'step%03d_nleap%02d/'%(steps[istep]*100, Nleapfrog)
    key = 'step %0.3f'%(steps[istep])
    try:
        if rank == 0: 
            if os.path.isfile(fpath + '/samples.npy'): pass
            elif os.path.isfile(fpath + '/samples/00.npy'): 
                combine(fpath)
            ranksamples = clean_samples(np.load(fpath + '/samples.npy')[::args.subsample])
            readkey = key
            print("rank read in : ", rank, key, ranksamples[0].shape, ranksamples[1].shape)
            rankacc = np.load(fpath + '/accepted.npy')
            #rankprobs = np.load(fpath + '/probs.npy')
            rankcounts = (Nleapfrog + 3)*50*1e5 #np.load(fpath + '/counts.npy')
        else: pass
    except Exception as e: 
        print(e)

    for fac in facs:

        for isub, sub in enumerate(subs):
            fpath = fpath0 + 'step%03d_nleap%02d_fac%02d_nsub%d/'%(steps[istep]*100, Nleapfrog, fac, sub)
            key = 'step %0.3f//%d-%d'%(steps[istep],  fac, sub)
            try: 
                if isub == rank-1:
                    if os.path.isfile(fpath + '/samples.npy'): pass
                    elif os.path.isfile(fpath + '/samples/00.npy'): 
                        combine(fpath)
                    ranksamples = clean_samples(np.load(fpath + '/samples.npy')[::args.subsample])
                    readkey = key
                    print("rank read in : ", rank, key, ranksamples[0].shape, ranksamples[1].shape)
                    #print("rank read in : ", rank, key, ranksamples.shape)
                    rankacc = np.load(fpath + '/accepted.npy')
                    #rankprobs = np.load(fpath + '/probs.npy')
                    rankcounts = np.load(fpath + '/counts.npy')
            except Exception as e:  print(e)
                
        comm.Barrier()
        samplekeys = comm.gather(readkey, root=0)
        samplevals = comm.gather(ranksamples, root=0)
        sampleacc = comm.gather(rankacc, root=0)
        #sampleprobs = comm.gather(rankprobs, root=0)
        samplecounts = comm.gather(rankcounts, root=0)
        print(samplekeys)
        if rank == 0:
            for ik in range(len(samplekeys)):
                if samplekeys[ik] is not None:
                    #print("gathered array : ",  ik, samplekeys[ik], samplevals[ik].shape)
                    samples[samplekeys[ik]] = samplevals[ik][0]
                    clsamples[samplekeys[ik]] = samplevals[ik][1]
                    acc[samplekeys[ik]] = sampleacc[ik]
                    #probs[samplekeys[ik]] = sampleprobs[ik]
                    vcounts[samplekeys[ik]] = samplecounts[ik]
                    
        comm.Barrier()
    
    if rank == 0:
        for kk in samples.keys():
            print('Sample shape ', kk, samples[kk].shape, clsamples[kk].shape)
                

        try: os.makedirs('./data/nbins%d/'%(nbins))
        except Exception as e: print(e)
        try: os.makedirs('./data/nbins%d/olong%d/'%(nbins, olong))
        except Exception as e: print(e)
        try: os.makedirs('./data/nbins%d/olong%d/Ndim%02d/'%(nbins, olong, ndim))
        except Exception as e: print(e)
        ################
        def ks(x):
            ssize = x.sum()
            probs = x/ssize
            bounds = np.linspace(0, 1, nbins+1)
            sscheck = np.random.random(int(ssize))
            rvs = []
            for i in range(nbins):
                rvs = rvs + list(sscheck[(sscheck > bounds[i]) & (sscheck < bounds[i+1])][:int(probs[i]*sscheck.size)])
            rvs  = np.array(rvs)
            np.random.shuffle(rvs)
            return kstest(rvs, 'uniform')
        
        for key in samples.keys():
            cdfs[key], counts[key] = get_cdf(clsamples[key])
            if '//' in key: 
                nevals = vcounts[key].sum(axis=(0, 1))[:2].sum()
            else: 
                nevals = vcounts[key] #.sum(axis=(0, 1))[:2].sum()

            costs[key] =  samples[key].shape[1]*(3/samples[key].mean(axis=0)[:, 0].std())**2 / nevals#clsamples[key].shape[0]/nevals

            chisq0 = (((cdfs[key][0] - unicdf.mean(axis=0))/(0.00011 + unicdf.std(axis=0)))**2).mean()**0.5
            chisq1 = ((((cdfs[key][1:] - unicdf.mean(axis=0))/(0.00011 + unicdf.std(axis=0)))**2).mean()**0.5) #/ (ndim-1)
            chisqs[key] = [chisq0, chisq1]

        with open('./data/nbins%d/olong%d/Ndim%02d/l%02d_chisqs.json'%(nbins, olong, ndim, lpath), 'w') as fp:
            json.dump(chisqs, fp, sort_keys=True, indent=4)
        with open('./data/nbins%d//olong%d/Ndim%02d/l%02d_chisqs.json'%(nbins, olong, ndim, lpath), 'w') as fp:
            json.dump(chisqs, fp, sort_keys=True, indent=4)
        with open('./data/nbins%d//olong%d/Ndim%02d/l%02d_costs.json'%(nbins, olong, ndim, lpath), 'w') as fp:
            json.dump(costs, fp, sort_keys=True, indent=4)

        for ic in range(ndim):
            todump = {}
            for key in counts.keys(): todump[key] = list(counts[key][ic])
            if ic == 0: 
                fpname = './data/nbins%d/olong%d/Ndim%02d/l%02d_counts_sigma.json'%(nbins, olong, ndim, lpath)
            else: fpname = './data/nbins%d/olong%d/Ndim%02d/l%02d_counts_alpha%d.json'%(nbins, olong, ndim, lpath, ic)
            with open(fpname, 'w') as fp:
                json.dump(todump, fp, sort_keys=True, indent=4)

        for key in samples.keys():
            ks0 = ks(counts[key][0]) #test(counts[key][0], 'uniform')
            ks1s, ks1p = 0, 100
            for j in range(1, ndim):
                ks1 = ks(counts[key][j]) #test(counts[key][j], 'uniform')
                if ks1.statistic > ks1s: ks1s = ks1.statistic
                if ks1.pvalue < ks1p: ks1p = ks1.pvalue
            kss[key] = [[ks0.statistic, ks0.pvalue], [ks1s, ks1p]]
            

        #with open('./data/Ndim%02d/l%02d_kstats.json'%(ndim, lpath), 'w') as fp:
        #    json.dump(kss, fp, sort_keys=True, indent=4)
            
            
        #################################
        ##plot histogram

        nplot = min(5, ndim)
        fig, axar = plt.subplots(nplot, nfacs+1, figsize=(nfacs*5, 3.5*nplot), sharey='row')
        bins = 100
        for d in range(nplot):
            ax = axar[d]
            ss = steps[istep]
            try: 
                mu, sig = samples['step %0.3f'%ss][...,d].flatten().mean(), samples['step %0.3f'%ss][...,d].flatten().std()
                lbl = '%0.3f\n%0.2f(%0.2f)'%(steps[istep], mu, sig)
                ax[0].hist(samples['step %0.3f'%ss][...,d].flatten(), bins=bins, 
                           alpha=1, label=lbl, histtype='step', lw=2, density=True)
            except: pass
            for j, ff in enumerate(facs):
                for k, sub in enumerate(subs):
                    try:
                        key = 'step %0.3f//%d-%d'%(ss, ff, sub)
                        mu, sig = samples[key][...,d].flatten().mean(), samples[key][...,d].flatten().std()
                        lbl = '//%d-%d\n%0.2f(%0.2f)'%(ff,sub,  mu, sig)
                        ax[j+1].hist(samples[key][...,d].flatten(), bins=bins, 
                             alpha=1, label=lbl, histtype='step', lw=2, density=True)
                    except: pass

        for axis in axar[0]:
            axis.hist(np.random.normal(0, 3, 10000), alpha=0.5, bins=bins, color='gray', density=True)
        for axis in axar.flatten():
            axis.grid(which='both')
            axis.semilogy()
            axis.legend(fontsize=10)
        axar[0, 0].set_title('Vanilla HMC')
        for j, ff in enumerate(facs): axar[0, j+1].set_title('Reduce by factor %d'%ff)        
        axar[0, 0].set_ylabel('log sigma')
        for i in range(nplot-1): axar[i+1, 0].set_ylabel('alpha %d'%(i+1))
        for axis in axar[1:].flatten(): axis.set_ylim(1e-5, 0.5)
        plt.tight_layout()
        plt.savefig('./figs/Ndim%02d/step%0.3f_l%02d_hist.png'%(ndim, ss, lpath))
        plt.close()
        print("Histogram saved in './figs/Ndim%02d/step%0.3f_l%02d_hist.png"%(ndim, ss, lpath))



        #######################################
        ####Rank plos
        uranks = np.random.randint(0, nbins, 10000)
        cdfuranks = np.cumsum(np.unique(uranks, return_counts=True)[1])
        cdfuranks = cdfuranks/cdfuranks[-1]
        xpts = np.arange(0, cdfuranks.size)
        try: maxsample = max([clsamples[key].shape[0] for key in clsamples.keys()])
        except: maxsample = 100000
        print('max sample : ', maxsample)
        unicdf = []
        #for i in range(10000):
        for i in range(2000):
            uranks = np.random.randint(0, nbins, maxsample)
            cdfuranks = np.cumsum(np.unique(uranks, return_counts=True)[1])
            cdfuranks = cdfuranks/cdfuranks[-1]
            unicdf.append(cdfuranks)
        unicdf = np.array(unicdf)

        nplot = min(5, ndim)
        fig, axar = plt.subplots(nplot, nfacs+1, figsize=(nfacs*4+1, 3*nplot), sharex=True, sharey=True)
        fig2, axar2 = plt.subplots(nplot, nfacs+1, figsize=(nfacs*4+1, 3*nplot), sharex=True, sharey=True)

        def get_lbl(key, d):
            lbl = None
            hmcsamples =  clsamples[key][...,d].flatten()
            try:  nevals = vcounts[key].sum(axis=(0, 1))[:2].sum()
            except: nevals = vcounts[key]
            if d == 0: lbl = key
            if d == 0 and "//" in key: lbl = key[10:]
            try:
                if d == 1: lbl =  "{}".format(fmt(clsamples[key].shape[0]))#=%0.1e"%hmcsamples.shape[0]
                if d == 2: lbl =  "{}\n{}".format(fmt(50*(3/samples[key].mean(axis=0)[:, 0].std())**2), fmt(50*(9.5/samples[key].mean(axis=0)[:, 1].std())**2))#=%0.1e"%hmcsamples.shape[0]
                if d == 3: lbl = "{}".format(fmt(nevals))
                if d == 4: lbl  = "{}".format(fmt(nevals/hmcsamples.shape[0]))
            except : lbl = None
            return lbl
     
        
        for d in range(nplot):
            ax = axar[d]
            ss = steps[istep]
            try: 
                key = 'step %0.3f'%ss
                lbl = get_lbl(key, d)
                x = np.arange(cdfs[key][d].size)
                axar[d, 0].step(x, cdfs[key][d] -  unicdf.mean(axis=0)[x], label=lbl, lw=2)
                axar2[d, 0].step(x, counts[key][d], label=lbl, lw=2)
            except Exception as e:  pass #print("excpetion : ", e)

            for j, ff in enumerate(facs):
                for k, sub in enumerate(subs):
                    try:
                        key = 'step %0.3f//%d-%d'%(ss, ff, sub)
                        lbl = get_lbl(key, d)
                        x = np.arange(cdfs[key][d].size)
                        axar[d, j+1].step(x, cdfs[key][d] -  unicdf.mean(axis=0)[x], label=lbl, lw=2)
                        axar2[d, j+1].step(x, counts[key][d], label=lbl, lw=2)
                    except Exception as e: pass #print("exception : ", key, e)

        for axis in axar.flatten():
            axis.grid(which='both')
            axis.legend(loc='lower right', ncol=2, fontsize=9)
            axis.fill_between(xpts, -unicdf.std(axis=0), unicdf.std(axis=0), color='gray', alpha=0.2)
            axis.set_ylim(-0.02, 0.02)
        axar[0, 0].set_title('Vanilla HMC')
        for axis in axar[1]: axis.text(2, 0.021, "ESS (rcc)")
        for axis in axar[2]: axis.text(2, 0.021, "ESS (sigma), ESS (alpha)")
        for axis in axar[3]: axis.text(2, 0.021, "#H")
        for axis in axar[4]: axis.text(2, 0.02, "Cost=#H/ESS")
        for j, ff in enumerate(facs): axar[0, j+1].set_title('Reduce by factor %d'%ff)        
        axar[0, 0].set_ylabel('log sigma')
        for i in range(nplot-1): axar[i+1, 0].set_ylabel('alpha %d'%(i+1))
        fig.tight_layout()
        fig.savefig('./figs/Ndim%02d/step%0.3f_l%02d_ecdf.png'%(ndim, ss, lpath))
        plt.close(fig)

        for axis in axar2.flatten():
            axis.grid()
            axis.legend(loc='lower center')
        axar2[0, 0].set_title('Vanilla HMC')
        for j, ff in enumerate(facs): axar2[0, j+1].set_title('Reduce by factor %d'%ff)        
        axar2[0, 0].set_ylabel('log sigma')
        for i in range(nplot-1): axar2[i+1, 0].set_ylabel('alpha %d'%(i+1))
        fig2.tight_layout()
        fig2.savefig('./figs/Ndim%02d/step%0.3f_l%02d_ranks.png'%(ndim, ss, lpath))
        plt.close(fig2)
        print("Ranks saved in './figs/Ndim%02d/step%0.3f_l%02d_ranks.png"%(ndim, ss, lpath))


        ############################
        ##plot mean and std
        
        fig, axar = plt.subplots(2, nfacs+1, figsize=(nfacs*4+1, 6), sharex=True, sharey='row')

        ss = steps[istep]
        key = 'step %0.3f'%ss
        try : xx = samples[key]
        except : continue
        nevals = vcounts[key] #.sum(axis=(0, 1))[:2].sum()
        costs[key] =  50*(3/samples[key].mean(axis=0)[:, 0].std())**2 / nevals#clsamples[key].shape[0]/nevals
        lbl = "ESS=%0.1e"%(clsamples[key].shape[0])
        axar[0, 0].errorbar(np.arange(ndim), xx.mean(axis=(0, 1)), xx.mean(axis=0).std(axis=0), alpha=0.7,  label=key, elinewidth=0.5)
        axar[1, 0].errorbar(np.arange(ndim), xx.std(axis=(0, 1)), xx.std(axis=0).std(axis=0),  alpha=0.7, label=lbl)
        for j, ff in enumerate(facs):
            for k, sub in enumerate(subs):
                key = 'step %0.3f//%d-%d'%(ss, ff, sub)
                try : xx = samples[key]
                except : continue
                try: 
                    nevals = vcounts[key].sum(axis=(0, 1))[:2].sum()
                    costs[key] = 50*(3/samples[key].mean(axis=0)[:, 0].std())**2 /nevals # clsamples[key].shape[0]/nevals
                    lbl = "%0.1e"%(clsamples[key].shape[0])
                except : lbl =  "ESS=%d"%xx.shape[0]
                axar[0, j+1].errorbar(np.arange(ndim) + k*0.05, xx.mean(axis=(0, 1)), xx.mean(axis=0).std(axis=0), alpha=0.7, label=key[5:], elinewidth=0.5)
                axar[1, j+1].errorbar(np.arange(ndim) + k*0.05, xx.std(axis=(0, 1)), 0*xx.std(axis=0).std(axis=0), alpha=0.7, label=lbl)
                #axar[1, j+1].plot(xx.std(axis=(0)), marker='o',  label=lbl)
                #axar[0, ik].text(1, -0.4,  samples0[key].shape[0])

        for axis in axar.flatten():
            axis.grid()
            axis.legend(ncol=2, fontsize=9, loc=4)

        for axis in axar[0].flatten():
            axis.set_ylim(-0.5, 0.5)
        for axis in axar[1].flatten():
            axis.set_ylim(2, 12)
            axis.axhline(3, ls='--', color='r')
        #for axis in axar[0]:
        axar[0, 0].set_ylabel('Mean values')
        #for axis in axar[2:, 0]:
        axar[1, 0].set_ylabel('Std dev values')
        for axis in axar[-1]:
            axis.set_xlabel('Dimension')

        plt.tight_layout()
        plt.savefig('./figs/Ndim%02d/step%0.3f_l%02d_means.png'%(ndim, ss, lpath))
        plt.close()
        print("Means saved in './figs/Ndim%02d/step%0.3f_l%02d_means.png"%(ndim, ss, lpath))


        ############################
        ##plot costs
            
        fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True, sharey=False)
        for key in costs:
            if str(steps[istep]) not in key: continue
            if clsamples[key].shape[0] == 1e5*50 : 
                print('all samples is worng')
                continue
            mm = 'o'
            if '//5' in key: mm = 'x'
            if '//10' in key: mm = '*'
            ax[0, 0].plot(1/costs[key], chisqs[key][0], mm, label=key[4:])
            ax[0, 1].plot(1/costs[key], kss[key][0][0], mm, label=key[9:])
            ax[0, 2].plot(1/costs[key], kss[key][0][1], mm, label=key[9:])
            ax[1, 0].plot(1/costs[key], chisqs[key][1], mm, label=key[4:])
            ax[1, 1].plot(1/costs[key], kss[key][1][0], mm, label=key[9:])
            ax[1, 2].plot(1/costs[key], kss[key][1][1], mm, label=key[9:])
        for axis in ax.flatten(): 
            axis.grid(which='both')
            axis.semilogx()
        for axis in ax[:, 0]: axis.loglog()
        for axis in ax[:, 2]: axis.axhline(0.05, color='r', ls='--')
        #for axis in ax[:, :-1].flatten(): axis.loglog()
        ax[1, 2].legend(ncol=2, fontsize=9)
        ax[0, 0].set_ylabel(r'log_sigma')
        ax[1, 0].set_ylabel(r'alpha')
        ax[0, 0].set_title(r'$\chi^2$')
        ax[0, 1].set_title(r'KS Statistic')
        ax[0, 2].set_title(r'p value')
        for axis in ax[-1].flatten(): axis.set_xlabel(r'Cost=#H/ESS$_{\rm logsigma}$')
        plt.tight_layout()
        plt.savefig('./figs/Ndim%02d/step%0.3f_l%02d_costs.png'%(ndim, ss, lpath))
        plt.close()
        print("Costs saved in './figs/Ndim%02d/step%0.3f_l%02d_costs.png"%(ndim, ss, lpath))
##

#####################

if rank == 0:

    fig, ax = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for istep in range(nsteps):    
        for key in costs:
            if str(steps[istep]) in key:
                pass
            else: continue
            if clsamples[key].shape[0] == 1e5*50 : 
                ax.flatten()[istep].plot(np.NaN, np.NaN, mm, label=key[9:])
                print('all samples is worng')
                continue
            mm = 'o'
            if '//5' in key: mm = 'x'
            if '//10' in key: mm = '*'
            chisq = (chisqs[key][0]**2 +  chisqs[key][1]**2)**0.5 #(((cdfs[key] - unicdf.mean(axis=0))/(0.00011 + unicdf.std(axis=0)))**2).sum()**0.5
            ax.flatten()[istep].plot(1/costs[key], chisq, mm, label=key[9:])
            ax.flatten()[istep].set_title(key[:10])
    for axis in ax.flatten(): 
        axis.legend(ncol=2, fontsize=9)
        axis.grid(which='both')
        axis.loglog()
    for axis in ax[:, 0]: axis.set_ylabel(r'$\chi^2$')
    for axis in ax[-1].flatten(): axis.set_xlabel(r'Cost=#H/ESS$_{\rm logsigma}$')
    #for axis in ax[1]: axis.set_xlabel('Cost=#H/ESS')

    plt.tight_layout()
    plt.savefig('./figs/Ndim%02d/l%02d_chisq.png'%(ndim, lpath))
    plt.close()
    print("Save all chisqs")


    fig, ax = plt.subplots(4, 4, figsize=(15, 13), sharex=True, sharey=True)
    fig2, ax2 = plt.subplots(4, 4, figsize=(15, 13), sharex=True, sharey=True)
    for istep in range(nsteps):    
        for key in costs:
            if str(steps[istep]) in key:
                pass
            else: continue
            if clsamples[key].shape[0] == 1e5*50 : 
                ax.flatten()[istep].plot(np.NaN, np.NaN, mm, label=key[9:])
                print('all samples is worng')
                continue
            mm = 'o'
            if '//5' in key: mm = 'x'
            if '//10' in key: mm = '*'
            ks0, ks1 = kss[key]
            ax.flatten()[istep].plot(1/costs[key], ks0[0], mm, label=key[9:])
            ax.flatten()[istep].set_title(key[:10])
            ax2.flatten()[istep].plot(1/costs[key], ks0[1], mm, label=key[9:])
            ax2.flatten()[istep].set_title(key[:10])
            ax.flatten()[istep+8].plot(1/costs[key], ks1[0], mm, label=key[9:])
            ax.flatten()[istep+8].set_title(key[:10])
            ax2.flatten()[istep+8].plot(1/costs[key], ks1[1], mm, label=key[9:])
            ax2.flatten()[istep+8].set_title(key[:10])
    for axis in ax.flatten(): 
        axis.legend(ncol=2, fontsize=9)
        axis.grid(which='both')
        axis.loglog()
    for axis in ax[:2, 0]: axis.set_ylabel(r'KS statistic log_sigma')
    for axis in ax[2:, 0]: axis.set_ylabel(r'KS statistic alpha')
    for axis in ax[-1].flatten(): axis.set_xlabel(r'Cost=#H/ESS$_{\rm logsigma}$')
    #for axis in ax[-1]: axis.set_xlabel('Cost=#H/ESS')

    for axis in ax2.flatten(): 
        axis.legend(ncol=2, fontsize=9)
        axis.grid(which='both')
        #axis.axhline(0.05, color='r', ls="--")
        axis.semilogx() #loglog()
    for axis in ax2[:2, 0]: axis.set_ylabel(r'pvalue log_sigma')
    for axis in ax2[2:, 0]: axis.set_ylabel(r'pvalue alpha')
    for axis in ax2[-1].flatten(): axis.set_xlabel(r'Cost=#H/ESS$_{\rm logsigma}$')
    #for axis in ax2[-1]: axis.set_xlabel('Cost=#H/ESS')

    fig.tight_layout()
    fig.savefig('./figs/Ndim%02d/l%02d_kstest.png'%(ndim, lpath))
    plt.close(fig)

    fig2.tight_layout()
    fig2.savefig('./figs/Ndim%02d/l%02d_pvalue.png'%(ndim, lpath))
    plt.close(fig2)
    print("Save all costs")
    #print("All Costs saved in './figs/Ndim%02d/step%0.3f_l%02d_means.png"%(ndim, ss, lpath))

