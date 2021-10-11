import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


nbins = 20
sigmarep = np.random.normal(0,3,1000000)
yvrep = np.random.normal(0,1,1000000)
alpharep = np.exp(sigmarep/2.)*yvrep
sigquantile = np.quantile(sigmarep, np.linspace(0, 1, nbins+1))
alpquantile = np.quantile(alpharep, np.linspace(0, 1, nbins+1))

unicdf = []
for i in range(10000):
    uranks = np.random.randint(0, nbins, 10000)
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
    taus = 2*np.cumsum(rcc, axis=0)-1
    idx2 = (idx.reshape(-1, 1, 1) < c *taus)
    window = np.argmin(idx2, axis=0)
    #window2 = np.argmin(rcc>0.2, axis=0)
    tc = np.array([taus[window[ii, jj], ii, jj] for ii in range(window.shape[0]) for jj in range(window.shape[1])])
    tc = tc.reshape(window.shape).astype(int)
#     tc2 = np.array([taus[window2[ii, jj], ii, jj] for ii in range(window.shape[0]) for jj in range(window.shape[1])])
#     tc = np.max(np.stack([tc, tc2]), 0)
#     tc2 = tc2.reshape(window.shape).astype(int)
    return rcc, tc


def clean_samples(x):
    rc, tc = get_rcc(x)
    subs = tc.max(axis=1)
    toret = np.concatenate([x[::subs[i], i] for i in range(subs.size)])
    return toret



########################################## 
ndim = 20
lpath = 30
steps = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
#steps = [0.1]
facs = [2, 5, 10]
subs = [2, 3, 4, 5, 6, 7, 8]
#subs = [2]
nsteps = len(steps)
nfacs = len(facs)
nsubs = len(subs)
samples = {}
acc = {}
probs = {}


keystoplot = []
alldiff = 1000
savekey = ''
    

for i in range(nsteps):    
    Nleapfrog = int(lpath / steps[i])
    Nleapfrog = max(10, Nleapfrog)
    fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long//Ndim%02d/'%ndim
    fpath = fpath0 + 'step%03d_nleap%02d/'%(steps[i]*100, Nleapfrog)
    key = 'step %0.3f'%(steps[i])
    try:
        samples[key] = clean_samples(np.load(fpath + '/samples.npy'))
        acc[key] = np.load(fpath + '/accepted.npy')
        probs[key] = np.load(fpath + '/probs.npy')
    except Exception as e: print(e)

    for fac in facs:
        fpath = fpath0 + 'step%03d_nleap%02d_fac%02d/'%(steps[i]*100, Nleapfrog, fac)
        key = 'step %0.3f//%d'%(steps[i],  fac)
        try: 
            samples[key] = clean_samples(np.load(fpath + '/samples.npy'))
        except Exception as e: print(e)

        for sub in subs:
            fpath = fpath0 + 'step%03d_nleap%02d_fac%02d_nsub%d/'%(steps[i]*100, Nleapfrog, fac, sub)
            key = 'step %0.3f//%d-%d'%(steps[i],  fac, sub)
            print(key)
            try: 
                samples[key] = clean_samples(np.load(fpath + '/samples.npy'))
                acc[key] = np.load(fpath + '/accepted.npy')
            except Exception as e: pass #print(e)
                

    for kk in samples.keys():
        print(kk, samples[kk].shape)


    #################################
    ##plot histogram

    nplot = min(5, ndim)
    fig, axar = plt.subplots(nplot, nfacs+1, figsize=(nfacs*4, 3*nplot))
    bins = 100
    for d in range(nplot):
        ax = axar[d]
        ss = steps[i]
        try: 
            mu, sig = samples['step %0.3f'%ss][...,d].flatten().mean(), samples['step %0.3f'%ss][...,d].flatten().std()
            lbl = '%0.3f\n%0.2f(%0.2f)'%(steps[i], mu, sig)
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
        axis.legend()
    plt.tight_layout()
    plt.savefig('./figs/Ndim%02d/step%0.3f_l%02d_hist.png'%(ndim, ss, lpath))
    plt.close()

    #######################################
    ####Rank plots

    uranks = np.random.randint(0, nbins, 10000)
    cdfuranks = np.cumsum(np.unique(uranks, return_counts=True)[1])
    cdfuranks = cdfuranks/cdfuranks[-1]
    xpts = np.arange(0, cdfuranks.size)


    nplot = min(5, ndim)
    fig, axar = plt.subplots(nplot, nfacs+1, figsize=(nfacs*4+1, 3*nplot))
    fig2, axar2 = plt.subplots(nplot, nfacs+1, figsize=(nfacs*4+1, 3*nplot))
    bins = 100
    for d in range(nplot):
        ax = axar[d]
        ss = steps[i]
        try: 
            key = 'step %0.3f'%ss
            hmcsamples = samples[key][...,d].flatten()
            if d == 0: ranks = np.searchsorted(sigquantile, hmcsamples) - 1                  
            else: ranks = np.searchsorted(alpquantile, hmcsamples) - 1                
            x, counts = np.unique(ranks, return_counts=True)
            cdfranks = np.cumsum(counts)
            cdfranks = cdfranks/cdfranks[-1]
            axar[d, 0].step(x, cdfranks -  unicdf.mean(axis=0)[x], label=key, lw=2)
            axar2[d, 0].step(x, counts, label=key, lw=2)
        except Exception as e: 
            print("excpetion : ", e)
            pass

        for j, ff in enumerate(facs):
            for k, sub in enumerate(subs):
                try:
                    key = 'step %0.3f//%d-%d'%(ss, ff, sub)
                    hmcsamples = samples[key][...,d].flatten()
                    if d == 0: ranks = np.searchsorted(sigquantile, hmcsamples) - 1                  
                    else: ranks = np.searchsorted(alpquantile, hmcsamples) - 1                
                    x, counts = np.unique(ranks, return_counts=True)
                    cdfranks = np.cumsum(counts)
                    cdfranks = cdfranks/cdfranks[-1]
                    axar[d, j+1].step(x, cdfranks -  unicdf.mean(axis=0)[x], label=key[5:], lw=2)
                    axar2[d, j+1].step(x, counts, label=key[5:], lw=2)
                except Exception as e: print("exception : ", key, e)

    for axis in axar.flatten():
        axis.grid(which='both')
        axis.legend(loc='lower right', ncol=2, fontsize=11)
        axis.fill_between(xpts, -unicdf.std(axis=0), unicdf.std(axis=0), color='gray', alpha=0.2)
        axis.set_ylim(-0.05, 0.05)
    fig.tight_layout()
    fig.savefig('./figs/Ndim%02d/step%0.3f_l%02d_ecdf.png'%(ndim, ss, lpath))
    #fig.savefig('./figs/Ndim%02d/step%0.3f_ecdf.png'%(ndim, ss))
    plt.close(fig)

    for axis in axar2.flatten():
        axis.grid()
        axis.legend(loc='lower center')
    fig2.tight_layout()
    fig2.savefig('./figs/Ndim%02d/step%0.3f_l%02d_ranks.png'%(ndim, ss, lpath))
    #fig2.savefig('./figs/Ndim%02d/step%0.3f_ranks.png'%(ndim, ss))
    plt.close(fig2)

#$#
#$#    fig, axar = plt.subplots(4, 5, figsize=(18, 12 ), sharex=True, sharey=True)
#$#
#$#    for iff, ss in enumerate([0.5, 0.2, 0.1, 0.05, 0.01]):
#$#        fidkey = 'step %0.3f'%ss    
#$#        ax = axar[0, iff]
#$#
#$#        for ik, nsub in enumerate([0, 2, 5, 10]):
#$#
#$#            if nsub == 0: kk = fidkey
#$#        else: kk = fidkey + '//%s'%nsub
#$#            try: 
#$#                test = samples0[kk]
#$#                hmcsamples = test[..., ii].copy().flatten() #clean_samples(test)    
#$#                if ii == 0: ranks = np.searchsorted(sigquantile, hmcsamples) - 1                  
#$#                else: ranks = np.searchsorted(alpquantile, hmcsamples) - 1                
#$#            except Exception as e: 
#$#                print("excpetion : ", e)
#$#                continue
#$#
#$#            cdfranks = np.cumsum(np.unique(ranks, return_counts=True)[1])
#$#            cdfranks = cdfranks/cdfranks[-1]
#$#            ax.step(xpts, cdfranks - unicdf.mean(axis=0), label=kk, lw=2)
#$#
#$#    for iss, ss in enumerate([2.0, 1.0, 0.5, 0.2, 0.1]):
#$#        for ifac, fac in enumerate([2, 5, 10]):
#$#
#$#            iff = iss + 4*ifac
#$#            fidkey = 'step %0.3f//%d'%(ss, fac)    
#$#
#$#    #         ax = axar[1:].flatten()[iff]
#$#            ax = axar[ifac+1, iss]
#$#
#$#            for insub, nsub in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
#$#                kk = fidkey + '-%d'%nsub
#$#                try: 
#$#                    test = samples[kk]
#$#                    hmcsamples = test[..., ii].copy().flatten() #clean_samples(test)    
#$#                    if ii == 0: ranks = np.searchsorted(sigquantile, hmcsamples) - 1                  
#$#                    else: ranks = np.searchsorted(alpquantile, hmcsamples) - 1                
#$#                except Exception as e:
#$#    #                 print("Exception : ", e)
#$#                    continue
#$#                x, counts = np.unique(ranks, return_counts=True)
#$#    #             x -= 1
#$#                cdfranks = np.cumsum(counts)
#$#                cdfranks = cdfranks/cdfranks[-1]
#$#                ax.step(x, cdfranks -  unicdf.mean(axis=0)[x], label=kk, lw=2)
#$#    #             cdfranks = np.cumsum(np.unique(ranks, return_counts=True)[1])
#$#    #             cdfranks = cdfranks/cdfranks[-1]
#$#    #             ax.step(xpts, cdfranks -  unicdf.mean(axis=0), label=kk, lw=2)
#$#
#$#
#$#    alpha = 0.01
#$#    nobs = hmcsamples.size
#$#    epsilon = np.sqrt(np.log(2.0 / alpha) / (2 * nobs))
#$#    print(epsilon)
#$#    for axis in axar.flatten():
#$#        axis.grid()
#$#        axis.legend(loc='lower center', ncol=2)
#$#        axis.fill_between(xpts, -unicdf.std(axis=0), unicdf.std(axis=0), color='gray', alpha=0.2)
#$#    #     axis.step(xpts, cdfuranks-cdfuranks2, 'k--')
#$#        axis.set_ylim(-0.05, 0.05)
#$#
    #plt.suptitle('Rank  plots for log sigma', fontsize=14)
    #plt.tight_layout()
