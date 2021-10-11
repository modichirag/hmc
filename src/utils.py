import numpy as np
from scipy import signal
import sys
import time


def get_rcc(x, c=5, threshold=0.2):
    tmp = x.copy()
    tmp = (tmp - tmp.mean(axis=0))
    autocorr2 = signal.fftconvolve(tmp, tmp[::-1], mode='full', axes=0)
    rcc = autocorr2[autocorr2.shape[0]//2:]
    rcc /= rcc[0]
    idx = np.arange(rcc.shape[0])
    taus = 2*np.cumsum(rcc, axis=0)+1
    idx2 = (idx.reshape(-1, 1, 1) < c *taus)
    window = np.argmin(idx2, axis=0)
    tc = np.array([taus[window[ii, jj], ii, jj] for ii in range(window.shape[0]) for jj in range(window.shape[1])])
    tc = tc.reshape(window.shape).astype(int) #+1 #because int rounds down
    return rcc, tc

def clean_samples(xx, maxv=None, fidsub=1):

    x = xx[::fidsub]
    rc, tc = get_rcc(x)
    subs = tc.max(axis=1)
    subs[subs==0]=1
    for i in range(subs.size):  
        if subs[i] == 1 and (x[0, i]-x[10, i]).sum()==0:
            subs[i] = x.shape[0]
            print('Something wrong for correlation length to be 1, setting it to be size of x', i, subs[i])
    if maxv is not None:
        print('Correlated samples : %d - %0.2f\n'%(maxv, subs.mean()), subs)
        for i in range(subs.size):
            if (subs[i] > maxv) & (subs[i] != x.shape[0]): subs[i] = maxv
        print('Updated subs :  %d\n'%maxv, subs)
    #print("Subs implemented : ", subs)                                                                                                                      
    toret = np.concatenate([x[::subs[i], i] for i in range(subs.size)])
    return toret, rc, tc



def get_cdf(x, xref=None, quantiles=None, nbins=20, qs=None, xmin=None, xmax=None):

    ndim = x.shape[1]
    if quantiles is None:
        if xref is None: 
            print('Need either xref or quantiles')
            return None
        quantiles = np.zeros((ndim, nbins+1))
        if qs is None: qs = np.linspace(0.0, 1., nbins+1)
        for i in range(ndim):
            quantiles[i] = np.quantile(xref[:, i], qs)
            if xmin is not None : quantiles[i][0] = xmin
            if xmax is not None : quantiles[i][-1] = xmax

    countsrank = []
    cdfranks = []
    for i in range(ndim):
        #qq = np.quantile(xref[:, i], qs)
        counts = np.histogram(x[:, i], bins=quantiles[i])[0]
        countsrank.append(counts)
        cdfranks.append(np.cumsum(counts)/sum(counts))
    return np.array(cdfranks), np.array(countsrank)


#def get_cdf(x, xref, nbins=20):
#    #nchains, ndim = x.shape[1], x.shape[2]
#    ndim = x.shape[1]
#    test = x.copy()
#    quantiles = []
#    
#    quantiles = np.vstack([np.quantile(xref[:, i], np.linspace(0, 1, nbins+1)) for i in range(ndim)]).T
#    #quantiles = np.vstack([sigquantile, [alpquantile]*(x.shape[-1]-1)]).T
#    ranks = np.array([np.searchsorted(quantiles[:, i], test[:, i]) for i in range(x.shape[1])])-1
#    cdfranks = np.zeros((nbins, x.shape[1]))
#    countsrank = np.zeros((nbins, x.shape[1]))
#    for i in range(x.shape[1]):
#        x, counts = np.unique(ranks[i], return_counts=True)
#        if x[-1] == nbins:
#            #print("ranks = ", x, counts)
#            if counts[-1] > 0.05*counts.mean():
#                print("Counts in the last bin are many", counts)
#                return np.NaN, np.NaN
#            #counts[-2] += counts[-1]
#            x = x[:-1]
#            counts = counts[:-1]
#        countsrank[x, i] = counts
#        cdfranks[x, i] = np.cumsum(counts)/np.sum(counts)
#    return cdfranks.T, countsrank.T
#

def getstanparams(smodel, posterior, nchains=10, niter=100000, seed=100):

    model, data = posterior.model, posterior.data
    stansamples = smodel.sampling(data=data.values(), chains=nchains, iter=niter, seed=seed, n_jobs=1,
                              control={"metric":"diag_e",
                                       "stepsize_jitter":0,
                                "adapt_delta":0.9 })


    stepsizefid = np.array(stansamples.get_stepsize()).mean()
    invmetricfid = np.array(stansamples.get_inv_metric()).mean(axis=0)

    nleapfrogs = np.concatenate([p['n_leapfrog__'] for p in stansamples.get_sampler_params()])
    stepsizes = np.concatenate([p['stepsize__'] for p in stansamples.get_sampler_params()])
    divs = np.concatenate([p['divergent__'] for p in stansamples.get_sampler_params()])
    nss = nleapfrogs[np.where((abs(stepsizes - stepsizefid) < stepsizefid/10.) & (divs==0))]
    Tint = stepsizefid*np.quantile(nss, 0.9)
    Nleapfrogfid = int(Tint/stepsizefid)

    todump = {}
    todump['stepsize'] = stepsizefid
    todump['invmetric'] = list(invmetricfid)
    todump['Tintegration'] = Tint
    todump['Nleapfrog'] = Nleapfrogfid
    return stansamples, todump

    


 
def do_hmc(stepfunc, initstate, nsamples=1000, burnin=100):

    q = initstate
    
    #
    samples = []
    accepts = []
    probsi = []
    countsi = []

    start = time.time()
    print("Starting HMC loop")
    for i in range(nsamples+burnin):
                                                
        out = list(map(stepfunc, q))
        q = [i[0] for i in out]
        acc = [i[2] for i in out]
        prob = [i[3] for i in out]
        count = [i[4] for i in out]
        samples.append(q)
        accepts.append(acc)
        probsi.append(prob)
        countsi.append(count)

    print("Time taken = ", time.time() - start)
    mysamples = np.array(samples)[burnin:]
    accepted = np.array(accepts)[burnin:]
    probs = np.array(probsi)[burnin:]
    counts = np.array(countsi)[burnin:]

    return mysamples, accepted, probs, counts

#
