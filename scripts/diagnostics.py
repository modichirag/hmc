import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot_hist(mysamples, fpath, sub=100):
    #
    ns, nc, ndim = mysamples.shape

    plt.hist(mysamples[::sub, ..., 0].flatten(), bins='auto', alpha=0.5)
    plt.grid(which='both')
    plt.savefig(fpath + '/hist_sigma.png')
    plt.close()

    fig, ax = plt.subplots(ndim-1, 1, figsize = (5, ndim*3-3), sharex=True, sharey=True)
    if ndim == 2: ax = [ax]
    for i in range(1, ndim):
        ax[i-1].hist(mysamples[::sub, ..., i].flatten(), bins='auto', alpha=0.5)
        ax[i-1].grid(which='both')
        ax[i-1].set_yscale('log')
    plt.tight_layout()
    plt.savefig(fpath + '/hist_alpha.png')
    plt.close()


def plot_trace(mysamples, fpath):
    #
    ns, nc, ndim = mysamples.shape

    fig, ax = plt.subplots(nc, 1, figsize=(5, nc*3), sharex=True)
    if nc == 1: ax = [ax]
    for j in range(nc):
        ax[j].plot(mysamples[:, j, 0])
        ax[j].grid(which='both')
    plt.tight_layout()
    plt.savefig(fpath + '/trace_sigma.png')
    plt.close()
    
    for i in range(1, ndim):
        fig, ax = plt.subplots(nc, 1, figsize=(5, nc*3), sharex=True)
        if nc == 1: ax = [ax]
        for j in range(nc):
            ax[j].plot(mysamples[:, j, i])
            ax[j].grid(which='both')
        plt.tight_layout()
        plt.savefig(fpath + '/trace_alpha%d.png'%i)
    plt.close()
            

        
def plot_scatter(mysamples, fpath, sub=100):
    #
    ns, nc, ndim = mysamples.shape
    fig, ax = plt.subplots(ndim-1, 1, figsize = (5, ndim*3-3), sharex=True, sharey=True)
    if ndim == 2: ax = [ax]
    for i in range(1, ndim):
        ax[i-1].scatter(mysamples[::sub,..., 0], mysamples[::sub,..., i].flatten(), marker='.')
        ax[i-1].grid(which='both')
    plt.tight_layout()
    plt.savefig(fpath + '/scatter.png')
    plt.close()



def plot_autorcc(mysamples, fpath):
    #
    ns, nc, ndim = mysamples.shape

    def get_rcc(x):
        xp = (x - x.mean(axis=0))/x.std(axis=0)
        rcc = np.array([np.correlate(xp[:, i], xp[:, i], mode='full') for i in range(xp.shape[1])])
        rcc = rcc[:, rcc.shape[1]//2:].T
        rcc /= rcc[0]
        tcc = []
        for j in range(x.shape[1]):
            for m in range(500):
                if m > 5*(1 + 2*rcc[:m, j].sum()): break
            tcc.append((1 + 2*rcc[:m, j].sum()))
        tcc = np.array(tcc)
        return rcc, tcc

    
    fig, ax = plt.subplots()
    rcc, tcc = get_rcc(mysamples[..., 0])
    plt.plot(rcc[:500])
    plt.grid()
    try: plt.title('%d ($\pm$%d)'%(tcc.mean(), tcc.std()))   
    except: pass
    plt.savefig(fpath + '/rcc_sigma.png')
    plt.close()

    
    fig, ax = plt.subplots(ndim-1, 1, figsize = (5, ndim*3-3), sharex=True, sharey=True)
    if ndim == 2: ax = [ax]
    for i in range(1, ndim):
        rcc, tcc = get_rcc(mysamples[...,i])
        ax[i-1].plot(rcc[:500])
        ax[i-1].grid()
        try: ax[i-1].set_title('%d ($\pm$%d)'%(tcc.mean(), tcc.std()))
        except : pass
    plt.savefig(fpath + '/rcc_alpha.png')
    plt.close()
