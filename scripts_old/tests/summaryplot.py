import numpy as np
import os
import matplotlib.pyplot as plt 
import json

fpath = '/mnt/ceph/users/cmodi/hmc/outputs_adaptive/funnel/Ndim05/'
#fpath = '/mnt/ceph/users/cmodi/hmc/outputs_adaptive/funnel/Ndim50/'

configs = [x[0] for x in os.walk(fpath)]
configs = os.listdir(fpath)


v = []
for i in range(len(configs)):
    print(configs[i])
    with open(fpath + configs[i] + "/checks.json", "r") as f:
        checks = json.load(f)
        mu, std, cc, cv = checks['mean'][0], checks['std'][0], checks['cost'][0], checks['costV_g'][0]
        count, azess = checks['counts'][0], checks['azess'][0]
        v.append([mu, std,  cv])

ntoplot = len(v[0])
iconfig = np.arange(len(v))

fig, axar = plt.subplots(1, ntoplot, figsize=(7 + 3*ntoplot, 15), sharey=True)

#for j in range(ntoplot):
#    for i in range(len(configs)):
#        axar[j].plot(v[i][j], iconfig[i], 'o')
#        if 'step' in configs[i]: 
#            axar[j].axhline(iconfig[i], color='k', ls="--")
#            axar[j].axvline(v[i][j], color='k', ls="--")
#
def mark(j, keys, counter, cc=None, skipkeys=[]):
    for i in range(len(configs)):
        if np.prod([key in configs[i] for key in keys]): 
            if not bool(np.sum([key in configs[i] for key in skipkeys])): 
                if j==0: lbls.append(configs[i])
                axar[j].plot(v[i][j], counter, 'o', color=cc)
                counter += 1
    axar[j].axhline(counter -0.5, color='b', ls=":")
    return counter

lbls = []
for j in range(ntoplot):
    counter = 0
    #fname = 'max_stepsize'
    #counter = mark(j, ['smaxp1', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.5', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])

    #fname = 'bias'
    #counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap10_', 'bias'], counter)
    #counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap10_', 'bias'], counter)
    
    #fname = 'nrefresh'
    #counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap5_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap20_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap5_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap20_'], counter, skipkeys=['bias'])
    #
    #fname = 'time_integration'
    #counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap10_'], counter, skipkeys=['bias'])
    #counter = mark(j, ['smaxp0.1','tint10', 'nleap10_'], counter, skipkeys=['bias'])
    #
    fname = 'step_ratio'
    counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap10_', 'r50'], counter, skipkeys=['bias'])
    counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap10_', 'r70'], counter, skipkeys=['bias'])
    counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap10_', 'r90'], counter, skipkeys=['bias'])
    counter = mark(j, ['smaxp0.1', 'nleap100', 'nleap10_', 'r95'], counter, skipkeys=['bias'])
    counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap10_', 'r50'], counter, skipkeys=['bias'])
    counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap10_', 'r70'], counter, skipkeys=['bias'])
    counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap10_', 'r90'], counter, skipkeys=['bias'])
    counter = mark(j, ['smaxp0.2', 'nleap100', 'nleap10_', 'r95'], counter, skipkeys=['bias'])

for j in range(ntoplot):
    for i in range(len(configs)):
        if 'step' in configs[i]: 
            if j==0: lbls.append(configs[i])
            axar[j].plot(v[i][j], counter, 'ko')
            axar[j].axvline(v[i][j], color='k', ls="--")

axar[0].set_yticks(np.arange(len(lbls)))
axar[0].set_yticklabels(lbls)
axar[0].axvline(0, color='r', ls="--")
axar[1].axvline(3, color='r', ls="--")
axar[0].set_xlabel('Mean', fontsize=12)
axar[1].set_xlabel('Scale', fontsize=12)
axar[2].set_xlabel('Cost=#gradV/ESS', fontsize=12)
for ax in axar: ax.grid(which='both')
for ax in axar[2:]: ax.semilogx()
plt.tight_layout()
plt.savefig('%s.png'%fname)
