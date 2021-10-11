import numpy as np
import os, sys

########################################## 
steps = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
facs = [2, 5, 10]
subs = [2, 3, 4, 5, 6, 7, 8]
nsteps = len(steps)
nfacs = len(facs)
nsubs = len(subs)
lpath = 30
ndim = 100 

nskip = 1 

def combine(fpath, ndim=None):
    print("Combining %s"%fpath)
    for ftype in ['samples', 'counts', 'accepted']:
        ss = []
        for i in range(50):
            if ndim is not None:
                ss.append(np.load(fpath + '/%s/%d.npy'%(ftype, i))[..., :ndim])
            else:
                ss.append(np.load(fpath + '/%s/%d.npy'%(ftype, i)))
        ss = np.squeeze(np.array(ss))
        if len(ss.shape) == 3: 
            ss = np.transpose(ss, (1, 0, 2))
        else: ss = np.transpose(ss, (1, 0))
        print(ftype, ss.shape)
        if ftype != 'accepts': np.save(fpath + '/%s.npy'%ftype, ss)
        else:  np.save(fpath + '/%s.npy'%'accepted', ss)

    
fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long/stoch_voltality-hmctnuts/'
for istep, step in enumerate([10, 20, 50, 5]):    
    fpath = fpath0 + 'step%02d/'%(step)
    ftype = 'samples'
    ndim = 50
    print(fpath)
    try:
        if os.path.isfile(fpath + '/%s.npy'%ftype):
            ss = np.load(fpath + '/%s.npy'%ftype)
            print("File found ", fpath + '/%s.npy'%ftype) 
            if ss.shape[0] < 10000: 
                print('Redo since shape is only ', ss.shape)
                combine(fpath, ndim=ndim)
        elif os.path.isfile(fpath + '/samples/00.npy'): 
            print('Redo', step)
            combine(fpath, ndim=ndim)
        elif os.path.isfile(fpath + '/samples/0.npy'): 
            print('Redo', step)
            combine(fpath, ndim=ndim)
        else:
            print(fpath + '/samples/00.npy Not found') 
    except Exception as e: print(e)
##    for _, fac in enumerate([2, 5, 10]):
##        for nsub in [2, 3, 4]:
##            fpath = fpath0 + 'step%02d_fac%02d_nsub%d/'%(step, fac, nsub)
##            print(fpath)
##            try:
##                if False : #os.path.isfile(fpath + '/%s.npy'%ftype):
##                    ss = np.load(fpath + '/%s.npy'%ftype)
##                    print("File found ", fpath + '/%s.npy'%ftype) 
##                    if ss.shape[0] < 10000: 
##                        print('Redo since shape is only ', ss.shape)
##                        combine(fpath, ndim=ndim)
##                else:# os.path.isfile(fpath + '/samples/00.npy'): 
##                    print('Redo', step)
##                    combine(fpath, ndim=ndim)
##            except Exception as e: print(e)
##

sys.exit()
##for istep, step in enumerate([0.02]):    
##    for lpath in [10]:
##        Nleapfrog = int(lpath / step)
##        Nleapfrog = max(10, Nleapfrog)
##        fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long/stoch_voltality//'
##        fpath = fpath0 + 'step%03d_nleap%02d/'%(step*100, Nleapfrog)
##        #for ftype in ['samples', 'counts', 'accepts', 'ptries', 'probs']:
##        for ftype in ['samples']:
##            try: 
##                if os.path.isfile(fpath + '/%s.npy'%ftype):
##                    pass #continue
##                if os.path.isfile(fpath + '/samples/00.npy'): 
##                    print('Redo', step, lpath)
##                    combine(fpath)
##                    #ss = []
##                    #for i in range(50):
##                    #    ss.append(np.load(fpath + '/%s/%02d.npy'%(ftype, i)))
##                    #ss = np.squeeze(np.array(ss))
##                    #if len(ss.shape) == 3: ss = np.transpose(ss, (1, 0, 2))
##                    #else: ss = np.transpose(ss, (1, 0))
##                    #ss = ss[::nskip]
##                    #print(ftype, ss.shape)
##                    #if ftype != 'accepts': 
##                    #    np.save(fpath + '/%s.npy'%ftype, ss)
##                    #else:  np.save(fpath + '/%s.npy'%'accepted', ss)
##            except Exception as e:
##                print(e)
##
##
for istep in range(nsteps):    
    Nleapfrog = int(lpath / steps[istep])
    Nleapfrog = max(10, Nleapfrog)
    fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long3/funnel//Ndim%02d/'%ndim
    for fac in facs:
        for isub, sub in enumerate(subs):
            print(steps[istep], fac, sub)
            fpath = fpath0 + 'step%03d_nleap%02d_fac%02d_nsub%d/'%(steps[istep]*100, Nleapfrog, fac, sub)
            key = 'step %0.3f//%d-%d'%(steps[istep],  fac, sub)
            for ftype in ['samples', 'counts', 'accepts', 'ptries', 'probs']:
                try: 
                    if os.path.isfile(fpath + '/%s.npy'%ftype):
                        ss = np.load(fpath + '/%s.npy'%ftype)
                        print(ftype, ss.shape)
                        if ss.shape[0] == 100000:
                            print('skip')
                            continue
                    else:
                        print('Redo')
                        ss = []
                        for i in range(50):
                            ss.append(np.load(fpath + '/%s/%02d.npy'%(ftype, i)))
                        ss = np.squeeze(np.array(ss))
                        if len(ss.shape) == 3: ss = np.transpose(ss, (1, 0, 2))
                        else: ss = np.transpose(ss, (1, 0))
                        ss = ss[::nskip]
                        print(ftype, ss.shape)
                        if ftype != 'accepts': np.save(fpath + '/%s.npy'%ftype, ss)
                        else:  np.save(fpath + '/%s.npy'%'accepted', ss)
                except Exception as e:
                    print(e)



##for istep in range(nsteps):    
##    Nleapfrog = int(lpath / steps[istep])
##    Nleapfrog = max(10, Nleapfrog)
##    fpath0 = '/mnt/ceph/users/cmodi/hmc/outputs_long3/funnel//Ndim%02d/'%ndim
##    for fac in facs:
##        for isub, sub in enumerate(subs):
##            print(steps[istep], fac, sub)
##            fpath = fpath0 + 'step%03d_nleap%02d_fac%02d_nsub%d/'%(steps[istep]*100, Nleapfrog, fac, sub)
##            key = 'step %0.3f//%d-%d'%(steps[istep],  fac, sub)
##            for ftype in ['samples', 'counts', 'accepts', 'ptries', 'probs']:
##                try: 
##                    ss = np.load(fpath + '/%s.npy'%ftype)
##                    print(ftype, ss.shape)
##                    if ss.shape[0] == 100000:
##                        print('skip')
##                        continue
##                    else:
##                        print('Redo')
##                        ss = []
##                        for i in range(50):
##                            ss.append(np.load(fpath + '/%s/%02d.npy'%(ftype, i)))
##                        ss = np.squeeze(np.array(ss))
##                        if len(ss.shape) == 3: ss = np.transpose(ss, (1, 0, 2))
##                        else: ss = np.transpose(ss, (1, 0))
##                        ss = ss[::nskip]
##                        print(ftype, ss.shape)
##                        if ftype != 'accepts': np.save(fpath + '/%s.npy'%ftype, ss)
##                        else:  np.save(fpath + '/%s.npy'%'accepted', ss)
##                except Exception as e:
##                    print(e)
##

