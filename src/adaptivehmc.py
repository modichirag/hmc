import numpy as np



class HMC():
    
    def __init__(self, log_prob, grad_log_prob, invmetric_diag=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x : self.log_prob(x)*-1.
        #self.V_g = lambda x : self.grad_log_prob(x)*-1.
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag).sum()
        self.KE_g = lambda p: p * self.invmetric_diag
    
    def V_g(self, x):
        self.Vgcount += 1
        return self.grad_log_prob(x)*-1.
        
    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()

    def unit_norm_KE_g(self, p):
        return p

    def H(self, q,p):
        self.Hcount += 1
        return self.V(q) + self.KE(p)

    def leapfrog(self, q, p, N, step_size):
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            p = p - 0.5*step_size * self.V_g(q) 
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q) 
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q) 
            return q, p
        except Exception as e:
            print(e)
            return q0, p0

    def leapfrog1(self, q, p, step_size, Vgq=None): #This needs to be optimized to not estimate V_g again and again
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            if Vgq is None: Vgq = self.V_g(q)
            p = p - 0.5*step_size * Vgq
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q) 
            return q, p, Vgq
        except Exception as e:
            print(e)
            return q0, p0, Vgq

    def metropolis(self, qp0, qp1):
        q0, p0 = qp0
        q1, p1 = qp1
        H0 = self.H(q0, p0)
        H1 = self.H(q1, p1)
        prob = np.exp(H0 - H1)
        #prob = min(1., np.exp(H0 - H1))
        if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
            return q0, p0, 2., [H0, H1]
        elif np.random.uniform(0., 1., size=1) > min(1., prob):
            return q0, p0, 0., [H0, H1]
        else: return q1, p1, 1., [H0, H1]


    def hmc_step(self, q, N, step_size):
        '''Single hmc iteration
        
        Parameters:
        ----------
        q: initial position
        N: number of leapfrog steps
        step_size: step size for leapfrog iteration

        Returns:
        --------
        A tuple of- 
        q
        p
        accepted (0/1/2)
        acceptance probability
        list of [Hcounts, Vcounts, nleapfrogs]
        '''
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
        q1, p1 = self.leapfrog(q, p, N, step_size)
        q, p, accepted, prob = self.metropolis([q, p], [q1, p1])
        return q, p, accepted, prob, [self.Hcount, self.Vgcount, self.leapcount]


 
######################
class AdHMC_eps0(HMC):
    
    def __init__(self, log_prob, grad_log_prob, invmetric_diag=None):
        super().__init__(log_prob, grad_log_prob, invmetric_diag)


    def get_stepsize(self, q0, p0, smin=0.01, smax=1.0, ntry=20, logspace=True, nsteps=1, eps=None):
        H0 = self.H(q0, p0)
        Hs = np.zeros(ntry)
        if logspace: steps = np.logspace(np.log10(smin), np.log10(smax), ntry)
        else:  steps = np.linspace(smin, smax, ntry)
        pwts = steps.copy()**0.5 #np.linspace(0.9, 1.1, steps.size)
        for iss, ss in enumerate(steps):
            #nsteps = int(steps.max()/ss)+1
            q1, p1 = self.leapfrog(q0, p0, nsteps, ss)
            Hs[iss] = self.H(q1, p1)
        pp = np.exp(H0 - Hs) * pwts
        pp[np.isnan(pp)] = 0 
        pp[np.isinf(pp)] = 0 
        pp /= pp.sum()
        cdf = np.cumsum(pp)
        if eps is None:
            sx = np.random.uniform(low=cdf.min()) 
            isx = np.where(sx > cdf)[0][-1]
            sx2 = np.random.uniform(steps[isx], steps[isx+1])
            prob = pp[isx+1] # * 1/(steps[isx+1]-steps[isx+1])
            return sx2, pp[isx+1]
        else: 
            prob = pp[np.where(steps > eps)[0][0]]
            return prob



    def hmc_step(self, q0, Nleap, smin=0.01, smax=1.0, Tint=0, ntry=10, nsteps=1):
        '''Single hmc iteration
        
        Parameters:
        ----------
        q: initial position
        N: number of leapfrog steps
        step_size: step size for leapfrog iteration
        smin: Minimum allowed step size
        smin: Maximum allowed step size
        Tint: Time of integration
        ntry: Number of points to try for estimating first step size
        nsteps: Number of steps per try for estimating first step size

        Returns:
        --------
        A tuple of- 
        q
        p
        accepted (0/1/2)
        acceptance probability
        array of [pfactor denominator, pfactor numberator, stepsize]
        list of [Hcounts, Vcounts, nleapfrogs]
        '''
        
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p0 = np.random.normal(size=q0.size).reshape(q0.shape) * self.metricstd
        H0 = self.H(q0, p0)
        if (Tint == 0) and (Nleap == 0):
            print("Tint and Nleap cannot be both zeros")
            import sys
            sys.exit()
        elif (Tint != 0) and (Nleap != 0):
            print("Tint and Nleap both given and are inconsistent")
            import sys
            sys.exit()
        #First step is drawn from a distribution
        ss, pf_den = self.get_stepsize(q0, p0, smin, smax, ntry=ntry, nsteps=nsteps)
        eps = ss
        if Tint == 0: N = Nleap
        else: N = int(Tint/eps) + 1
        #print("Steps size is %0.2f, and number of steps is %d"%(eps, N))
        q1, p1 = self.leapfrog(q0, p0, N, ss)
        H1 = self.H(q1, p1)        
        pb_num = self.get_stepsize(q1, -p1, smin=smin, smax=smax, eps=ss, ntry=ntry, nsteps=nsteps)

        hastings_factor = pb_num/pf_den
        prob = np.exp(H0 - H1) * hastings_factor
        #print("prb, fac, metrop : ", prob, adfac, prob/adfac, pb_num, pf_den)
        toret = [[prob, prob/hastings_factor, hastings_factor], np.stack([pf_den, pb_num, eps]), [self.Hcount, self.Vgcount, self.leapcount]]
        if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
            return q0, p0, 2., *toret
        elif np.random.uniform(0., 1., size=1) > min(1., prob):
            return q0, p0, 0., *toret
        else: return q1, p1, 1., *toret 
 
##
######################
class AdHMC(HMC):
    
    def __init__(self, log_prob, grad_log_prob, invmetric_diag=None):
        super().__init__(log_prob, grad_log_prob, invmetric_diag)


    def get_stepsize(self, q0, p0, smin=0.01, smax=1.0, ntry=10, logspace=True, nsteps=1, eps=None):
        H0 = self.H(q0, p0)
        Hs = np.zeros(ntry)
        if logspace: steps = np.logspace(np.log10(smin), np.log10(smax), ntry)
        else:  steps = np.linspace(smin, smax, ntry)
        pwts = steps.copy()**0.5 #np.linspace(0.9, 1.1, steps.size)
        for iss, ss in enumerate(steps):
            #nsteps = int(steps.max()/ss)+1
            q1, p1 = self.leapfrog(q0, p0, nsteps, ss)
            Hs[iss] = self.H(q1, p1)
        pp = np.exp(H0 - Hs) * pwts
        pp[np.isnan(pp)] = 0 
        pp[np.isinf(pp)] = 0 
        pp /= pp.sum()
        cdf = np.cumsum(pp)
        if eps is None:
            sx = np.random.uniform(low=cdf.min()) 
            isx = np.where(sx > cdf)[0][-1]
            sx2 = np.random.uniform(steps[isx], steps[isx+1])
            prob = pp[isx+1] # * 1/(steps[isx+1]-steps[isx+1])
            return sx2, pp[isx+1]
        else: 
            prob = pp[np.where(steps > eps)[0][0]]
            return prob


    def hmc_step(self, q0, Nleap=100, nleap=10, ratios= [1/np.sqrt(2), np.sqrt(2)], pwts0 = [1., 1.], smin=0.01, smax=1.0, ntry_eps0=10, nsteps_eps0=1, logeps=True, verbose=False):
        '''
        Parameters:
        ----------
        q: initial position
        Nleap: number of leapfrog steps
        nleap: number of leapfrog steps to adapt step size
        smin: Minimum allowed step size
        smin: Maximum allowed step size
        ratios: ratio to change step size with after nleap steps- expected in INCREASING order
        ntry_eps0: Number of points to try for estimating first step size
        nsteps_eps0: Number of steps per try for estimating first step size

        Returns:
        --------
        A tuple of- 
        q
        p
        accepted (0/1/2)
        list of probabiliies [acc_prob, acc_prob/hastings_factor, hastings_factor]
        array of checks [pfactor denominator, pfactor numberator, stepsize]
        list of counts [Hcounts, Vcounts, nleapfrogs]
        '''

        #normprob is not implemented
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p0 = np.random.normal(size=q0.size).reshape(q0.shape) * self.metricstd
        N = int(Nleap//nleap)
        #First step is drawn from a distribution
        eps, pf_den, pb_num = np.zeros(N), np.zeros(N), np.zeros(N)
        #pwts0 = 1. #np.array([0.9, 1.0,  1.1])
        nr = len(ratios)
        H0 = self.H(q0, p0)
        
        #First step is drawn from a distribution
        ss, pf_den[0] = self.get_stepsize(q0, p0, smin, smax, ntry=ntry_eps0, nsteps=nsteps_eps0, logspace=logeps)
        eps[0] = ss
        q1, p1 = self.leapfrog(q0, p0, nleap, ss)
        H1 = self.H(q1, p1)
        Hprev = H0
        sigma = np.log(0.5)/2.

        for i in range(N - 1):
            #q1, p1, H1 is the current position.
            #ss is the current step size i.e. the last taken

            ##Forward
            pf, pb = np.zeros(nr), np.zeros(nr)
            qs, ps, Hs = [], [], []
            for j in range(nr):
                ss2 = ss*ratios[j]
                q, p= self.leapfrog(q1, p1, nleap, ss2)
                qs.append(q)
                ps.append(p)
                Hs.append(self.H(q, p))
                pH = np.exp(H1 - Hs[-1])
                if np.isnan(pH) or np.isinf(pH): pH = 0
                pf[j] = pH

            pwts = np.ones(nr) * pwts0
            if smin > ss*ratios[0]: pwts[0] = 0
            if smax < ss*ratios[-1]: pwts[-1] = 0
            pf *= pwts
            pfraw = pf.copy()
            pf /= pf.sum()
            if np.isnan(pf.sum()) or np.isinf(pf.sum()): 
                if verbose: 
                    print("Something blew up so returning initial position")
                    print(pfraw, pwts, pf, Hs, ss)
                return q0, p0, 100+i, [np.NaN, np.NaN, np.NaN], np.stack([pf_den, pb_num, eps]), [self.Hcount, self.Vgcount, self.leapcount]
            #Select a step size to carry forth
            pind = np.random.choice(nr, p=pf)
            ssnew = ss*ratios[pind]

            q2, p2, H2 = qs[pind], ps[pind], Hs[pind]
            pf_den[i+1] = pf[pind]
 
            ##Reverse
            #step from q1, p1 if we arrive here with ssnew step size in reverse direction
            Hsb = []
            for j in range(nr):
                if np.allclose(ssnew*ratios[j] , ss):
                    Hsb.append(Hprev)
                    pbind = j
                else:
                    ss2 = ssnew*ratios[j]
                    q, p = self.leapfrog(q1, -p1, nleap, ss2)
                    Hsb.append(self.H(q, p))
                pH =  np.exp(H1 - Hsb[-1])
                if np.isnan(pH) or np.isinf(pH): pH = 0
                pb[j] = pH
            pwts = np.ones(nr) *pwts0
            if smin > ssnew*ratios[0]: pwts[0] = 0
            if smax < ssnew*ratios[-1]: pwts[-1] = 0
            pb *= pwts
            pb /= pb.sum()
            pb_num[i] = pb[pbind]

            #setup for next step
            eps[i+1] = ssnew
            ss = ssnew
            Hprev = H1
            q1, p1, H1 = q2, p2, H2
            #print(pf, pb, pf[pind], pb[pbind])
            #print(ss)

        #print('started and ended with step sizes ', eps[0], eps[i+1])
        #print('Number of steps taken is %d out of %d'%(i, N))
        if (ssnew > smin) and (ssnew < smax):
            #pb_num[-1] = self.get_stepsize(q2, -p2, smin=smin, smax=smax, eps=ssnew, ntry=ntry, nsteps=nsteps)
            pb_num[i+1] = self.get_stepsize(q2, -p2, smin=smin, smax=smax, eps=ssnew, ntry=ntry_eps0, nsteps=nsteps_eps0, logspace=logeps)

        hastings_factor = np.prod(pb_num[:i+2])/np.prod(pf_den[:i+2])
        prob = np.exp(H0 - H2) * hastings_factor
        if verbose: print("prb, fac, metrop : ", prob, hastings_factor, prob/hastings_factor, pb_num[-1], pf_den[0])
        #Return
        toret = [[prob, prob/hastings_factor, hastings_factor], np.stack([pf_den, pb_num, eps]), [self.Hcount, self.Vgcount, self.leapcount]]
        if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
            return q0, p0, 2., *toret
        elif np.random.uniform(0., 1., size=1) > min(1., prob):
            return q0, p0, 0., *toret
        else: 
            return q2, p2, 1., *toret
 
##



######################
######################
class AdHMC_May(HMC):
    ##DO NOT DELETE. HAS normprob and other methods implemented that need to be moved above
    def __init__(self, log_prob, grad_log_prob, invmetric_diag=None):
        super.__init__(log_prob, grad_log_prob, invmetric_diag)


    def get_stepsize(self, q0, p0, smin=0.01, smax=1.0, ntry=20, logspace=True, nsteps=1, eps=None):
        H0 = self.H(q0, p0)
        Hs = np.zeros(ntry)
        if logspace: steps = np.logspace(np.log10(smin), np.log10(smax), ntry)
        else:  steps = np.linspace(smin, smax, ntry)
        pwts = steps.copy()**0.5 #np.linspace(0.9, 1.1, steps.size)
        for iss, ss in enumerate(steps):
            #nsteps = int(steps.max()/ss)+1
            q1, p1 = self.leapfrog(q0, p0, nsteps, ss)
            Hs[iss] = self.H(q1, p1)
        pp = np.exp(H0 - Hs) * pwts
        pp[np.isnan(pp)] = 0 
        pp[np.isinf(pp)] = 0 
        pp /= pp.sum()
        cdf = np.cumsum(pp)
        if eps is None:
            sx = np.random.uniform(low=cdf.min()) 
            isx = np.where(sx > cdf)[0][-1]
            sx2 = np.random.uniform(steps[isx], steps[isx+1])
            prob = pp[isx+1] # * 1/(steps[isx+1]-steps[isx+1])
            return sx2, pp[isx+1]
        else: 
            prob = pp[np.where(steps > eps)[0][0]]
            return prob


    def hmc_step(self, q0, Nleap, smin=0.01, smax=1.0, ratios= [0.75, 1.0, 1/0.75], Tint=0, ntry=20, nsteps=1, normprob=True):
        
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p0 = np.random.normal(size=q0.size).reshape(q0.shape) * self.metricstd
        
        if (Tint == 0) and (Nleap == 0):
            print("Tint and Nleap cannot be both zeros")
            import sys
            sys.exit()
        elif (Tint != 0) and (Nleap != 0):
            print("Tint and Nleap both given and are inconsistent")
            import sys
            sys.exit()
        if Tint == 0: N = Nleap
        else: N = int(Tint/smin)

        #First step is drawn from a distribution
        eps, pf_den, pb_num = np.zeros(N), np.zeros(N), np.zeros(N)
        pwts0 = 1. #np.array([0.9, 1.0,  1.1])
        nr = len(ratios)
        H0 = self.H(q0, p0)


        def halfleap(q, p, step_size, Vgq=None): #This needs to be optimized to not estimate V_g again and again
            self.leapcount += 1 
            q0, p0 = q, p
            if Vgq is None: Vgq = self.V_g(q)
            p = p - 0.5*step_size * Vgq
            q = q + step_size * self.KE_g(p)
            return q, p, Vgq

        
        #First step is drawn from a distribution
        ss, pf_den[0] = self.get_stepsize(q0, p0, smin, smax, ntry=ntry, nsteps=nsteps)
        eps[0] = ss
        q1, p1, _ = self.leapfrog1(q0, p0, ss)
        H1 = self.H(q1, p1)
        Hprev = H0
        sigma = np.log(0.5)/2.
        #print('Doing half step to estimate goodness')
        for i in range(N-1):
            #q1, p1, H1 is the current position.
            #ss is the current step size i.e. the last taken

            #Forward
            pf, pb = np.zeros(nr), np.zeros(nr)
            qs, ps, Hs = [], [], []
            Vgq = None
            for j in range(nr):
                ss2 = ss*ratios[j]
                q, p, Vgq = self.leapfrog1(q1, p1, ss2, Vgq)
                #q, p, Vgq = halfleap(q1, p1, ss2, Vgq)
                qs.append(q)
                ps.append(p)
                Hs.append(self.H(q, p))
                #if normprob: pH = np.exp(-0.5 * (Hs[-1] - Hprev)**2 / sigma**2) 
                #if normprob: pH = np.exp(- abs(Hs[-1] - Hprev)) 
                if normprob: pH = np.exp(- abs(Hs[-1] - H1)) 
                else: pH = np.exp(H1 - Hs[-1])
                if np.isnan(pH) or np.isinf(pH): pH = 0
                pf[j] = pH
            pwts = np.ones(nr) * pwts0
            if smin > ss*ratios[0]: pwts[0] = 0
            if smax < ss*ratios[-1]: pwts[-1] = 0
            pf *= pwts
            pf /= pf.sum()
            if np.isnan(pf.sum()) or np.isinf(pf.sum()): 
                return q0, p0, 100+i, 0, np.stack([pf_den, pb_num, eps]), [self.Hcount, self.Vgcount, self.leapcount]
            pind = np.random.choice(nr, p=pf)
            ssnew = ss*ratios[pind]
            q2, p2, _ = self.leapfrog1(q1, p1, ssnew, Vgq)
            H2 = self.H(q2, p2)
            #q2, p2, H2 = qs[pind], ps[pind], Hs[pind]
            pf_den[i+1] = pf[pind]
 
            #step from q1, p1 if we arrive here with ssnew step size in reverse direction
            Hsb = []
            for j in range(nr):
                if np.allclose(ssnew*ratios[j] , ss):
                    Hsb.append(Hprev)
                    pbind = j
                else:
                    ss2 = ssnew*ratios[j]
                    q, p, Vgq = self.leapfrog1(q1, -p1, ss2, Vgq)
                    Hsb.append(self.H(q, p))
                #if normprob: pH = np.exp(-0.5 * (Hsb[-1] - H2)**2 / sigma**2) 
                #if normprob: pH = np.exp(- abs(Hsb[-1] - H2)) 
                if normprob: pH = np.exp(- abs(Hsb[-1] - H1)) 
                else: pH =  np.exp(H1 - Hsb[-1])
                if np.isnan(pH) or np.isinf(pH): pH = 0
                pb[j] = pH
            pwts = np.ones(nr) *pwts0
            if smin > ssnew*ratios[0]: pwts[0] = 0
            if smax < ssnew*ratios[-1]: pwts[-1] = 0
            pb *= pwts
            pb /= pb.sum()
            pb_num[i] = pb[pbind]

            #setup for next step
            eps[i+1] =ssnew # ratios[pind]
            ss = ssnew
            Hprev = H1
            q1, p1, H1 = q2, p2, H2
            #print(pf, pb, pf[pind], pb[pbind])
            #print(ss)
            ##THIS VIOLATES DB BUT LETS SEE HOW BAD THIS IS
            ##THIS MIGHT BE RELATED TO NUTS GOING IN BOTH DIRECTIONS
            ##CONSIDER SEQ OF STEPSIZES 0.5,1,2,3,6 with TINT=10, AND CASE WHEN WE START FROM 1 OR 3
            #if Tint > 0 : 
            #    if eps[:i+1].sum() > Tint : break 

        #print('started and ended with step sizes ', eps[0], eps[i+1])
        #print('Number of steps taken is %d out of %d'%(i, N))
        if (ssnew > smin) and (ssnew < smax):
            #pb_num[-1] = self.get_stepsize(q2, -p2, smin=smin, smax=smax, eps=ssnew, ntry=ntry, nsteps=nsteps)
            pb_num[i+1] = self.get_stepsize(q2, -p2, smin=smin, smax=smax, eps=ssnew, ntry=ntry, nsteps=nsteps)

        adfac = np.prod(pb_num[:i+2])/np.prod(pf_den[:i+2])
        prob = np.exp(H0 - H2) * adfac
        #print("prb, fac, metrop : ", prob, adfac, prob/adfac, pb_num[-1], pf_den[0])
        if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
            return q0, p0, 2., prob, np.stack([pf_den, pb_num, eps]), [self.Hcount, self.Vgcount, self.leapcount]
        elif np.random.uniform(0., 1., size=1) > min(1., prob):
            return q0, p0, 0., prob, np.stack([pf_den, pb_num, eps]), [self.Hcount, self.Vgcount, self.leapcount]
        else: return q2, p2, 1., prob, np.stack([pf_den, pb_num, eps]), [self.Hcount, self.Vgcount, self.leapcount]
 
##






