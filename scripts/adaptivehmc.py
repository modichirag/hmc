import numpy as np



class AdHMC():
    
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


    def hmc_step(self, q0, N, smin=0.01, smax=1.0):
        
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p0 = np.random.normal(size=q0.size).reshape(q0.shape) * self.metricstd
        eps, pf_den, pb_num = np.zeros(N), np.zeros(N), np.zeros(N)
        ratios = [0.5, 1.0, 2.0]
        pwts = np.array([0.5, 1.0, 2.0])
        nr = len(ratios)
        H0 = self.H(q0, p0)

        #First step is random
        ss = np.random.uniform(smin, smax)
        eps[0] = ss
        pf_den[0] = 1/(smax-smin)
        q1, p1, _ = self.leapfrog1(q0, p0, ss)
        H1 = self.H(q1, p1)
        Hprev = H0
        
        for i in range(N-1):
            #q1, p1, H1 is the current position.
            #ss is the current step size i.e. the last taken

            #Forward
            pf, pb = np.zeros(nr), np.zeros(nr)
            qs, ps, Hs = [], [], []
            Vgq = None
            for j in range(nr):
                q, p, Vgq = self.leapfrog1(q1, p1, ss*ratios[j], Vgq)
                qs.append(q)
                ps.append(p)
                Hs.append(self.H(q, p))
                pf[j] = np.exp(H1 - Hs[-1])
            pf *= pwts
            pf /= pf.sum()
            pfc = np.cumsum(pf)
            pind = np.where(np.cumsum(pfc) > np.random.uniform())[0][0]
            q2, p2, H2 = qs[pind], ps[pind], Hs[pind]
            pf_den[i+1] = pf[pind]
            eps[i+1] = ratios[pind]
            ssnew = ss*ratios[pind]

            #step from q1, p1 if we arrive here with ssnew step size in reverse direction
            Hsb = []
            for j in range(nr):
                if ssnew*ratios[j] == ss:
                    Hsb.append(Hprev)
                    pbind = j
                else:
                    q, p, Vgq = self.leapfrog1(q1, -p1, ssnew*ratios[j], Vgq)
                    Hsb.append(self.H(q, p))
                pb[j] = np.exp(H1 - Hsb[-1])
            pb *= pwts
            pb /= pb.sum()
            pb_num[i] = pb[pbind]

            ss = ssnew
            Hprev = H1
            q1, p1, H1 = q2, p2, H2
            #print(pf, pb)
            #print(ss, ssnew)
        if (ssnew > smin) and (ssnew < smax):
            pb_num[-1] = 1/(smax - smin)
        
        adfac = np.prod(pb_num)/np.prod(pf_den)
        #print(pf_den)
        #print(pb_num)
        #print(eps, adfac)
        prob = np.exp(H0 - H2) * adfac
        if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
            return q0, p0, 2., [H0, H1]
        elif np.random.uniform(0., 1., size=1) > min(1., prob):
            return q0, p0, 0., [H0, H1]
        else: return q2, p2, 1., [H0, H1]

        #q, p, accepted, prob = self.metropolis([q, p], [q2, p2])
        #return q, p, accepted, prob, [self.Hcount, self.Vgcount, self.leapcount]

##

