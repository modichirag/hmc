import numpy as np


class PyHMC():
    
    def __init__(self, log_prob, grad_log_prob, KE=None, KE_g=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x : self.log_prob(x)*-1.
        #self.V_g = lambda x : self.grad_log_prob(x)*-1.
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

        if KE is None or KE_g is None:
            self.KE = self.unit_norm_KE
            self.KE_g = self.unit_norm_KE_g
        
    
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
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = np.random.normal(size=q.size).reshape(q.shape)
        q1, p1 = self.leapfrog(q, p, N, step_size)
        q, p, accepted, prob = self.metropolis([q, p], [q1, p1])
        return q, p, accepted, prob, [self.Hcount, self.Vgcount, self.leapcount]

##


class PyHMC_2step():
    
    def __init__(self, log_prob, grad_log_prob, KE=None, KE_g=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x : self.log_prob(x)*-1.
        self.V_g = lambda x : self.grad_log_prob(x)*-1.
        
        if KE is None or KE_g is None:
            self.KE = self.unit_norm_KE
            self.KE_g = self.unit_norm_KE_g
            
    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()

    def unit_norm_KE_g(self, p):
        return p

    def H(self, q,p):
        return self.V(q) + self.KE(p)


    def leapfrog(self, q, p, N, step_size):
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



    def hmc_step(self, q, N, step_size, two_factor):
        p = np.random.normal(size=q.size).reshape(q.shape)
        q1, p1 = self.leapfrog(q, p, N, step_size)
        accepted = False
        H0 = self.H(q, p)
        H1 = self.H(q1, p1)
        prob1 = np.exp(H0 - H1)
        if np.isnan(prob1) or np.isinf(prob1) or (q-q1).sum()==0:
            prob1 = 0.   ##since prob1 = 1. if q == q1
            accepted = False
        elif np.random.uniform(0., 1., size=1) > min(1., prob1):
            accepted = False
        else:
            accepted = True
        #
        if accepted:
            return q1, p1, 1., [H0, H1, H0, H1]
        else:
            N2 = int(N*two_factor)
            s2 = step_size/two_factor
            q2, p2 = self.leapfrog(q, p, N2, s2)
            H2 = self.H(q2, p2)
            prob2 = np.exp(H0 - H2)
            if np.isnan(prob2) or np.isinf(prob2) or (q-q2).sum()==0: 
                return q, p, -1., [H0, H1, H2, H1]
                accepted = False
            else:
                q21, p21 = self.leapfrog(q2, -p2, N, step_size)
                H21 = self.H(q21, p21)
                prob21 = np.exp(H2 - H21)
                
                #if np.isnan(prob1): prob1 = 0.
                #if np.isnan(prob21) or np.isinf(prob21): 
                if prob1 == 1:
                    import sys
                    print("prob1 should not be 1")
                    #sys.exit()
                prob = prob2 * (1.-prob21)/(1.-prob1)
                
                if np.isnan(prob)  :
                    return q, p, -1, [H0, H1, H2, H21]
                elif np.random.uniform(size=1) > min(1., prob):
                    return q, p, 0., [H0, H1, H2, H21]
                else:
                    return q2, p2, 2., [H0, H1, H2, H21]
                






class PyHMC_multistep():
    
    def __init__(self, log_prob, grad_log_prob, KE=None, KE_g=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x : self.log_prob(x)*-1.
        #self.V_g = lambda x : self.grad_log_prob(x)*-1.
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

        if KE is None or KE_g is None:
            self.KE = self.unit_norm_KE
            self.KE_g = self.unit_norm_KE_g
        
    
    def V_g(self, x):
        self.Vgcount += 1
        return self.grad_log_prob(x)*-1.
    
    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()

    def unit_norm_KE_g(self, p):
        return p

    def H(self, q, p):
        self.Hcount +=1
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
            #print(e)
            return q0, p0


    def get_num(self, m, q, p, N, ss, fsub): 

        avec = np.zeros(m)
        H0 = self.H(q, p)    
        for j in range(m):
            fac = fsub**(j)
            qj, pj = self.leapfrog(q, p, int(N*fac), ss/fac)
            Hj = self.H(qj, pj)
            pfac = np.exp(H0 - Hj)
            if  (q - qj).sum()==0: 
                pfac = 0.
            if j:
                den = np.prod(1-avec[:j])
                num = self.get_num(j, qj, -pj, N, ss, fsub)
                prob = pfac*num/den
            else: 
                prob = pfac
            if np.isnan(prob) or np.isinf(prob): 
                return 0. #np.nan
            else: avec[j] = min(1., prob)
            if np.prod(1-avec): pass
            else: 
                return np.prod(1-avec)
        return np.prod(1-avec)


    def multi_step(self, m, q0, N, ss, fsub):

        
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p0 = np.random.normal(size=q0.size).reshape(q0.shape)
        avec = np.zeros(m)
        H0 = self.H(q0, p0)
        q1, p1 = self.leapfrog(q0, p0, N, ss)
        H1 = self.H(q1, p1)
        pfac = np.exp(H0 - H1)
        prob = pfac
        if np.isnan(prob) or np.isinf(prob) or (q0 - q1).sum()==0: 
            prob = 0.
        prob = min(1., prob)
        avec[0] = prob
        acc = np.random.uniform()
        if  acc <= avec[0]: 
            return q1, p1, 0, avec, [self.Hcount, self.Vgcount, self.leapcount]
        else:
            for j in range(1, m):
                fac = fsub**(j)
                qj, pj = self.leapfrog(q0, p0, int(N*fac), ss/fac)
                Hj = self.H(qj, pj)
                pfac = np.exp(H0 - Hj)
                if  (q0 - qj).sum()==0:
                    pfac = 0.
                den = np.prod(1-avec[:j])
                num = self.get_num(j, qj, -pj, N, ss, fsub)
                prob = pfac*num/den
                if np.isnan(prob) or np.isinf(prob): prob = 0.
                avec[j] = min(1., prob)
                acc = np.random.uniform()
                if acc < avec[j]: 
                    return qj, pj, j, avec, [self.Hcount, self.Vgcount, self.leapcount]
            return q0, p0, -1, avec, [self.Hcount, self.Vgcount, self.leapcount]


