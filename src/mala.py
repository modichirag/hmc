import numpy as np


class MALA():
    
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

    def metropolis(self, q0, q1, V_q0, V_q1, V_gq0, V_gq1, eps):
        p0 = np.exp(V_q0 - V_q1)
        mu0 = q0 - 0.5*eps**2 * V_gq0
        mu1 = q1 - 0.5*eps**2 * V_gq1
        logden = ( -0.5* np.sum((q1 - mu0)**2)/eps**2)
        lognum = ( -0.5* np.sum((q0 - mu1)**2)/eps**2)
        prob = p0* np.exp(lognum - logden)

        if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
            if np.isnan(prob): 
                print("Nan : ", p0, lognum, logden)
            elif np.isinf(prob): 
                print("inf : ", p0, lognum, logden)
            else : 
                print("no movement : ", (q0-q1).sum())
            return q0, 2. , V_q0, V_gq0
        elif np.random.uniform(0., 1., size=1) > min(1., prob):
            return q0, 0., V_q0, V_gq0
        else: return q1, 1., V_q1, V_gq1


    def step(self, q, eps, V_q= None, V_gq=None):
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        z = np.random.normal(size=q.size).reshape(q.shape) 

        if V_gq is None:
            V_gq = self.V_g(q)
        if V_q is None:
            V_q = self.V(q)

        q1 = q - 0.5*eps**2 * V_gq +  eps*z
        V_gq1 = self.V_g(q1)
        V_q1 = self.V(q1)
        
        q2, accepted, V_q2, V_gq2 = self.metropolis(q, q1, V_q, V_q1, V_gq, V_gq1, eps)
        
        return q2, accepted, V_q2, V_gq2

##


