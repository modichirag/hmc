import numpy as np


class PyHMC():
    
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
        p = p - 0.5*step_size * self.V_g(q) 
        for i in range(N-1):
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q) 
        q = q + step_size * self.KE_g(p)
        p = p - 0.5*step_size * self.V_g(q) 
        return q, p


    def metropolis(self, qp0, qp1):
        q0, p0 = qp0
        q1, p1 = qp1
        H0 = self.H(q0, p0)
        H1 = self.H(q1, p1)
        prob = min(1., np.exp(H0 - H1))
        if np.isnan(prob): 
            return q0, p0, 2.
        if np.random.uniform(size=1) > prob:
            return q0, p0, 0.
        else: return q1, p1, 1.


    def hmc_step(self, q, N, step_size):
        p = np.random.normal(size=q.size).reshape(q.shape)
        q1, p1 = self.leapfrog(q, p, N, step_size)
        q, p, accepted = self.metropolis([q, p], [q1, p1])
        return q, p, accepted

