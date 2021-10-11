import tensorflow as tf
import tensorflow_probability as tfp

# tf.enable_eager_execution()

@tf.function
def KE(p):
    return 0.5 * tf.reduce_sum(p**2)



@tf.function
def get_grads(f, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        fv = f(x)
    grad = tape.gradient(fv, x)
    return fv, grad
    

@tf.function
def leapfrog(N, q, p, step_size, V, K):
    p = p - 0.5*step_size * get_grads(V, q)[1]
    for i in range(N-1):
        q = q + step_size * get_grads(K, p)[1]
        p = p - step_size * get_grads(V, q)[1]
    q = q + step_size * get_grads(K, p)[1]
    p = p - 0.5*step_size * get_grads(V, q)[1]
    return q, p

@tf.function
def leapfrog_steps(N, q, p, step_size, V, K):
    toret = []
    for i in range(N):
        g0 = get_grads(V, q)
        phalf = p - 0.5*step_size * g0[1]
        q1 = q + step_size * get_grads(K, phalf)[1]
        g1 = get_grads(V, q1)
        p1 = phalf - 0.5*step_size * g1[1]
        toret.append([[q, q1], [p, phalf,p1], [g0, g1]])
        p = p1
        q = q1
    return q, p, toret


@tf.function
def metropolis(qp0, qp1, H):
    q0, p0 = qp0
    q1, p1 = qp1
    H0 = H(q0, p0)
    H1 = H(q1, p1)
    prob = tf.minimum(1., tf.exp(H0 - H1))
    if tf.math.is_nan(prob): 
        return q0, p0, 2.
    if tf.random.uniform([1], maxval=1.) > prob:
        return q0, p0, 0.
    else: return q1, p1, 1.
        
    
