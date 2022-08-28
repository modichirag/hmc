/**
 * Example from Inference Gym:
 *
 * https://github.com/tensorflow/probability/blob/master/spinoffs/inference_gym/inference_gym/targets/non_identifiable_quartic.py
 *
 * Au, K. X., Graham, M. M., & Thiery, A. H. (2020). Manifold lifting: scaling
 *  MCMC to the vanishing noise regime. (2), 1-18. Retrieved from
 *  http://arxiv.org/abs/2003.03950
 */
data {
  int<lower = 0> N;
  vector[N] y;
}
transformed data {
  real<lower = 0> sigma = 0.1;
}
parameters {
  // model not identifiable with unconstrained alpha
  real<lower = 0> alpha;
  real<lower = 0> beta;
}
model {
  // Inference Gym priors don't match Au et al.
  alpha ~ std_normal();
  beta ~ std_normal();
  y ~ normal(3 * alpha^2 * (alpha^2 - 1) + beta^2, sigma);
}
generated quantities {
  real alpha_full = bernoulli_rng(0.5) ? alpha : -alpha;
  real beta_full = bernoulli_rng(0.5) ? beta : - beta;
}
