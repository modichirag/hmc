import numpy 
import sys, os, json
import stan_models as sm

MODELDIR = '../compiled_models/'


#####
model_code_funnel = """
parameters {
  real v; 
  vector[%s] theta;
}

model {
v ~ normal(0, 3);
theta ~ normal(0, exp(v/2));
}
"""

###
def create_funnel_model(ndim):
    print("Create funnel model for n=%d"%ndim)
    stanmodel = model_code_funnel%ndim
    modelname = "funnel_%02d"%ndim
    data = None
    model = sm.setup_stan_models(stanmodel, data, modelname)
    print(model)
    print(model.dims())


#####
model_code_normal = """
parameters {
  vector[%s] alpha;
}

model {
  alpha ~ normal(0, 1);
}
"""

###
def create_normal_model(ndim):
    print("Create normal model for n=%d"%ndim)
    stanmodel = model_code_normal%ndim
    modelname = "normal_%02d"%ndim
    data = None
    model = sm.setup_stan_models(stanmodel, data, modelname)
    print(model)
    print(model.dims())



if __name__=="__main__":

    ndim = 2
    create_funnel_model(ndim)
    create_normal_model(ndim)
    
