import numpy as np
import sys, os
import json

#
from posteriordb import PosteriorDatabase

##Setup your posterior DB environment
PDBPATH = os.path.join('/mnt/home/cmodi/Research/Projects/posteriordb/posterior_database/')
CMDSTAN = '/mnt/home/cmodi/Research/Projects/cmdstan/'
BRIDGESTAN = '/mnt/home/cmodi/Research/Projects/bridgestan/'
MODELDIR = '../compiled_models/'

sys.path.append(BRIDGESTAN)
import PythonClient as pbs

###
def get_bridgestan_model(sopath, datapath):

    try:
        if os.path.isfile(sopath):
            print("%s file exists"%sopath)
        model = pbs.PyBridgeStan(sopath, datapath)
    except Exception as e:
        print(e)
        cwd = os.getcwd()
        print(cwd)
        os.chdir(BRIDGESTAN)
        os.system('make  %s'%sopath) 
        os.chdir(cwd)
        model = pbs.PyBridgeStan(sopath, datapath)
    return model


###
def get_pdb_model(model_n):

    pdb = PosteriorDatabase(PDBPATH)
    modelnames = pdb.posterior_names()
    posterior = pdb.posterior(modelnames[model_n])
    posname = posterior.name
    print("Model name :", posname)

    stanmodel, data = posterior.model, posterior.data.values()

    refdrawsdict = posterior.reference_draws()
    keys = refdrawsdict[0].keys()
    stansamples = []
    for key in keys:
        stansamples.append(np.array([refdrawsdict[i][key] for i in range(len(refdrawsdict))]).flatten())
    samples = np.array(stansamples).copy().astype(np.float32).T

    return stanmodel, data, samples


###
def setup_pdb_model(model_n):

    stanmodel, data, samples = get_pdb_model(model_n)

    #Save stan model code
    modeldir = MODELDIR + '/PDB_%02d/'%model_n
    os.makedirs(modeldir, exist_ok=True)
    modpath = modeldir + 'PDB_%02d'%model_n
    modpath = os.path.abspath(modpath)
    print(modpath)
    with open(modpath + '.stan', 'w') as f:
        f.write(stanmodel.code('stan'))

    #Save corresponding data
    datapath = modpath + '.data.json'
    with open(datapath, 'w') as f:
        json.dump(data, f,)

    #Save compiled shared object
    sopath = modpath + '_model.so'
    model = get_bridgestan_model(sopath, datapath)

    return model, samples


###
def setup_stan_models(stanmodel, data, modelname):

    #Save stan model code
    modeldir = MODELDIR + '/%s/'%modelname
    os.makedirs(modeldir, exist_ok=True)
    modpath = modeldir + '%s'%modelname
    modpath = os.path.abspath(modpath)
    print(modpath)
    with open(modpath + '.stan', 'w') as f:
        f.write(stanmodel)

    #Save corresponding data
    datapath = modpath + '.data.json'
    with open(datapath, 'w') as f:
        if data is not None:
            json.dump(data, f,)
        else:
            json.dump({}, f,)
        
    #Save compiled shared object
    sopath = modpath + '_model.so'
    model = get_bridgestan_model(sopath, datapath)

    return model


###
def constrain_samples(model, samples):
    
    csamples = []
    if len(samples.shape) > 1:
        for i in range(samples.shape[0]):
            csamples.append(model.param_constrain(samples[i])*1.)
        csamples = np.array(csamples)
    elif len(samples.shape) == 1:
        csamples = model.param_constrain(samples)
    
    return csamples

