#bin/python
# Skeleton!

import numpy as np
from copy import deepcopy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import gzip
import time
#import pathos.multiprocessing as mp

# --- CONSTANTS --- #
EXACT=False
PERSISTENT=True
VERBOSE=True
NOISE=False
if NOISE: EXACT=False
if EXACT or NOISE: PERSISTENT=False
if THEANO:
    print 'WARNING: Asking for theano functions, but they are not declared here.'
    from theano import function, shared, scan
    import theano.tensor as tten
    from bf2f_theano_params import *
print 'EXACT:', str(EXACT)
print 'PERSISTENT:', str(PERSISTENT)
print 'NOISE:', str(NOISE)
print 'THEANO:', str(THEANO)
# yolo
#linn = mp.ProcessingPool(5)

class data_stream(object):
    """
    Class for data stream.
    (can use this as a generator)
    """
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        """
        Just spits out lines from the file.
        """
        fi = gzip.open(self.path,'r')
        header = fi.readline()
        while True:
            line = fi.readline()
            if len(line) == 0: # EOF
                break
            else:
                example = map(int, line.split())
                yield example
    def get_vocab_sizes(self):
        """
        The first line of the data file should contain W, R.
        """
        fi = gzip.open(self.path,'r')
        header = fi.readline()
        fi.close()
        values = map(int, header.split())
        if len(values) == 2:
            W, R = values
        else:
            sys.exit('ERROR: data file incorrectly formatted.')
        return W, R
    def acquire_all(self):
        """
        Just suck it all in!
        """
        traindata = [[0, 0, 0]]
        fi = gzip.open(self.path, 'r')
        header = fi.readline()
        for line in fi:
            s, r, t = map(int, line.split())
            traindata.append([s, r, t])
        return np.array(traindata[1:])

def log_likelihood(parameters, data):
    """
    WARNING: Probably don't want to do this most of the time.
    Requires 'data' to be a full list (not just a generator, I think...)
    """
    W = parameters.W
    R = parameters.R
    locations = np.array([[s, r, t] for s in xrange(W) for r in xrange(R) for t in xrange(W) ])
    energy = parameters.E(locations).reshape(W, R, W)
    logZ = np.log(np.sum(np.exp(-energy)))
    ll = np.sum([(-energy[s, r, t] - logZ) for s, r, t in data])
    return ll

def sample_noise(W, R, M):
    """
    Return M totally random samples.
    TODO: allow for other noise distribution.
    """
    noise_samples = np.array(zip(np.random.randint(0, W, M),
                                 np.random.randint(0, R, M),
                                 np.random.randint(0, W, M)))
    return noise_samples

def Z_gradient(parameters):
    """
    Calculates EXACT gradient of the partition function.
    NOTE: intractable most of the time.
    This should possibly belong to the parameters.
    """
    W = parameters.W
    R = parameters.R
    d = parameters.d
    locations = np.array([[s, r, t] for s in xrange(W) for r in xrange(R) for t in xrange(W) ])
    # get exponentiated energy
    energy = parameters.E(locations).reshape(W, R, W)
    expmE = np.exp(-energy)
    Z = np.sum(expmE)
    # get gradients
    dE_C, dE_G, dE_V = parameters.grad_E(locations)
    # empty arrays
    dC_partition = np.zeros(shape=(W, d+1))
    dG_partition = np.zeros(shape=(R, d+1, d+1))
    dV_partition = np.zeros(shape=(W, d+1))
    for (n, (s, r, t)) in enumerate(locations):
        dC_partition[s, :] -= dE_C[n, :]*expmE[s, r, t]
        dG_partition[r, :, :] -= dE_G[n, :, :]*expmE[s, r, t]
        dV_partition[t, :] -= dE_V[n, :]*expmE[s, r, t]
    dC_partition /= Z
    dG_partition /= Z
    dV_partition /= Z
    return dC_partition, dG_partition, dV_partition

def batch_gradient(parameters, batch):
    """
    Gradient is a difference of contributions from:
    1. data distribution (batch of training examples)
    2. model distribution (batch of model samples)
    In both cases, we need to evaluate a gradient over a batch of triples.
    This is a general function for both tasks
    (so we expect to call it twice for each 'true' gradient evaluation.)
    """
    W = parameters.W
    R = parameters.R
    d = parameters.d
    dC_batch = np.zeros(shape=(W, d+1))
    dG_batch = np.zeros(shape=(R, d+1, d+1))
    dV_batch = np.zeros(shape=(W, d+1))
    dE_C_batch, dE_G_batch, dE_V_batch = parameters.grad_E(batch)
    for (i, (s, r, t)) in enumerate(batch):
        dC_batch[s, :] -= dE_C_batch[i]
        dG_batch[r, :, :] -= dE_G_batch[i]
        dV_batch[t, :] -= dE_V_batch[i]
    return (dC_batch, dG_batch, dV_batch)

def combine_gradients(delta_data, delta_model, B, M):
    """
    Just combines two triples...
    """
    # TODO: make this logic more clear/move it elsewhere
    if EXACT:
        prefactor = float(B)
    else:
        prefactor = float(B)/M
    delta_C = delta_data[0] - prefactor*delta_model[0]
    delta_G = delta_data[1] - prefactor*delta_model[1]
    delta_V = delta_data[2] - prefactor*delta_model[2]
    # impose constraints
    delta_C[:, -1] = 0
    delta_V[:, -1] = 0
    #delta_G[:, -1, :] = 0
    # yolo
    delta_G[:, :, :] = 0
    return delta_C, delta_G, delta_V

def train(training_data, start_parameters, options):
    """
    Perform (stochastic) gradient ascent on the parameters.
    INPUTS:
        training_data       iterator of examples.
        start_parameters    triple of (C, G, V)
        options             dictionary
    RETURNS:
        parameters      triple of (C, G, V)
        [[ some measure of convergence ]]
    """
    # unwrap options
    B = options['batch_size']
    S = options['sampling_rate']
    M = options['num_samples']
    D = options['diagnostics_rate']
    K = options['gibbs_iterations']
    calculate_ll = options['calculate_ll']
    alpha, mu = options['alpha'], options['mu']
    logfile = options['logfile']
    # initialise
    vali_set = set()
    batch = np.empty(shape=(B, 3),dtype=np.int)
    # TODO: proper sample initialisation
    samples = np.zeros(shape=(M, 3),dtype=np.int)
    if THEANO:
        parameters = theano_params(start_parameters)
    else:
        parameters = params(start_parameters)
    # diagnostic things
    logf = open(logfile,'w')
    logf.write('n\tt\tll\tde\tme\tve\tre\n')
    W = parameters.W
    R = parameters.R
    n = 0
    t0 = time.time()
    for example in training_data:
        if len(vali_set) < D:
            vali_set.add(tuple(example))
            continue
        # yolo...
        if not W == 5:
            if tuple(example) in vali_set:
                continue
        batch[n%B, :] = example
        n += 1
        if not EXACT and n%S == 0:
            if NOISE:
                samples = sample_noise(W, R, S)
            else:
                if not PERSISTENT: samples[:, :] = batch[np.random.choice(B, M), :]
                for (m, samp) in enumerate(samples):
                    samples[m, :] = parameters.sample(samp, K)
            delta_model = batch_gradient(parameters, samples)
        if n%B == 0 and n > S:
            if EXACT:
                delta_model = Z_gradient(parameters)
            delta_data = batch_gradient(parameters, batch)
            delta_params = combine_gradients(delta_data, delta_model, B, len(samples))
            parameters.update(delta_params, alpha, mu)
        if n%D == 0 and n > B and n > S:
            t = time.time() - t0
            if calculate_ll:
                ll = log_likelihood(parameters, training_data)
            else:
                ll = 'NA'
            data_energy = np.mean(parameters.E(batch))
            vali_energy = np.mean(parameters.E(np.array(list(vali_set))))
            random_lox = np.array(zip(np.random.randint(0, W, 100),
                                      np.random.randint(0, R, 100),
                                      np.random.randint(0, W, 100)))
            rand_energy = np.mean(parameters.E(random_lox))
            if PERSISTENT:
                model_energy = np.mean(parameters.E(samples))
            else:
                model_energy = 'NA'
            logline = [n, t, ll, data_energy, model_energy, vali_energy, rand_energy]
            if VERBOSE:
                for val in logline:
                    if type(val) == float:
                        print '\t','%.3f' % val,
                    else:
                        print '\t', val,
                print ''
            logf.write('\t'.join(map(str, logline))+'\n')
            logf.flush()
    print 'Training done,', n, 'examples seen.'
    return parameters
