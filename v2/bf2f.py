#!/bin/python
# Skeleton!

import numpy as np
from copy import deepcopy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sys
import gzip
from theano import function, shared
from theano.tensor import fscalar, fmatrix, ftensor3

# --- CONSTANTS --- #
EXACT=True
PERSISTENT=True
THEANO=False
if EXACT: PERSISTENT=False
print 'EXACT:', str(EXACT)
print 'PERSISTENT:', str(PERSISTENT)
print 'THEANO:', str(THEANO)

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

class theano_params(object):
    """
    Parameter object which is... theano-ey.
    """
    def __init__(self, initial_parameters):
        C, G, V = initial_parameters
        if C.shape != V.shape:
            raise ValueError
        if G.shape[1] != C.shape[1]:
            raise ValueError
        if G.shape[2] != C.shape[1]:
            raise ValueError
        self.W = C.shape[0]
        self.R = G.shape[0]
        self.d = C.shape[1] - 1
        # --- initialise shared variables --- #
        # weights
        self.C = shared(np.float32(C), 'C')
        self.G = shared(np.float32(G), 'G')
        self.V = shared(np.float32(V), 'V')
        # velocities
        self.C_vel = shared(np.zeros(shape=C.shape, dtype=np.float32), 'C_vel')
        self.G_vel = shared(np.zeros(shape=G.shape, dtype=np.float32), 'G_vel')
        self.V_vel = shared(np.zeros(shape=V.shape, dtype=np.float32), 'V_vel')
        # --- define theano functions --- #
        # symbolic variables
        muC = tten.fscalar('muC')
        muG = tten.fscalar('muG')
        muV = tten.fscalar('muV')
        alphaC = tten.fscalar('alphaC')
        alphaG = tten.fscalar('alphaG')
        alphaV = tten.fscalar('alphaV')
        deltaC = tten.fmatrix('deltaC')
        deltaG = tten.ftensor3('deltaG')
        deltaV = tten.fmatrix('deltaV')
        # updates
        velocity_inputs = [deltaC, deltaG, deltaV, muC, muG, muV]
        velocity_updates = [(self.C_vel, muC*self.C_vel + (1 - muC)*deltaC),
                            (self.G_vel, muG*self.G_vel + (1 - muG)*deltaG),
                            (self.V_vel, muV*self.V_vel + (1 - muV)*deltaV)]
        parameter_updates = [(self.C, self.C + alphaC*self.C_vel),
                             (self.G, self.G + alphaG*self.G_vel),
                             (self.V, self.V + alphaV*self.V_vel)]
        parameter_inputs = [alphaC, alphaG, alphaV]
        self.update_velocities = function(velocity_inputs, [], updates=velocity_updates, allow_input_downcast=True)
        self.update_parameters = function(parameter_inputs, [], updates=parameter_updates, allow_input_downcast=True)
        
    def update(self, delta_parameters, alpha, mu):
        """
        Update velocities and then parameters.
        """
        # unwrap
        deltaC, deltaG, deltaV = delta_parameters
        alphaC, alphaG, alphaV = alpha
        muC, muG, muV = mu
        # call theano fns
        self.update_velocities(deltaC, deltaG, deltaV, muC, muG, muV)
        self.update_parameters(alphaC, alphaG, alphaV)

    def grad_E(self, locations):
        """
        Gradients of the energy, evaluated at a list of triples.
        NOTE: this clearly depends on the choice of energy.
        Returns tensors whose first index corresponds to the input triple list.
        """
        C_sub = self.C.get_value()[locations[:, 0]]
        G_sub = self.G.get_value()[locations[:, 1]]
        V_sub = self.V.get_value()[locations[:, 2]]
        # this is for Etype == 'dot'
        # TODO: theanofy
        dE_C = -np.einsum('...i,...ij', V_sub, G_sub)
        dE_G = -np.einsum('...i,...j', V_sub, C_sub)
        dE_V = -np.einsum('...ij,...j', G_sub, C_sub)
        return dE_C, dE_G, dE_V

    def E(self, locations):
        """
        Just plain old energy between triples.
        locations is an array of triples.
        Outputs a list (same length as 'locations') of energy of each triple.
        """
        C_sub = self.C.get_value()[locations[:, 0]]
        G_sub = self.G.get_value()[locations[:, 1]]
        V_sub = self.V.get_value()[locations[:, 2]]
        # this is for Etype == 'dot'
        # TODO: theanofy
        GC_sub = np.einsum('...ij,...j', G_sub, C_sub)
        energy = -np.einsum('...i,...i', V_sub, GC_sub)
        return energy

    def sample(self, seed, K):
        """
        Draws samples from the model, given a (single!) seed.
        (iterates through Gibbs sampling K times)
        """
        W = self.C.get_value().shape[0]
        R = self.G.get_value().shape[0]
        ss = deepcopy(seed)
        for iteration in xrange(K):
            order = np.random.permutation(3)
            for triple_drop in order:
                if triple_drop == 0:
                    locs = np.array([ [i, ss[1], ss[2]] for i in xrange(W) ])
                if triple_drop == 1:
                    locs = np.array([ [ss[0], i, ss[2]] for i in xrange(R) ])
                if triple_drop == 2:
                    locs = np.array([ [ss[0], ss[1], i] for i in xrange(W) ])
                expmE = np.exp(-self.E(locs))
                probs = expmE/np.sum(expmE)
                samp = np.random.choice(len(probs), p=probs, size=1)[0]
                ss[triple_drop] = samp
        return ss

class params(object):
    """
    Parameter object.
    Contains C, G, V and velocities for all.
    """
    def __init__(self, initial_parameters):
        C, G, V = initial_parameters
        if C.shape != V.shape:
            raise ValueError
        if G.shape[1] != C.shape[1]:
            raise ValueError
        if G.shape[2] != C.shape[1]:
            raise ValueError
        self.W = C.shape[0]
        self.R = G.shape[0]
        self.d = C.shape[1] - 1
        # weights
        self.C = C
        self.G = G
        self.V = V
        # velocities
        self.C_vel = np.zeros(shape=self.C.shape)
        self.G_vel = np.zeros(shape=self.G.shape)
        self.V_vel = np.zeros(shape=self.V.shape)

    def update(self, delta_parameters, alpha, mu):
        """
        Updates velocities and then parameters.
        """
        # unwrap
        deltaC, deltaG, deltaV = delta_parameters
        alphaC, alphaG, alphaV = alpha
        muC, muG, muV = mu
        # update velocities
        self.C_vel = muC*self.C_vel + (1-muC)*deltaC
        self.G_vel = muG*self.G_vel + (1-muG)*deltaG
        self.V_vel = muV*self.V_vel + (1-muV)*deltaV
        # update parameters
        self.C = self.C + alphaC*self.C_vel
        self.G = self.G + alphaG*self.G_vel
        self.V = self.V + alphaV*self.V_vel

    def grad_E(self, locations):
        """
        Gradients of the energy, evaluated at a list of triples.
        NOTE: this clearly depends on the choice of energy.
        Returns tensors whose first index corresponds to the input triple list.
        """
        C_sub = self.C[locations[:, 0]]
        G_sub = self.G[locations[:, 1]]
        V_sub = self.V[locations[:, 2]]
        # this is for Etype == 'dot'
        # TODO: make this efficient
        dE_C = -np.einsum('...i,...ij', V_sub, G_sub)
        dE_G = -np.einsum('...i,...j', V_sub, C_sub)
        dE_V = -np.einsum('...ij,...j', G_sub, C_sub)
        return dE_C, dE_G, dE_V
 
    def E(self, locations):
        """
        Just plain old energy between triples.
        locations is an array of triples.
        Outputs a list (same length as 'locations') of energy of each triple.
        """
        C_sub = self.C[locations[:, 0]]
        G_sub = self.G[locations[:, 1]]
        V_sub = self.V[locations[:, 2]]
        # this is for Etype == 'dot'
        # TODO: 
        #   profile speed wrt order
        #   wrt just looping through locations
        GC_sub = np.einsum('...ij,...j', G_sub, C_sub)
        energy = -np.einsum('...i,...i', V_sub, GC_sub)
        return energy
  
    def sample(self, seed, K):
        """
        Draws samples from the model, given a (single!) seed.
        (iterates through Gibbs sampling K times)
        """
        W = self.C.shape[0]
        R = self.G.shape[0]
        ss = deepcopy(seed)
        for iteration in xrange(K):
            order = np.random.permutation(3)
            for triple_drop in order:
                if triple_drop == 0:
                    locs = np.array([ [i, ss[1], ss[2]] for i in xrange(W) ])
                if triple_drop == 1:
                    locs = np.array([ [ss[0], i, ss[2]] for i in xrange(R) ])
                if triple_drop == 2:
                    locs = np.array([ [ss[0], ss[1], i] for i in xrange(W) ])
                expmE = np.exp(-self.E(locs))
                probs = expmE/np.sum(expmE)
                samp = np.random.choice(len(probs), p=probs, size=1)[0]
                ss[triple_drop] = samp
        return ss

def plot_trace(trace):
    """
    Simple scatter-plot of the trace of log-likelihood.
    """
    # TODO: make beautiful
    # todo: fix
    fig = plt.figure()
    n = trace[:, 0]
    ll = trace[:, 1]
    plt.scatter(n, ll, linewidth=0, color='red')
    plt.ylabel('log-likelihood of training data')
    plt.xlabel('number of training examples seen')
    plt.savefig('trace.png')
    plt.close()
    return True

def log_likelihood(parameters, data):
    """
    WARNING: Probably don't want to do this most of the time.
    """
    W = parameters.W
    R = parameters.R
    locations = np.array([[s, r, t] for s in xrange(W) for r in xrange(R) for t in xrange(W) ])
    energy = parameters.E(locations).reshape(W, R, W)
    logZ = np.log(np.sum(np.exp(-energy)))
    ll = np.sum([(-energy[s, r, t] - logZ) for s, r, t in data])
    return ll

def Z_gradient(parameters):
    """
    Calculates EXACT gradient of the partition function.
    NOTE: intractable most of the time.
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
    D = options['diagnostics_rate']
    K = options['gibbs_iterations']
    alpha, mu = options['alpha'], options['mu']
    # initialise
    batch = np.empty(shape=(B, 3),dtype=np.int)
    # TODO: proper sample initialisation
    samples = np.zeros(shape=(S, 3),dtype=np.int)
    if THEANO:
        parameters = theano_params(start_parameters)
    else:
        parameters = params(start_parameters)
    # diagnostic things
    ll_trace = [[0, log_likelihood(parameters, training_data)]]
    # energy traces
    de_trace, me_trace, ve_trace, re_trace = [[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]
    W = parameters.W
    R = parameters.R
    vali_set = set()
    n = 0
    for example in training_data:
        if len(vali_set) < D:
            vali_set.add(tuple(example))
            continue
        #if tuple(example) in vali_set:
        #    continue
        batch[n%B, :] = example
        n += 1
        if not EXACT and n%S == 0:
            if not PERSISTENT: samples[:, :] = batch[:, :]
            for (m, samp) in enumerate(samples):
                samples[m, :] = parameters.sample(samp, K)
            delta_model = batch_gradient(parameters, samples)
        if n%B == 0:
            if EXACT:
                delta_model = Z_gradient(parameters)
            delta_data = batch_gradient(parameters, batch)
            delta_params = combine_gradients(delta_data, delta_model, B, len(samples))
            parameters.update(delta_params, alpha, mu)
        if n%D == 0 and n > B:
            ll_trace.append([n, log_likelihood(parameters, training_data)])
            de_trace.append([n, np.mean(parameters.E(batch))])
            ve_trace.append([n, np.mean(parameters.E(np.array(list(vali_set))))])
            random_lox = np.array(zip(np.random.randint(0, W, D*10), np.random.randint(0, R, D*10), np.random.randint(0, W, D*10)))
            re_trace.append([n, np.mean(parameters.E(random_lox))])
            if PERSISTENT:
                me_trace.append([n, np.mean(parameters.E(samples))])
            else:
                me_trace.append([n, 0])
            convergence = 10
    print 'Training done,', n, 'examples seen.'
    return parameters, (ll_trace, de_trace, me_trace, ve_trace, re_trace), convergence
