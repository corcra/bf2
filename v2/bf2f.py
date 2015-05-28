#!/bin/python
# Main functions/classes for bri-focal v2.

import numpy as np
# DO NOT PLAY NICE WITH NANS
np.seterr(all='raise')
np.seterr(under='warn')
from copy import deepcopy
import sys
import gzip
import time
import re
from copy import deepcopy
#import pathos.multiprocessing as mp

# --- CONSTANTS --- #
THEANO=False
if THEANO:
    print 'WARNING: Asking for theano functions, but they are not declared here.'
    from theano import function, shared, scan
    import theano.tensor as tten
    from bf2f_theano_params import *
# yolo
#linn = mp.ProcessingPool(5)

# --- functions ! --- #
def clean_word(word):
    # lowercase
    word1 = word.lower()
    # remove: . , ( )
    word2 = re.sub('[,\.()""]', '', word1)
    # replace digits with NUM
    word3 = re.sub('[\d#]+', '#', word2)
    # strip trailing space
    word4 = word3.rstrip(' ')
    return word4

# --- data stream --- #
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
        if '.gz' in self.path:
            fi = gzip.open(self.path,'r')
        else:
            fi = open(self.path,'r')
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
        if '.gz' in self.path:
            fi = gzip.open(self.path,'r')
        else:
            fi = open(self.path,'r')
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
        if '.gz' in self.path:
            fi = gzip.open(self.path, 'r')
        else:
            fi = open(self.path, 'r')
        header = fi.readline()
        for line in fi:
            s, r, t = map(int, line.split())
            traindata.append([s, r, t])
        return np.array(traindata[1:])

# --- parameters object --- #
class params(object):
    """
    Parameter object.
    Contains C, G, V and velocities for all.
    """
    def __init__(self, initial_parameters, vocab=None,
                 fix_words=False, fix_relas=False, trans_rela=False):
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
        # vocab
        try:
            self.words = vocab['words']
            self.relas = vocab['relas']
        except TypeError:
            # no vocab
            self.words = map(str, range(self.W))
            self.relas = map(str, range(self.R))
        # weights
        self.C = deepcopy(C)
        self.G = deepcopy(G)
        self.V = deepcopy(V)
        # velocities
        self.C_vel = np.zeros(shape=self.C.shape)
        self.G_vel = np.zeros(shape=self.G.shape)
        self.V_vel = np.zeros(shape=self.V.shape)
        # fix some parameters?
        # (never update these)
        self.fix_words = fix_words
        self.fix_relas = fix_relas
        # special type of relationship (translations only)
        self.trans_rela = trans_rela

    def update(self, delta_parameters, alpha, mu):
        """
        Updates velocities and then parameters.
        """
        # unwrap
        deltaC, deltaG, deltaV = delta_parameters
        alphaC, alphaG, alphaV = alpha
        muC, muG, muV = mu
        # update velocities
        if not self.fix_words:
            self.C_vel = muC*self.C_vel + (1-muC)*deltaC
            self.V_vel = muV*self.V_vel + (1-muV)*deltaV
        if not self.fix_relas:
            self.G_vel = muG*self.G_vel + (1-muG)*deltaG
        # update parameters
        if not self.fix_words:
            self.C += alphaC*self.C_vel
            self.V += alphaV*self.V_vel
        if not self.fix_relas:
            if self.trans_rela:
                # only update the final column of G
                self.G[:, :, -1] = alphaG*self.G_vel[:, :, -1]
            else:
                self.G += alphaG*self.G_vel

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

    def E_axis(self, triple, switch):
        """
        Returns energies over an axis (S, R, T) given two of the triple.
        """
        s, r, t = triple
        if switch == 'C':
            # return over all S
            #GC = np.dot(self.C, self.G[r].T)
            #energy = -np.dot(GC, self.V[t])
            # note: above version is significantly slower than the below
            VG = np.dot(self.V[t], self.G[r])
            energy = -np.dot(self.C, VG)
        elif switch == 'G':
            # return over all R
            VG = np.dot(self.V[t], self.G)
            energy = -np.dot(VG, self.C[s])
        elif switch == 'V':
            #return over all T
            GC = np.dot(self.G[r], self.C[s])
            energy = -np.dot(self.V, GC)
        else:
            print 'ERROR: Cannot parse switch.'
            sys.exit()
        return energy

    def E_triple(self, triple):
        """
        The energy of a SINGLE triple.
        """
        return -np.dot(self.V[triple[2]], np.dot(self.G[triple[1]], self.C[triple[0]]))

    def E(self, locations=None):
        """
        Just plain old energy between triples.
        locations is an array of triples.
        Outputs a list (same length as 'locations') of energy of each triple.
        """
        #C_sub = self.C[locations[:, 0]]
        #G_sub = self.G[locations[:, 1]]
        #V_sub = self.V[locations[:, 2]]
        # this is for Etype == 'dot'
        # TODO:
        #   profile speed wrt order
        #   wrt just looping through locations
        #   # yolo
        # profiling...
        # V1
        #GC_sub = np.einsum('...ij,...j', G_sub, C_sub)
        #energy = -np.einsum('...i,...i', V_sub, GC_sub)
        # V2
        #energy = np.empty(shape=(len(locations)))
        #for i in xrange(len(locations)):
        #    energy[i] = -np.dot(C_sub[i],np.dot(V_sub[i], G_sub[i]))
        # V3
        #VG_sub = np.einsum('...i,...ij', V_sub, G_sub)
        #energy = -np.einsum('...i,...i', VG_sub, C_sub)
        # V4
        #energy = map(lambda triple: -np.dot(self.C[triple[0]], np.dot(self.V[triple[2]], self.G[triple[1]])), locations)
        #energy = np.array(map(lambda (s, r, t): -np.dot(self.C[s], np.dot(self.V[t], self.G[r])), locations))
        # V5
        #energy = map(lambda i: -np.dot(V_sub[i], np.dot(G_sub[i], C_sub[i])), xrange(len(locations)))
        # V6
        #energy = linn.amap(self.E_triple, locations)
        # V7
        #parmz = []
        #for triple in locations:
        #    parmz.append(((self.C, self.G, self.V), triple))
        #energy = map(silly_energy, parmz)
        # V8
        #energy = np.empty(shape=len(locations))
        #for (i, triple) in enumerate(locations):
        #    energy[i] = self.E_triple(triple)
        # V9
        if locations == None:
            W = self.W
            R = self.R
            locations = np.array([[s, r, t] for s in xrange(W) for r in xrange(R) for t in xrange(W) ])
        energy = np.empty(shape=len(locations), dtype=np.float)
        for (i, triple) in enumerate(locations):
            energy[i] = -np.dot(self.C[triple[0]], np.dot(self.V[triple[2]], self.G[triple[1]]))
        # V10
        #energy = []
        #for triple in locations:
        #    energy.append(-np.dot(self.C[triple[0]], np.dot(self.V[triple[2]], self.G[triple[1]])))
        return energy

    def sample(self, seed, K):
        """
        Draws samples from the model, given a (single!) seed.
        (iterates through Gibbs sampling K times)
        """
        W = self.W
        R = self.R
        ss = deepcopy(seed)
        for iteration in xrange(K):
            order = np.random.permutation(3)
            for triple_drop in order:
                if triple_drop == 0:
                    energy = self.E_axis(ss, 'C')
                    #locs = np.array([ [i, ss[1], ss[2]] for i in xrange(W) ])
                if triple_drop == 1:
                    energy = self.E_axis(ss, 'G')
                    #locs = np.array([ [ss[0], i, ss[2]] for i in xrange(R) ])
                if triple_drop == 2:
                    energy = self.E_axis(ss, 'V')
                    #locs = np.array([ [ss[0], ss[1], i] for i in xrange(W) ])
                #expmE = np.exp(-self.E(locs))
                expmE = np.exp(-energy)
                probs = expmE/np.sum(expmE)
                samp = np.random.choice(len(probs), p=probs, size=1)[0]
                ss[triple_drop] = samp
        return ss

    def get(self):
        """
        Method to return the (C, G, V) triple.
        """
        return (self.C, self.G, self.V)

    def save(self, filename):
        """
        Method to save the parameters to file.
        """
        if not 'XXX' in filename:
            print 'WARNING: Save expects an XXX in the filename. Fixed that for you.'
            filename = filename+'_XXX'
        if '.npy' in filename:
            C_dict = dict(zip(self.words, self.C[:,:-1]))
            G_dict = dict(zip(self.relas, self.G))
            V_dict = dict(zip(self.words, self.V[:, :-1]))
            np.save(re.sub('XXX','C',filename),C_dict)
            np.save(re.sub('XXX','G',filename),G_dict)
            np.save(re.sub('XXX','V',filename),V_dict)
        elif '.txt' in filename:
            fC = open(re.sub('XXX','C',filename), 'w')
            fG = open(re.sub('XXX','G',filename), 'w')
            fV = open(re.sub('XXX','V',filename), 'w')
            fC.write(str(self.W)+' '+str(self.C.shape[1]-1)+'\n')
            fG.write(str(self.R)+' '+str(self.C.shape[1]-1)+'\n')
            fV.write(str(self.W)+' '+str(self.C.shape[1]-1)+'\n')
            for i in xrange(self.W):
                try:
                    word = self.words[i]
                except TypeError:
                    word = 'word_'+str(i)
                fC.write(word+' '+' '.join(map(str, self.C[i,:-1]))+'\n')
                fV.write(word+' '+' '.join(map(str, self.V[i,:-1]))+'\n')
            for i in xrange(self.R):
                try:
                    rela = self.relas[i]
                except TypeError:
                    rela = 'rela_'+str(i)
                fG.write('rela_'+str(i)+' '+' '.join(map(str, self.G[i,:-1,:].reshape((self.C.shape[1])*(self.C.shape[1]-1),)))+'\n')
            fC.close()
            fV.close()
            fG.close()
        return True

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

def combine_gradients(delta_data, delta_model, prefactor):
    """
    Just combines two triples...
    """
    delta_C = delta_data[0] - prefactor*delta_model[0]
    delta_G = delta_data[1] - prefactor*delta_model[1]
    delta_V = delta_data[2] - prefactor*delta_model[2]
    # impose constraints
    delta_C[:, -1] = 0
    delta_V[:, -1] = 0
    delta_G[:, -1, :] = 0
    delta_G[0, :, :] = 0
    return delta_C, delta_G, delta_V

def permute_batch(word_perm, rela_perm, batch):
    """
    This function will take a list of triples and a pair of KNOWN permutations
    (W -> W', R -> R')
    and output triples after applying the transformation.
    """
    mapped_batch = np.empty(shape=batch.shape)
    for (i, (s, r, t)) in enumerate(batch):
        mapped_batch[i] = (word_perm[s], rela_perm[r], word_perm[t])
    return mapped_batch

def update_learning_rate(alpha0, tau, t):
    """
    Update learning rate according to some schedule.
    """
    alpha_new = [alpha0[0], alpha0[1], alpha0[2]]
    for z in xrange(3):
        if not tau[z] == 0:
            alpha_new[z] = alpha0[z]/(1+float(t)/tau[z])
    return alpha_new
        
def train(training_data, start_parameters, options,
          EXACT=False, PERSISTENT=True, NOISE=False, VERBOSE=True):
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
    if NOISE: EXACT=False
    if EXACT or NOISE: PERSISTENT=False
    # unwrap options
    B = options['batch_size']
    S = options['sampling_rate']
    M = options['num_samples']
    D = options['diagnostics_rate']
    K = options['gibbs_iterations']
    calculate_ll = options['calculate_ll']
    alpha0, mu, tau = options['alpha'], options['mu'], options['tau']
    name = options['name']
    try:
        vali_set_size = options['vali_set_size']
    except KeyError:
        vali_set_size = D
    # initialise
    vali_set = set()
    batch = np.empty(shape=(B, 3),dtype=np.int)
    # TODO: proper sample initialisation
    samples = np.zeros(shape=(M, 3),dtype=np.int)
    if not type(start_parameters) == params:
        parameters = params(start_parameters)
    else:
        parameters = start_parameters
    # diagnostic things
    logf = open(name+'_logfile.txt','a')
    logf.write('n\ttime\tll\tdata_energy\tmodel_energy\tvaliset_energy\trandom_energy\tperm_energy\tC_lens\tG_lens\tV_lens\n')
    W = parameters.W
    R = parameters.R
    # a fixed permutation, for testing my strange likelihood ratio thing
    W_perm = dict(enumerate(np.random.permutation(W)))
    R_perm = dict(enumerate(np.random.permutation(R)))
    # record sampling frequencies
    #sampled_counts = dict((i, 0) for i in xrange(W))
    n = 0
    t0 = time.time()
    for example in training_data:
        if len(vali_set) < vali_set_size:
            vali_set.add(tuple(example))
            continue
        if len(vali_set) == vali_set_size:
            perm_vali_batch = permute_batch(W_perm, R_perm, np.array(list(vali_set)))
        # explanation for this:
        # in W=5 dataset, if you exclude vali_set, you lose a significant %
        # of the training data...
        if not W == 5:
            if tuple(example) in vali_set:
                continue
        batch[n%B, :] = example
        #yolo
        #sampled_counts[example[0]] +=1
        #sampled_counts[example[2]] +=1
        n += 1
        if not EXACT and n%S == 0:
            if NOISE:
                samples = sample_noise(W, R, S)
            else:
                if not PERSISTENT: samples[:, :] = batch[np.random.choice(B, M), :]
                for (m, samp) in enumerate(samples):
                    sampled_triple = parameters.sample(samp, K)
                    samples[m, :] = sampled_triple
                    # yolo
                    #sampled_counts[sampled_triple[0]] += 1
                    #sampled_counts[sampled_triple[2]] += 1
            # yolo
            #print sampled_counts.values()
            delta_model = batch_gradient(parameters, samples)
            prefactor = float(B)/len(samples)
        if n%B == 0 and n > S:
            alpha = update_learning_rate(alpha0, tau, n/B)
            if EXACT:
                delta_model = Z_gradient(parameters)
                prefactor = float(B)
            delta_data = batch_gradient(parameters, batch)
            delta_params = combine_gradients(delta_data, delta_model, prefactor)
            parameters.update(delta_params, alpha, mu)
        if n%D == 0 and n > B and n > S:
            t = time.time() - t0
            if calculate_ll:
                ll = log_likelihood(parameters, training_data)
                #ll = log_likelihood(parameters, vali_set)
            else:
                ll = 'NA'
            data_energy = np.mean(parameters.E(batch))
            vali_energy = np.mean(parameters.E(np.array(list(vali_set))))
            random_lox = np.array(zip(np.random.randint(0, W, 100),
                                      np.random.randint(0, R, 100),
                                      np.random.randint(0, W, 100)))
            rand_energy = np.mean(parameters.E(random_lox))
            # so this is different to the rand, cause it's a permuted version of the validation set
            perm_energy = np.mean(parameters.E(perm_vali_batch))
            if PERSISTENT:
                model_energy = np.mean(parameters.E(samples))
            else:
                model_energy = 'NA'
            # get some vector lengths
            # TODO: make this more elegant
            see, gee, vee = parameters.get()
            C_lens = np.mean(np.linalg.norm(see[random_lox[:, 0], :-1], axis=1))
            G_lens = np.mean(np.linalg.norm(gee[random_lox[:, 1], :-1], axis=(1,2)))
            V_lens = np.mean(np.linalg.norm(vee[random_lox[:, 2], :-1], axis=1))
            # record to logfile
            logline = [n, t, ll, data_energy, model_energy, vali_energy, rand_energy, perm_energy, C_lens, G_lens, V_lens]
            if VERBOSE:
                for val in logline:
                    if type(val) == str: 
                        print '\t', val,
                    else:
                        print '\t','%.3f' % val,
                print '\nC_lenz:', "%.5f" % C_lens, 'G_lenz:', "%.5f" % G_lens, 'V_lenz:', "%.5f" % V_lens
            logf.write('\t'.join(map(str, logline))+'\n')
            logf.flush()
            # yolo
            #if np.random.random() < 0.2:
            #    for r in xrange(R):
            #        anim_fo = open('animations/anim_R'+str(r)+'_'+str(n).zfill(5)+'.txt','w')
            #        for w in xrange(W):
            #            #anim_fo.write('C'+str(w)+' '+' '.join(map(str, parameters.C[w, :-1]))+'\n')
            #            anim_fo.write('V'+str(w)+' '+' '.join(map(str, np.dot(parameters.G[r, :, :],parameters.V[w, :])[:-1]))+'\n')
            #        anim_fo.close()
            # endyolo
        if n%(D*10) == 0:
            parameters.save(name+'_XXX.npy')
    if VERBOSE: print 'Training done,', n, 'examples seen.'
    parameters.save(name+'_XXX.npy')
    return vali_set
