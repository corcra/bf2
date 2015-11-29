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
# yolo
#from thresholds_fn import devset_accuracy
from math import pi

# --- CONSTANTS --- #
# ADAM settings... (http://arxiv.org/pdf/1412.6980.pdf)
EPSILON=1e-8
LAMBDA=(1-1e-8)
# code for unobserved relationship
MISS_R=9999

# --- helper fns --- #
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

def generate_traindata(droot, W, R, N=None):
    # define joint probabilities of all triples
    # not sure why i'm using a beta distribution but w/e
    probs = np.random.beta(a=0.5, b=0.5, size=(W, R, W))
    Z = np.sum(probs)
    probs /= Z
    if N is None:
        # how many training examples?
        N = 50*W*R
    np.save(droot+'/W'+str(W)+'_R'+str(R)+'_N'+str(N)+'_probs.npy', probs)
    print 'Generating', N, 'training examples with', W, 'words and', R, 'relas.'
    locs = [(0,0,0)]*(W*R*W)
    loc_probs = [0]*(W*R*W)
    i = 0
    for s in xrange(W):
        for r in xrange(R):
            for t in xrange(W):
                locs[i] = (s, r, t)
                loc_probs[i] = probs[s, r, t]
                i += 1
    fo = gzip.open(droot+'/W'+str(W)+'_R'+str(R)+'_N'+str(N)+'_train.txt.gz','wb')
    fo.write(str(W)+' '+str(R)+'\n')
    triples = np.random.choice(len(locs), p=loc_probs, size=N, replace=True)
    for n in xrange(N):
        triple = locs[triples[n]]
        fo.write(' '.join(map(str, triple))+'\n')
    fo.close()
    return True

# --- options object --- #
class options_dict(dict):
    """
    Class for object.
    """
    def __init__(self):
        """ initialise all settings to defaults """
        # default values
        self['online'] = True
        self['exact'] = False
        self['persistent'] = True
        self['noise'] = False
        self['adam'] = True
        self['normalise'] = False
        self['etype'] = 'dot'
        self['calc_ll'] = False
        self['wordlist'] = None
        self['relalist'] = None
        self['fix_words'] = False
        self['fix_relas'] = False
        self['trans_rela'] = False
        self['kappa'] = [0, 0, 0]
        self['seed'] = 1337
        # need to input the rest of the defaults
        # (every possible option should be initialised here somehow)
    def pretty_print(self):
        """ print out """
        for (name, value) in self.iteritems():
            print value, '\t:', name

    def load(self, path, verbose=False):
        # BRITTLE
        print 'Reading options from',  path
        options_raw = open(path, 'r').readlines()
        options = dict()
        for line in options_raw:
            if '#' in line:
                # skip 'comments'
                continue
            option_name = line.split(' ')[0]
            option_value = ' '.join(line.split(' ')[1:])
            # this is gross
            if '(' in option_value:
                value = tuple(map(float, re.sub('[\(\)]', '', option_value).split(',')))
            elif '[' in option_value:
                value = np.array(map(float, re.sub('[\[\]]', '', option_value).split(',')))
            elif option_value == 'False\n':
                value = False
            elif option_value == 'True\n':
                value = True
            else:
                try:
                    value = int(option_value)
                except ValueError:
                    # not an int
                    value = option_value.strip()
            self[option_name] = value
        # make np arrays
        self['mu'] = np.array(self['mu'])
        self['nu'] = np.array(self['nu'])
        self['alpha'] = np.array(self['alpha'])
        # optional
        if 'omega' in self:
            self['omega'] = np.array(self['omega'])
        self.check(verbose)
        if verbose:
            self.pretty_print()

    def save(self, path, verbose=False):
        """ save to a file """
        fo = open(path, 'w')
        for (option_name, value) in self.iteritems():
            if type(value) == np.ndarray:
                value = tuple(value)
            fo.write(option_name+' '+str(value)+'\n')
        fo.close()
        if verbose:
            print 'Options saved to', path

    def check(self, verbose=False):
        """ sanity check """
        # hard constraints
        if not 'training_data_path' in self:
            sys.exit('ERROR: missing training_data_path')
        if 'batch' in self['output_root']:
            assert not self['online']
        if 'inexact' in self['output_root']:
            assert not self['exact']
        if 'nonpersistent' in self['output_root']:
            assert not self['persistent']
        if 'noise' in self['output_root']:
            assert self['noise']
        if 'ADAM' in self['output_root']:
            assert self['adam']
        if 'SGD' in self['output_root']:
            assert not self['adam']
        # soft warnings
        if self['wordlist'] is None:
            print 'WARNING: wordlist doesn\'t exist.'
        if self['relalist'] is None:
            print 'WARNING: relalist doesn\'t exist.'
        if self['diagnostics_rate'] <= 0:
            print 'WARNING: no diagnostics.'
        if verbose:
            print 'Options passed all checks.'
    
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
    def acquire_all(self, SHUFFLE=True):
        """
        Just suck it all in!
        By default, we shuffle the training data.
        (this may help with multiple epochs)
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
        traindata = np.array(traindata[1:])
        if SHUFFLE:
            np.random.shuffle(traindata)
        return np.array(traindata[1:])

# --- parameters object --- #
class params(object):
    """
    Parameter object.
    Contains C, G, V and velocities for all.
    """
    def __init__(self, initial_parameters, options=None, vocab=None):
        if options is None:
            options = options_dict()
        self.etype = options['etype']
        if type(initial_parameters) == str:
            # assume a PATH has been given
            params_path = initial_parameters
            self.load(params_path)
        elif type(initial_parameters) == tuple:
            # assume a triple of parameters has been given
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
            if type(vocab) == tuple:
                # assume a tuple of PATHs has been given
                if not vocab[0] is None:
                    # words first
                    wordlist_path = vocab[0]
                    words_raw = open(wordlist_path, 'r').readlines()
                    words = ['']*len(words_raw)
                    if '.tsv' in wordlist_path:
                        split_on = '\t'
                    else:
                        split_on = ' '
                    for line in words_raw:
                        sl = line.strip('\n').split(split_on)
                        index = int(sl[0])
                        word = sl[1]
                        words[index] = word
                else:
                    words = map(str, range(self.W))
                if not vocab[1] is None:
                    # relas first
                    relalist_path = vocab[1]
                    relas_raw = open(relalist_path, 'r').readlines()
                    relas = ['']*len(relas_raw)
                    if '.tsv' in relalist_path:
                        split_on = '\t'
                    else:
                        split_on = ' '
                    for line in relas_raw:
                        sl = line.strip('\n').split(split_on)
                        index = int(sl[0])
                        rela = sl[1]
                        relas[index] = rela
                else:
                    relas = map(str, range(self.R))
                # assign
                self.words = words
                self.relas = relas
            elif type(vocab) == dict:
                self.words = vocab['words']
                self.relas = vocab['relas']
            else:
                sys.exit('ERROR: vocab is of unexpected type. Aborting.')
            # weights
            self.C = deepcopy(C)
            self.G = deepcopy(G)
            self.V = deepcopy(V)
        # these things are set regardless of status of initial parameters
        # velocities (this is m_t in Adam paper)
        self.C_vel = np.zeros(shape=self.C.shape)
        self.G_vel = np.zeros(shape=self.G.shape)
        self.V_vel = np.zeros(shape=self.V.shape)
        # acceleration (this is v_t in Adam paper)
        self.C_acc = np.zeros(shape=self.C.shape)
        self.G_acc = np.zeros(shape=self.G.shape)
        self.V_acc = np.zeros(shape=self.V.shape)
        # fix some parameters?
        # (never update these)
        self.fix_words = options['fix_words']
        self.fix_relas = options['fix_relas']
        # special type of relationship (translations only)
        self.trans_rela = options['trans_rela']

    def update(self, grad_parameters, alpha, mu, 
               nu=None, kappa=[0, 0, 0], ADAM=True, NORMALISE=False):
        """
        Updates parameters.
        Note: assumes alpha, mu, nu are pre-updated.
        """
        # unwrap
        gradC, gradG, gradV = grad_parameters
        # regularise (REGOPT 1)
        if sum(kappa) > 0:
            gradC[:, :-1] -= kappa[0]*self.C[:, :-1]
            gradG[1:, :-1, :-1] -= kappa[1]*self.G[1:, :-1, :-1]
            gradV[:, :-1] -= kappa[2]*self.V[:, :-1]
        alphaC, alphaG, alphaV = alpha
        muC, muG, muV = mu
        # update velocities
        if not self.fix_words:
            self.C_vel = muC*self.C_vel + (1-muC)*gradC
            self.V_vel = muV*self.V_vel + (1-muV)*gradV
        if not self.fix_relas:
            self.G_vel = muG*self.G_vel + (1-muG)*gradG
            # regularise (REGOPT 2)
            # old...
            #self.G_vel[1:, :-1, :-1] -= kappa*self.G[1:, :-1, :-1]
        if ADAM:
            nuC, nuG, nuV = nu
            # accels (elementwise squaring)
            gradsqC = gradC*gradC
            gradsqG = gradG*gradG
            gradsqV = gradV*gradV
            # update accelerations
            if not self.fix_words:
                self.C_acc = nuC*self.C_acc + (1-nuC)*gradsqC
                self.V_acc = nuV*self.V_acc + (1-nuV)*gradsqV
            if not self.fix_relas:
                self.G_acc = nuG*self.G_acc + (1-nuG)*gradsqG
            # get update-specific alphas
            alphaC_hat = alphaC*np.sqrt(1-nuC)/(1-muC)
            alphaG_hat = alphaG*np.sqrt(1-nuG)/(1-muG)
            alphaV_hat = alphaV*np.sqrt(1-nuV)/(1-muV)
            # how to increment the parameter?
            deltaC = self.C_vel/(np.sqrt(self.C_acc) + EPSILON)
            # possibly add regulariser here (before variance normalisation, or
            # after ... test this)
            # the other option is to put it in the 'true' gradient, but this
            # may change the final solution...
            deltaG = self.G_vel/(np.sqrt(self.G_acc) + EPSILON)
            # regularise (REGOPT 3)
            # ... old
            #deltaG[1:, :-1, :-1] -= kappa*self.G[1:, :-1, :-1]
            deltaV = self.V_vel/(np.sqrt(self.V_acc) + EPSILON)
        else:
            deltaC = self.C_vel
            deltaG = self.G_vel
            deltaV = self.V_vel
            alphaC_hat = alphaC
            alphaG_hat = alphaG
            alphaV_hat = alphaV
        # update parameters
        if not self.fix_words:
            self.C += alphaC_hat*deltaC
            self.V += alphaV_hat*deltaV
        if not self.fix_relas:
            if self.trans_rela:
                # only update the final column of G
                self.G[:, :, -1] += alphaG_hat*deltaG[:, :, -1]
            else:
                self.G += alphaG_hat*deltaG
        if NORMALISE:
            # normalise vectors to 1, matrices to have max element = 1 (weird, weird)
            # the vectors are simple
            self.C[:, :-1] /= np.linalg.norm(self.C[:, :-1]).reshape(-1,1)
            self.V[:, :-1] /= np.linalg.norm(self.V[:, :-1]).reshape(-1,1)
            # the matrices are less simple
            for r in xrange(self.R):
                self.G[r, :-1, :] /= np.max(abs(self.G[r, :-1, :]))

    def grad_E(self, locations):
        """
        Gradients of the energy, evaluated at a list of triples.
        NOTE: this clearly depends on the choice of energy.
        Returns tensors whose first index corresponds to the input triple list.
        """
        C_sub = self.C[locations[:, 0]]
        G_sub = self.G[locations[:, 1]]
        V_sub = self.V[locations[:, 2]]
        if self.etype == 'dot':
            # TODO: make this efficient
            dE_C = -np.einsum('...i,...ij', V_sub, G_sub)
            dE_G = -np.einsum('...i,...j', V_sub, C_sub)
            dE_V = -np.einsum('...ij,...j', G_sub, C_sub)
        elif self.etype == 'euclidean':
            # TODO: make efficient, probably
            # NOTE: applying G to V, not C
            GV_C = np.einsum('...ij,...j', G_sub, V_sub) - C_sub
            lens = np.linalg.norm(GV_C, axis=1).reshape(-1, 1)
            GV_Cl = GV_C/lens
            dE_C = 1.0*GV_C/lens
            dE_G = -np.einsum('...i,...j', GV_Cl, V_sub)
            dE_V = -np.einsum('...i,...ij', GV_Cl, G_sub)
        elif self.etype =='angular':
            # TODO: make efficient, probably
            # also test
            # NOTE: applying G to V, not C
            GV = np.einsum('...ij,...j', G_sub, V_sub)
            GVC = np.einsum('...i,...i', GV, C_sub).reshape(-1, 1)
            GV_len = np.linalg.norm(GV, axis=1).reshape(-1, 1)
            C_len = np.linalg.norm(C_sub, axis=1).reshape(-1, 1)
            # yolo
            print GV_len.shape
            print C_len.shape
            print GVC.shape
            print (C_sub/C_len).shape
            print GV.shape
            print 'csubshape'
            print C_sub.shape
            print C_len.shape
            dE_C = 1.0/(GV_len*C_len*C_len)*(C_len*GV - GVC*(C_sub/C_len))
            prefactor = 1.0/(GV_len*GV_len*C_len)
            print prefactor.shape
            dE_G = prefactor*(GV_len*np.einsum('...i,...j', C_sub, V_sub) - (GVC/GV_len)*(np.einsum('...i,...j', GV, V_sub)))
            dE_V = prefactor*(GV_len*np.einsum('...ij,...i', G_sub, C_sub) - (GVC/GV_len)*(np.einsum('...i,...ij', GV, G_sub)))
        elif self.etype == 'cosine':
            # TODO: make efficient, I assume. Everything else has that TODO.
            GC = np.einsum('...ij,...j', G_sub, C_sub)
            VGC = np.einsum('...i,...i', V_sub, GC)
            GC_lens = np.linalg.norm(GC, axis=1)
            V_lens = np.linalg.norm(V_sub, axis=1)
            VG = np.einsum('...i,...ij', V_sub, G_sub)
            GCG = np.einsum('...i,...ij', GC, G_sub)
            term1_norm = (V_lens*GC_lens)
            term2_norm = (V_lens*pow(GC_lens, 3))
            # dE_C
            dE_C = (-VG/term1_norm.reshape(-1, 1) + 
                   VGC.reshape(-1, 1)*GCG/term2_norm.reshape(-1, 1))
            # dE_G
            VC = np.einsum('...i,...j', V_sub, C_sub)
            GCC = np.einsum('...i,...j', GC, C_sub)
            dE_G = (-VC/term1_norm.reshape(-1, 1, 1) +
                   VGC.reshape(-1, 1, 1)*GCC/term2_norm.reshape(-1, 1, 1))
            # dE_V
            term2_norm_V = GC_lens*pow(V_lens, 3)
            dE_V = (-GC/term1_norm.reshape(-1, 1) +
                   VGC.reshape(-1, 1)*V_sub/term2_norm_V.reshape(-1, 1))
        elif self.etype == 'frobenius':
            GC = np.einsum('...ij,...j', G_sub, C_sub)
            VG = np.einsum('...i,...ij', V_sub, G_sub)
            VC = np.einsum('...i,...j', V_sub, C_sub)
            VGC = np.einsum('...i,...i', V_sub, GC)
            # the norms
            C_len = np.linalg.norm(C_sub, axis=1).reshape(-1, 1)
            V_len = np.linalg.norm(V_sub, axis=1).reshape(-1, 1)
            G_len = np.linalg.norm(G_sub, ord='fro', axis=(1, 2)).reshape(-1, 1, 1)
            # dE_C
            dE_C = -VG/C_len + (VGC.reshape(-1, 1))*C_sub/pow(C_len, 3)
            # dE_G
            dE_G = -VC/G_len + (VGC.reshape(-1, 1, 1))*G_sub/pow(G_len, 3)
            # dE_V
            dE_V = -GC/V_len + (VGC.reshape(-1, 1))*V_sub/pow(V_len, 3)
        else:
            sys.exit('ERROR: Not implemented (gradE)')
        return dE_C, dE_G, dE_V

    def noR_batch_gradient(self, batch, omega):
        """
        In the case of training examples with *unknown* relationship 
            (so triples missing the R term),
        the contribution to the gradient is a weighted sum of derivatives,
        weighted by the conditional probability of the missing relationship.

        In this case, we need to calculate grad_E at many more locations
            (R times as many, for each S,T pair)
        and then create a weighted contribution.

        Note that batch is assumed to have shape[1] == 3, but its 2nd column
        is empty/nonsense.
        """
        W = self.W
        R = self.R
        d = self.d
        dC_batch = np.zeros(shape=(W, d+1))
        dG_batch = np.zeros(shape=(R, d+1, d+1))
        dV_batch = np.zeros(shape=(W, d+1))
        # create virtual batch (repeat each entry R times)
        b = batch.shape[0]
        virtual_batch = batch.repeat(R, axis=0)
        virtual_batch[:, 1] = range(R)*b
        # get virtual grads
        dE_C_virtual_batch, dE_G_virtual_batch, dE_V_virtual_batch = self.grad_E(virtual_batch)
        # iterate through, calculating weighting factors (conditional probability of R)
        for (i, (s, _, t)) in enumerate(batch):
            energy = self.E_axis((s, _, t), 'R')
            expmE = np.exp(-energy)
            # omega is an additional weighting here - note, this definitely
            # messes up the probabilistic interpretation of this step
            probs = (expmE/np.sum(expmE))*omega
            dC_batch[s, :] -= np.dot(probs, dE_C_virtual_batch[i:(i+R), :])
            dG_batch[:, :, :] -= probs.reshape(-1, 1, 1)*dE_G_virtual_batch[i:(i+R), :, :]
            dV_batch[t, :] -= np.dot(probs, dE_V_virtual_batch[i:(i+R), :])
        return (dC_batch, dG_batch, dV_batch)

    def visible_batch_gradient(self, batch, omega):
        """
        Gradient is a difference of contributions from:
        1. data distribution (batch of training examples)
        2. model distribution (batch of model samples)
        In both cases, we need to evaluate a gradient over a batch of triples.
        This is a general function for both tasks
        (so we expect to call it twice for each 'true' gradient evaluation.)

        omega is a vector of weights associated with relationships
        (length = R)
        each gradient contribution is scaled by omega_r
        """
        W = self.W
        R = self.R
        d = self.d
        dC_batch = np.zeros(shape=(W, d+1))
        dG_batch = np.zeros(shape=(R, d+1, d+1))
        dV_batch = np.zeros(shape=(W, d+1))
        dE_C_batch, dE_G_batch, dE_V_batch = self.grad_E(batch)
        for (i, (s, r, t)) in enumerate(batch):
            prefactor = omega[r]
            dC_batch[s, :] -= prefactor*dE_C_batch[i]
            dG_batch[r, :, :] -= prefactor*dE_G_batch[i]
            dV_batch[t, :] -= prefactor*dE_V_batch[i]
        return (dC_batch, dG_batch, dV_batch)

    def E_axis(self, triple, switch):
        """
        Returns energies over an axis (S, R, T) given two of the triple.
        """
        s, r, t = triple
        if switch == 'S':
            # return over all S
            #GC = np.dot(self.C, self.G[r].T)
            #energy = -np.dot(GC, self.V[t])
            # note: above version is significantly slower than the below
            if self.etype == 'dot':
                VG = np.dot(self.V[t], self.G[r])
                energy = -np.dot(self.C, VG)
            elif self.etype == 'euclidean':
                GV = np.dot(self.G[r, :, :], self.V[t, :])
                energy = -np.linalg.norm(GV - self.C, axis=1)
            elif self.etype == 'angular':
                GV = np.dot(self.G[r, :, :], self.V[t, :])
                GVC = np.einsum('...j,...ij', GV, self.C)
                GV_len = np.linalg.norm(GV)
                C_len = np.linalg.norm(self.C, axis=1)
                energy = 1 - (1/pi)*np.arccos(GVC/(GV_len*C_len))
            elif self.etype == 'cosine':
                CG = np.dot(self.C, self.G[r].T)
                CG_lens = np.linalg.norm(CG, axis=1)
                numerator = -np.dot(CG, self.V[t, :])
                denominator = np.linalg.norm(self.V[t, :])*CG_lens
                energy = numerator/denominator
            elif self.etype == 'frobenius':
                # mostly copied from cosine
                CG = np.dot(self.C, self.G[r].T)
                numerator = -np.dot(CG, self.V[t, :])
                C_lens = np.linalg.norm(self.C, axis=1)
                G_len = np.linalg.norm(self.G[r, :, :], ord='fro')
                V_len = np.linalg.norm(self.V[t, :])
                denominator = C_lens*G_len*V_len
                energy = numerator/denominator
            else: sys.exit('ERROR: Not implemented (E_axis)')
        elif switch == 'R':
            # return over all R
            if self.etype == 'dot':
                VG = np.dot(self.V[t], self.G)
                energy = -np.dot(VG, self.C[s])
            elif self.etype == 'euclidean':
                GV = np.dot(self.G[:, :, :], self.V[t, :])
                energy = -np.linalg.norm(GV - self.C[s, :], axis=1)
            elif self.etype == 'angular':
                GV = np.dot(self.G[:, :, :], self.V[t, :])
                GVC = np.einsum('...ij,...j', GV, self.C[s, :])
                GV_len = np.linalg.norm(GV, axis=1)
                C_len = np.linalg.norm(self.C[s, :])
                energy = 1 - (1/pi)*np.arccos(GVC/(GV_len*C_len))
            elif self.etype == 'cosine':
                # note this GC is different to before :)
                GC = np.dot(self.G, self.C[s, :])
                GC_lens = np.linalg.norm(GC, axis=1)
                numerator = -np.dot(GC, self.V[t, :])
                denominator = np.linalg.norm(self.V[t, :])*GC_lens
                energy = numerator/denominator
            elif self.etype == 'frobenius':
                GC = np.dot(self.G, self.C[s, :])
                GC_lens = np.linalg.norm(GC, axis=1)
                numerator = -np.dot(GC, self.V[t, :])
                C_len = np.linalg.norm(self.C[s, :])
                G_lens = np.linalg.norm(self.G, ord='fro', axis=(1, 2))
                V_len = np.linalg.norm(self.V[t, :])
                denominator = C_len*G_lens*V_len
                energy = numerator/denominator
            else: sys.exit('ERROR: Not implemented (E_axis)')
        elif switch == 'T':
            #return over all T
            if self.etype == 'dot':
                GC = np.dot(self.G[r], self.C[s])
                energy = -np.dot(self.V, GC)
            elif self.etype == 'euclidean':
                GV = np.einsum('...jk,...ik', self.G[r, :, :], self.V)
                energy = -np.linalg.norm(GV - self.C[s, :], axis=1)
            elif self.etype == 'angular':
                GV = np.einsum('...jk,...ik', self.G[r,:, :], self.V)
                GVC = np.dot(GV, self.C[s, :])
                GV_len = np.linalg.norm(GV, axis=1)
                C_len = np.linalg.norm(self.C[s, :])
                energy = 1 - (1/pi)*np.arccos(GVC/(GV_len*C_len))
            elif self.etype == 'cosine':
                GC = np.dot(self.G[r], self.C[s])
                V_lens = np.linalg.norm(self.V, axis=1)
                numerator = -np.dot(self.V, GC)
                denominator = np.linalg.norm(GC)*V_lens
                energy = numerator/denominator
            elif self.etype == 'frobenius':
                GC = np.dot(self.G[r], self.C[s])
                numerator = -np.dot(self.V, GC)
                C_len = np.linalg.norm(self.C[s, :])
                G_len = np.linalg.norm(self.G[r, :, :], ord='fro')
                V_lens = np.linalg.norm(self.V, axis=1)
                denominator = C_len*G_len*V_lens
                energy = numerator/denominator
            else: sys.exit('ERROR: Not implemented (E_axis)')
        else:
            print 'ERROR: Cannot parse switch.'
            sys.exit()
        return energy

    def E_triple(self, triple):
        """
        The energy of a SINGLE triple.
        """
        s, r, t = triple
        if self.etype == 'dot':
            energy = -np.dot(self.V[t], np.dot(self.G[r], self.C[s]))
        elif self.etype == 'euclidean':
            energy = -np.linalg.norm(np.dot(self.G[r], self.V[t]) - self.C[s])
        elif self.etype == 'angular':
            GV = np.dot(self.G[r], self.V[t])
            GVC = np.dot(GV, self.C[s])
            GV_len = np.linalg.norm(GV)
            C_len = np.linalg.norm(self.C[s])
            energy = 1 - (1/pi)*np.arccos(GVC/(GV_len*C_len))
        elif self.etype == 'cosine':
            GC = np.dot(self.G[r], self.C[s])
            GC_norm = np.linalg.norm(GC)
            V_norm = np.linalg.norm(self.V[t])
            energy = -np.dot(self.V[t], GC)/(GC_norm*V_norm)
        elif self.etype == 'frobenius':
            GC = np.dot(self.G[r], self.C[s])
            numerator = -np.dot(self.V[t], GC)
            C_norm = np.linalg.norm(self.C[s, :])
            G_norm = np.linalg.norm(self.G[r, :, :], ord='fro')
            V_norm = np.linalg.norm(self.V[t, :])
            denominator = C_norm*G_norm*V_norm
            energy = numerator/denominator
        else: sys.exit('ERROR: Not implemented (E_triple)')
        return energy

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
        if locations is None:
            W = self.W
            R = self.R
            locations = np.array([[s, r, t] for s in xrange(W) for r in xrange(R) for t in xrange(W) ])
        energy = np.empty(shape=len(locations), dtype=np.float)
        if self.etype == 'dot':
            for (i, triple) in enumerate(locations):
                energy[i] = -np.dot(self.C[triple[0]], 
                                    np.dot(self.V[triple[2]], 
                                           self.G[triple[1]]))
        elif self.etype == 'euclidean':
            for (i, triple) in enumerate(locations):
                energy[i] = -np.linalg.norm(np.dot(self.G[triple[1]], 
                                                   self.V[triple[2]]) -\
                                            self.C[triple[0]])
         # haha wtf is going on here why did I do this
        elif self.etype == 'angular':
            for (i, triple) in enumerate(locations):
                energy[i] = self.E_triple(triple)
        elif self.etype == 'cosine':
            for (i, triple) in enumerate(locations):
                energy[i] = self.E_triple(triple)
        elif self.etype == 'frobenius':
            for (i, triple) in enumerate(locations):
                energy[i] = self.E_triple(triple)
        else: sys.exit('ERROR: Not implemented (E)')
        # V10
        #energy = []
        #for triple in locations:
        #    energy.append(-np.dot(self.C[triple[0]], np.dot(self.V[triple[2]], self.G[triple[1]])))
        return energy

    def marginal(self, switch):
        """
        Calculates marginal distribution of a single type of var: S, R, T.
        WARNING: may be *extremely* slow/memory intensive for big data.
            (calculates partition function)
        """
        W = self.W
        R = self.R
        energies = self.E().reshape(W, R, W)
        expmE = np.exp(-energies)
        Z = np.sum(expmE)
        if switch == 'S':
            marginal_numerator = np.sum(expmE, axis=(1,2))
        elif switch == 'R':
            marginal_numerator = np.sum(expmE, axis=(0,2))
        elif switch == 'T':
            marginal_numerator = np.sum(expmE, axis=(0,1))
        marginal_distribution = marginal_numerator/Z
        return marginal_distribution 

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
                    energy = self.E_axis(ss, 'S')
                    #locs = np.array([ [i, ss[1], ss[2]] for i in xrange(W) ])
                if triple_drop == 1:
                    energy = self.E_axis(ss, 'R')
                    #locs = np.array([ [ss[0], i, ss[2]] for i in xrange(R) ])
                if triple_drop == 2:
                    energy = self.E_axis(ss, 'T')
                    #locs = np.array([ [ss[0], ss[1], i] for i in xrange(W) ])
                #expmE = np.exp(-self.E(locs))
                expmE = np.exp(-energy)
                probs = expmE/np.sum(expmE)
                samp = np.random.choice(len(probs), p=probs, size=1)[0]
                ss[triple_drop] = samp
        return ss
    
    def complete_triple(self, triple, n_samples=1):
        """
        Completes a triple (['sword', 'rela', 'tword']) with a missing entry,
        expects STRINGS belonging to the vocabulary. (human readable)
        
        the n_samples option allows multiple completions to be generated
        """
        lens = map(len, triple)
        if not lens.count(0) == 1:
            print 'ERROR: please drop exactly one entry from the triple'
            return False
        s, r, t = triple
        # which axis we'll need to look over
        triple_drop = lens.index(0)
        if triple_drop == 0:
            si = 0
        else:
            try:
                si = self.words.index(s)
            except ValueError:
                print 'ERROR:', s, 'is not in vocabulary.'
                return False
        if triple_drop == 1:
            ri = 0
        else:
            try:
                ri = self.relas.index(r)
            except ValueError:
                print 'ERROR:', r, 'is not in vocabulary.'
                return False
        if triple_drop == 2:
            ti = 0
        else:
            try:
                ti = self.words.index(t)
            except ValueError:
                print 'ERROR:', t, 'is not in vocabulary.'
                return False
        # now do sampling
        energy = self.E_axis((si, ri, ti), ['S', 'R', 'T'][triple_drop])
        expmE = np.exp(-energy)
        probs = expmE/np.sum(expmE)
        samples = np.random.choice(len(probs), p=probs, size=n_samples)
        completed_triples = np.array([triple]*n_samples)
        if triple_drop == 1:
            # rela
            sampled_list = [self.relas[s] for s in samples]
        else:
            # word
            sampled_list = [self.words[s] for s in samples]
        completed_triples[:,triple_drop] = sampled_list
        return completed_triples

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
            C_vals = {'words': self.words, 'vecs': self.C[:, :-1]}
            G_vals = {'relas': self.relas, 'mats': self.G}
            V_vals = {'words': self.words, 'vecs': self.V[:, :-1]}
            np.save(re.sub('XXX','C',filename), C_vals)
            np.save(re.sub('XXX','G',filename), G_vals)
            np.save(re.sub('XXX','V',filename), V_vals)
        elif '.txt' in filename:
            fC = open(re.sub('XXX','C',filename), 'w')
            fG = open(re.sub('XXX','G',filename), 'w')
            fV = open(re.sub('XXX','V',filename), 'w')
            fC.write(str(self.W)+' '+str(self.C.shape[1]-1)+'\n')
            fG.write(str(self.R)+' '+str(self.C.shape[1]-1)+'\n')
            fV.write(str(self.W)+' '+str(self.C.shape[1]-1)+'\n')
            for i in xrange(self.W):
                word = self.words[i]
                fC.write(word+' '+' '.join(map(str, self.C[i,:-1]))+'\n')
                fV.write(word+' '+' '.join(map(str, self.V[i,:-1]))+'\n')
            for i in xrange(self.R):
                rela = self.relas[i]
                fG.write(rela+' '+' '.join(map(str, self.G[i,:-1,:].reshape((self.C.shape[1])*(self.C.shape[1]-1),)))+'\n')
            fC.close()
            fV.close()
            fG.close()
        return True

    def load(self, filename):
        """
        Method to load parameters from a file, as encoded by 'save'
        Please note that, at present, this can't know ordering
         - that is, it may assign words new indices
        Which means it is not necessarily compatible with arbitrary training data,
        or even the training data used to create the parameters that are being loaded.
        To fix this, save method must also record these indices more faithfully...
        For .txt the order in the file is actually faithful, but for npy, 
        since they are saved as dictionaries, we cannot guarantee this.
        Sorry!
        """
        if not 'XXX' in filename:
            print 'WARNING: Load expects an XXX in the filename. Fixed that for you.'
            filename = filename+'_XXX'
        if '.npy' in filename:
            print 'WARNING: behaviour has changed recently'
            C_dict = np.load(re.sub('XXX','C',filename)).item()
            C_words = C_dict['words']
            C_pruned = C_dict['vecs']
            G_dict = np.load(re.sub('XXX','G',filename)).item()
            relas = G_dict['relas']
            G = G_dict['mats']
            V_words, V_pruned = np.load(re.sub('XXX','V',filename)).item()
            V_dict = np.load(re.sub('XXX','V',filename)).item()
            V_words = V_dict['words']
            V_pruned = V_dict['vecs']
            assert C_words == V_words
            words = C_words
            W = len(words)
            R = len(relas)
            d = C_pruned.shape[1]
            assert V_pruned.shape == C_pruned.shape
            assert G.shape[1] == d+1
            assert G.shape[0] == R
            # extend to include the trailing 1
            C = np.ones(shape=(W, d+1))
            V = np.ones(shape=(W, d+1))
            C[:, :-1] = C_pruned
            V[:, :-1] = V_pruned
        elif '.txt' in filename:
            # try C and V first...
            fC = open(re.sub('XXX','C',filename), 'r')
            fV = open(re.sub('XXX','V',filename), 'r')
            C_header = fC.readline()
            V_header = fV.readline()
            assert C_header == V_header
            W, d = map(int, C_header.split())
            # create empty
            C = np.ones(shape=(W, d+1))
            V = np.ones(shape=(W, d+1))
            words = []
            for (i, line) in enumerate(fC):
                sl = line.split()
                word = sl[0]
                vec = np.array(map(np.float, sl[1:]))
                words.append(word)
                C[i, :-1] = vec[:]
            for line in fV:
                sl = line.split()
                word = sl[0]
                vec = np.array(map(np.float, sl[1:]))
                i = words.index(word)
                V[i, :-1] = vec[:]
            # now for G
            try:
                fG = open(re.sub('XXX','G',filename), 'r')
                G_header = fG.readline()
                R, d = map(int, G_header.split())
                # create empty matrices
                G = np.zeros(shape=(R, d+1, d+1))
                relas = []
                for (i, line) in enumerate(fG):
                    sl = line.split()
                    rela = sl[0]
                    relas.append(rela)
                    matrix = np.array(map(np.float, sl[1:])).reshape(d, d+1)
                    G[i, :-1, :] = matrix[:, :]
                    G[i, -1, -1] = 1
            except IOError:
                print 'WARNING: no G matrices found.'
                G = np.array([np.eye(d+1)])
                R = 1
                relas = ['0']
        self.W = W
        self.R = R
        self.d = d
        self.words = words
        self.relas = relas
        self.C = deepcopy(C)
        self.G = deepcopy(G)
        self.V = deepcopy(V)
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
    ll = np.sum([(-energy[s, r, t] - logZ) for s, r, t in data if not r == MISS_R])
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

def train(training_data, start_parameters, options, VERBOSE=True):
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
    EXACT = options['exact']
    PERSISTENT = options['persistent']
    NOISE = options['noise']
    ADAM = options['adam']
    NORMALISE = options['normalise']
    if NOISE: EXACT=False
    if EXACT or NOISE: PERSISTENT=False
    B = options['batch_size']
    S = options['sampling_rate']
    M = options['num_samples']
    D = options['diagnostics_rate']
    K = options['gibbs_iterations']
    calc_ll = options['calc_ll']
    alpha0 = options['alpha']
    mu = options['mu']
    kappa = options['kappa']
    try:
        nu =  options['nu']
    except KeyError:
        nu = None
    mu_t = mu[:]
    alpha = alpha0[:]
    tau = options['tau']
    output_root = options['output_root']
    print output_root
    offset = options['offset']
    try:
        vali_set_size = options['vali_set_size']
    except KeyError:
        # YMMV
        vali_set_size = 1000
    # initialise
    vali_set = set()
    vali_set_done = False
    batch = np.empty(shape=(B, 3),dtype=np.int)
    # TODO: proper sample initialisation
    samples = np.zeros(shape=(M, 3),dtype=np.int)
    if not type(start_parameters) == params:
        parameters = params(start_parameters)
    else:
        parameters = start_parameters
    # diagnostic things
    logf = open(output_root+'_logfile.txt','a')
    W = parameters.W
    R = parameters.R
    try:
        omega = options['omega']
        assert len(omega) == R
    except KeyError:
        # no downweighting!
        omega = [1]*R
    # a fixed permutation, for testing my strange likelihood ratio thing
    W_perm = dict(enumerate(np.random.permutation(W)))
    R_perm = dict(enumerate(np.random.permutation(R)))
    R_perm[MISS_R] = MISS_R
    # record sampling frequencies
    #sampled_counts = dict((i, 0) for i in xrange(W))
    n = 0
    t0 = time.time()
    for example in training_data:
        if len(vali_set) < vali_set_size and not example[1] == MISS_R:
            vali_set.add(tuple(example))
        if len(vali_set) == vali_set_size and not vali_set_done:
            perm_vali_batch = permute_batch(W_perm, R_perm, np.array(list(vali_set)))
            vali_set_done = True
        # do not train on validation set
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
            delta_model = parameters.visible_batch_gradient(samples, omega)
            prefactor = float(B)/len(samples)
        if n%B == 0 and n > S:
            if EXACT:
                delta_model = Z_gradient(parameters)
                prefactor = float(B)
            # split into visible R and no R
            visible_mask = batch[:,1] != MISS_R
            visible_batch = batch[visible_mask, :]
            delta_visible_data = parameters.visible_batch_gradient(visible_batch, omega)
            noR_batch = batch[~visible_mask, :]
            delta_noR_data = parameters.noR_batch_gradient(noR_batch, omega)
            # combine
            delta_data = (delta_visible_data[0] + delta_noR_data[0],
                          delta_visible_data[1] + delta_noR_data[1],
                          delta_visible_data[2] + delta_noR_data[2])
            # combine with model samples to get final gradients
            delta_params = combine_gradients(delta_data, delta_model, prefactor)
            if ADAM:
                mu_t = mu_t*LAMBDA
            else:
                if not 0 in tau:
                    alpha = alpha0/(1+(n+offset)/(tau*B))
            parameters.update(delta_params, alpha, mu_t,
                              nu, kappa=kappa, ADAM=ADAM, NORMALISE=NORMALISE)
        if D > 0:
            # if D == 0 or < 0, this means NO DIAGNOSTICS ARE RUN
            # the reason this is an option is clearly speed
            if n%D == 0 and n > B and n > S:
                t = time.time() - t0
                if calc_ll:
                    ll = log_likelihood(parameters, training_data)
                    #ll = log_likelihood(parameters, vali_set)
                else:
                    ll = 'NA'
                if len(visible_batch) > 0:
                    data_energy = np.mean(parameters.E(visible_batch))
                else:
                    data_energy = 'NA'
                if len(vali_set) > 0:
                    vali_energy = np.mean(parameters.E(np.array(list(vali_set))))
                else:
                    vali_energy = 'NA'
                random_lox = np.array(zip(np.random.randint(0, W, 100),
                                          np.random.randint(0, R, 100),
                                          np.random.randint(0, W, 100)))
                rand_energy = np.mean(parameters.E(random_lox))
                # so this is different to the rand, cause it's a permuted version of the validation set
                if vali_set_done:
                    perm_energy = np.mean(parameters.E(perm_vali_batch))
                else:
                    perm_energy = 'NA'
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
                logline = [n + offset, t, ll, 
                           data_energy, model_energy, vali_energy,
                           rand_energy, perm_energy,
                           C_lens, G_lens, V_lens]
                if VERBOSE:
                    print '\t', logline[0],
                    for val in logline[1:]:
                        if type(val) == str: 
                            print '\t', val,
                        else:
                            print '\t','%.3f' % val,
                print ''
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
                # yololo
                # record performance on devset (nips2013)
                #devset_accuracy(devpath, devlogpath, parameters, n + offset)
                # endyolo
            if n%(D*100) == 0:
                parameters.save(output_root+'_XXX.npy')
                if VERBOSE:
                    print 'Saved parameters to', output_root+'_XXX.npy'
    logf.close()
    if VERBOSE: print 'Training done,', n, 'examples seen.'
    parameters.save(output_root+'_XXX.npy')
    options['alpha'] = alpha
    options['offset'] += n
    return vali_set
