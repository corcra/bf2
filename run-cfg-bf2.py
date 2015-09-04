#!/bin/python
# Helper script to train a model.
# NOTE: this REQUIRES a valid cfg file.

import bf2f as bf2f
import cProfile
import re
from subprocess import call
import sys

bf2f.np.random.seed(1337)

# --- load options --- #
options = bf2f.options()
try:
    options.load(sys.argv[1], verbose=True)
except IndexError:
    sys.exit('ERROR: requires cfg file.')

# --- unroll options --- #
output_root = options['output_root']
training_data_path = options['training_data_path']
vocab = (options['wordlist'], options['relalist'])

# --- save the options --- #
options.save(output_root+'_options.txt')

# --- training data --- #
dstream = bf2f.data_stream(training_data_path)

if options['online']:
    train_data = dstream
else:
    train_data = dstream.acquire_all()

# --- initialise parameters --- #
try:
    # load parameters from file
    initial_params_path = options['initial_params_path']
    initial_params = initial_params_path
except KeyError:
    # create random
    W, R = dstream.get_vocab_sizes()
    d = options['dimension']
    # C, V
    C = bf2f.np.random.normal(scale=0.1, size=(W, d+1))
    V = bf2f.np.random.normal(scale=0.1, size=(W, d+1))
    C[:,-1] = 1
    V[:,-1] = 1
    # G
    G = bf2f.np.random.normal(scale=0.01, size=(R, d+1, d+1))
    G[0, :, :] = bf2f.np.eye(d+1)
    G[:, -1, :] = 0
    G[:, -1, -1] = 1
    initial_params = (C, G, V)

if options['diagnostics_rate'] > 0:
    DIAGNOSTICS = True
else:
    DIAGNOSTICS = False

# --- initialise parameters --- #
pp = bf2f.params(initial_params, options, vocab)

# --- ll before --- #
if options['calc_ll']:
    pre_ll = bf2f.log_likelihood(pp, train_data)
    print 'pre ll:', pre_ll

# --- start the logfile --- #
if DIAGNOSTICS:
    logf = open(output_root+'_logfile.txt','w')
    logf.write('n\ttime\tll\tdata_energy\tmodel_energy\tvaliset_energy\trandom_energy\tperm_energy\tC_lens\tG_lens\tV_lens\n')
    logf.close()

# ---- TRAIN! --- #
for epoch in xrange(options['n_epochs']):
    print 'epoch:', epoch
    cProfile.runctx('vali_set = bf2f.train(train_data, pp, options)', None, locals())
    if options['online']:
        train_data = dstream.acquire_all(SHUFFLE=True)

# --- save n stuff --- #
pp.save(output_root+'_XXX.txt')

vf = open(output_root+'_valiset.txt','w')
for triple in vali_set:
    vf.write(' '.join(map(str, triple))+'\n')
vf.close()

# --- ll after --- #
if options['calc_ll']:
    post_ll =  bf2f.log_likelihood(pp, train_data)
    print 'post ll:', post_ll

if DIAGNOSTICS:
    # --- sure let's just call R --- #
    call('R --slave --file=plot_logfile.R --args '+output_root+'_logfile.txt', shell=True)
