#!/bin/python
# The independent implementation is growing in size.
# 
import bf2f as bf2f
import cProfile
import re
from subprocess import call
import sys
import os.path

DATA = 'cfg'

#ONLINE=True
ONLINE=False
#EXACT=True
EXACT=False
PERSISTENT=True
#PERSISTENT=False
NOISE=False

if EXACT or NOISE: PERSISTENT=False
#CALC_LL=True
CALC_LL = False
#CALC_LL=not(ONLINE)

# --- paths --- #
outroot = './output'
droot = './data'

# --- encode options in filename --- #
if ONLINE:
    online_or_batch = 'online'
else:
    online_or_batch = 'batch'
if NOISE:
    exact_or_not = 'noise'
    persistent_or_not = ''
    EXACT = False
    PERSISTENT = False
else:
    if EXACT:
        exact_or_not = 'exact'
    else:
        exact_or_not = 'inexact'
    if PERSISTENT:
        persistent_or_not = 'persistent'
    else:
        persistent_or_not = 'nonpersistent'

if bf2f.ADAM:
    train_method = 'ADAM'
else:
    train_method = 'SGD'

if bf2f.NORMALISE:
    normed = 'normed'
else:
    normed = 'unnormed'

etype = bf2f.ETYPE
 
# --- training options --- #
if len(sys.argv) > 1:
    options = bf2f.load_options(sys.argv[1])
    d = options['dimension']
    D = options['diagnostics_rate']
    fix_words = options['fix_words']
    fix_relas = options['fix_relas']
    trans_rela = options['trans_rela']
    n_epochs = options['n_epochs']
    CALC_LL = options['calculate_ll']
    # optional options :)
    if 'wordlist' in options:
        wordlist = options['wordlist']
    if 'relalist' in options:
        wordlist = options['wordlist']
else:
    print 'No options file provided, falling back to defaults.'
    # default options
    B = 100
    S = 100
    M = 5
    D = 1000
    K = 1
    d = 100
    vali_set_size = 3
    alphaC, alphaG, alphaV = 0.01, 0.01, 0.01
    muC, muG, muV = 0.9, 0.9, 0.9
    nuC, nuG, nuV = 0.999, 0.999, 0.999           # nu required for Adam
    tauC, tauG, tauV = 0, 0, 0
    fix_words = False
    fix_relas = False
    trans_rela = True
    n_epochs = 1
    offset = 0
    options = {'dimension':d,
               'batch_size':B,
               'diagnostics_rate':D,
               'sampling_rate':S,
               'gibbs_iterations':K,
               'num_samples':M,
               'alpha':bf2f.np.array([alphaC, alphaG, alphaV]),
               'mu':bf2f.np.array([muC, muG, muV]),
               'nu':bf2f.np.array([nuC, nuG, nuV]),
               'tau':bf2f.np.array([tauC, tauG, tauV]),
               'calculate_ll':CALC_LL,
               'vali_set_size':vali_set_size,
               'fix_words':fix_words,
               'fix_relas':fix_relas,
               'trans_rela':trans_rela,
               'n_epochs':n_epochs,
               'offset':offset}
    for (key, value) in options.iteritems():
        print value, '\t:', key
    # note that some of these options are not used by bf2f

if 'name' in options:
    # THIS IS DANGEROUS
    fname = options['name']
    if 'batch' in fname:
        assert not ONLINE
    if 'inexact' in fname:
        assert not EXACT
    if 'nonpersistent' in fname:
        assert not PERSISTENT
    if 'noise' in fname:
        assert NOISE
    if 'ADAM' in fname:
        assert bf2f.ADAM
    if 'SGD' in fname:
        assert bf2f.SGD
else:
    fname = outroot+'/'+DATA+'_'+online_or_batch+'_'+exact_or_not+'_'+persistent_or_not+'_'+str(d)+'d_'+train_method+'_'+normed+'_'+etype
    options['name'] = fname
paramfile = fname+'_XXX.txt'
valifile = fname+'_valiset.txt'
optionsfile = fname+'_options.txt'
   
# --- datafiles --- #

# overwrite from the options file
# WARNING: DANGEROUS
if 'dpath' in options:
    dpath = options['dpath']
else:
    print 'No training data provided, using toy data.'
    W, R = 1000, 5
    dpath = droot+'/w'+str(W)+'r'+str(R)+'_train.txt.gz'
    if not os.path.isfile(dpath):
        # create data if it doesn't exist
        bf2f.generate_traindata(droot, W, R)
        assert os.path.isfile(dpath)
    options['dpath'] = dpath

# save the options
fo = open(optionsfile,'w')
for (option, value) in options.iteritems():
    if type(value) == bf2f.np.ndarray:
        value = tuple(value)
    fo.write(option+' '+str(value)+'\n')
fo.close()

# --- initialise things --- #
dstream = bf2f.data_stream(dpath)
W, R = dstream.get_vocab_sizes()
if W == 5 and d == 3:
    C = bf2f.np.array([[ 0.01481961, -0.01517603,  0.00596634,  1.        ],
                       [-0.0080693 ,  0.00852271, -0.00106983,  1.        ],
                       [-0.0012176 ,  0.02482517,  0.01040345,  1.        ],
                       [ 0.00962732,  0.0100687 ,  0.00756443,  1.        ],
                       [ 0.00841503,  0.00188252,  0.02689446,  1.        ]])
    V = bf2f.np.array([[-0.00878185, -0.01871243, -0.01610301,  1.        ],
                       [-0.02036443, -0.02137387,  0.00874672,  1.        ],
                       [ 0.00898955,  0.00722872, -0.00504091,  1.        ],
                       [ 0.00324052,  0.02674052,  0.00166536,  1.        ],
                       [ 0.01199952,  0.00430334,  0.0040228 ,  1.        ]])
#elif W == 5 and d == 2:
#    C = bf2f.np.array([[ 1.0, 0.0, 1.0 ],
#                       [ 0.0, 1.0, 1.0 ],
#                       [ 1.0, 0.0, 1.0 ],
#                       [ 0.0, 1.0, 1.0 ],
#                       [ 1.414, 1.414, 1.0 ]])
#    V = bf2f.np.array([[ 1.0, 0.0, 1.0 ],
#                       [ 0.0, 1.0, 1.0 ],
#                       [ 1.0, 0.0, 1.0 ],
#                       [ 0.0, 1.0, 1.0 ],
#                       [ 1.414, 1.414, 1.0 ]])
else:
    C = bf2f.np.random.normal(scale=0.1, size=(W, d+1))
    V = bf2f.np.random.normal(scale=0.1, size=(W, d+1))

G = bf2f.np.random.normal(scale=0.01, size=(R, d+1, d+1))
G[0, :, :] = bf2f.np.eye(d+1)
G[:, -1, :] = 0
G[:, -1, -1] = 1
C[:,-1] = 1
V[:,-1] = 1

if D > 0:
    DIAGNOSTICS = True
else:
    DIAGNOSTICS = False

# --- get vocabulary (words, relationships) --- #
try:
    words_raw = open(wordlist, 'r').readlines()
    print 'Reading vocabulary from', wordlist
    words = ['']*len(words_raw)
    for line in words_raw:
        index = int(line.split()[0])
        word = line.split()[1]
        words[index] = word
except NameError:
    print 'WARNING: wordlist doesn\'t exist.'
    # wordlist doesn't exist
    words = map(str, range(W))
try:
    relas_raw = open(relalist, 'r').readlines()
    relas = ['']*len(relas_raw)
    for line in relas_raw:
        index = int(line.split()[0])
        rela = line.split()[1]
        relas[index] = rela
except NameError:
    print 'WARNING: relalist doesn\'t exist.'
    # relalist doesn't exist
    relas = map(str, range(R))

vocab = {'words':words, 'relas': relas}

# --- actually get the training data --- #
if ONLINE:
    train_data = dstream
else:
    train_data = dstream.acquire_all()

# --- initialise parameters --- #
pp = bf2f.params((C, G, V), vocab, fix_words=fix_words, fix_relas=fix_relas, trans_rela=trans_rela)

# --- ll before --- #
if CALC_LL:
    print 'pre ll:', bf2f.log_likelihood(pp, train_data)

# --- start the logfile --- #
if DIAGNOSTICS:
    logf = open(fname+'_logfile.txt','w')
    logf.write('n\ttime\tll\tdata_energy\tmodel_energy\tvaliset_energy\trandom_energy\tperm_energy\tC_lens\tG_lens\tV_lens\n')
    logf.close()
else:
    print 'WARNING: no diagnostics.'

# ---- TRAIN! --- #
for epoch in xrange(n_epochs):
    print 'epoch:', epoch
    cProfile.runctx('vali_set = bf2f.train(train_data, pp, options, EXACT, PERSISTENT, NOISE)', None, locals())
    if ONLINE:
        # (the purpose of this is to shuffle the training data)
        train_data = dstream.acquire_all()

# --- save n stuff --- #
C_out, G_out, V_out = pp.get()
W = C_out.shape[0]
d = C_out.shape[1] - 1
pp.save(paramfile)

vf = open(valifile,'w')
for triple in vali_set:
    vf.write(' '.join(map(str, triple))+'\n')
vf.close()

# --- ll after --- #
if CALC_LL:
    print 'post ll:', bf2f.log_likelihood(pp, train_data)

if DIAGNOSTICS:
    # --- sure let's just call R --- #
    call('R --slave --file=plot_logfile.R --args '+fname+'_logfile.txt', shell=True)
