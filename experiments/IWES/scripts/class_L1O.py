#!/bin/python
# simple classification task

import numpy as np
# classifiers
from sklearn.ensemble import RandomForestClassifier
# crossval
from sklearn.cross_validation import StratifiedKFold
import sys
import re

try:
    miss_class = int(sys.argv[1])
except IndexError:
    sys.exit('require missing class')
try:
    params_path = sys.argv[2]
    # silly convention I use
    assert 'XXX' in params_path
    C_path = re.sub('XXX','C', params_path)
    V_path = re.sub('XXX','V', params_path)
except IndexError:
    sys.exit('require vector path')

# --- options --- #
K = 5      # (how many folds)
combine = 'concat'

# --- implement choices --- #
CONCAT = False
ADD = False
MEAN = False
if combine == 'concat':
    print 'Concatenating vectors.'
    CONCAT = True
elif combine == 'add':
    print 'Adding vectors.'
    ADD = True
elif combine == 'mean':
    print 'Averaging vectors.'
    MEAN = True
else:
    raise ValueError('Invalid combine value provided.')

clf = RandomForestClassifier()

labelled_examples_path = '../data/dev_triples_simple.txt'

# --- init --- #

# --- load vectors --- #
print 'Reading vectors from\n\t', C_path, '\n\t', V_path
if '.npy' in C_path and '.npy' in V_path:
    C = np.load(C_path)
    V = np.load(V_path)
elif '.txt' in C_path and '.txt' in V_path:
    Ci = open(C_path, 'r')
    Vi = open(V_path, 'r')
    header = Ci.readline()
    header_V = Vi.readline()
    assert header == header_V
    W, d = map(int, header.split())
    C = np.ones(shape=(W, d+1))
    V = np.ones(shape=(W, d+1))
    for (i, line) in enumerate(Ci):
        sl_C = line.split()
        Cword = sl_C[0]
        Cvec = map(float, sl_C[1:])
        C[i, :-1] = Cvec
        sl_V = Vi.readline().split()
        Vword = sl_V[0]
        assert Cword == Vword
        Vvec = map(float, sl_V[1:])
        V[i, :-1] = Vvec
else:
    sys.exit('are you kidding me? give me two nps OR two txts')

# --- load labelled (training) data --- #
# this will be triples of S R T... if we're doing 1 v all then R is 1 or 0
print 'Reading in labelled data from\n\t', labelled_examples_path
if '.npy' in labelled_examples_path:
    labelled_examples = np.load(labelled_examples_path)
else:
    labelled_examples = []
    for line in open(labelled_examples_path,'r'):
        sl = map(int, line.split())
        if len(sl) == 3:
            S, R, T = sl
            labelled_examples.append([S, R, T])
        elif len(sl) == 4:
            if sl[-1] == 1:
                S, R, T = sl[:-1]
                labelled_examples.append([S, R, T])
        else:
            print 'wat'
            continue
    labelled_examples = np.array(labelled_examples)

# init a little
Y = labelled_examples[:, 1]
n_class = max(Y)+1

# --- preprocess --- #
# trim if necessary
if np.mean(C[:,-1]) == 1:
    C = C[:, :-1]
    V = V[:, :-1]
# process data
if CONCAT:
    X = np.empty(shape=(Y.shape[0], C.shape[1]+V.shape[1]))
    for (i, (S, R, T)) in enumerate(labelled_examples):
        X[i, :C.shape[1]] = C[S, :]
        X[i, C.shape[1]:] = V[T, :]
        assert Y[i] == R
elif ADD:
    assert C.shape[1] == V.shape[1]
    X = np.empty(shape=(Y.shape[0], C.shape[1]))
    for (i, (S, R, T)) in enumerate(labelled_examples):
        X[i, :] = C[S, :] + V[T, :]
        assert Y[i] == R
elif MEAN:
    assert C.shape[1] == V.shape[1]
    X = np.empty(shape=(Y.shape[0], C.shape[1]))
    for (i, (S, R, T)) in enumerate(labelled_examples):
        X[i, :] = 0.5*(C[S, :] + V[T, :])
        assert Y[i] == R
else:
    raise NotImplementedError('Please specify one of CONCAT, ADD, MEAN for combining features.')

# --- test/train split, with evaluation --- #
print 'Running crossvalidation... (K='+str(K)+')'
M = np.zeros(shape=(K, n_class, n_class))
skf = StratifiedKFold(Y, K)
for (k, (train, test)) in enumerate(skf):
    X_train = X[train, :]
    Y_train = Y[train]
    X_test = X[test, :]
    Y_test = Y[test]
    clf = clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    for (predicted, truth) in zip(predictions, Y_test):
        M[k, truth, predicted] += 1

# combine information across folds
co = open('l10_predz.txt', 'a')
precision = np.zeros(shape=(K, n_class))
recall = np.zeros(shape=(K, n_class))
for k in xrange(K):
    for c in xrange(n_class):
        precision[k, c] = np.float(M[k, c, c])/np.sum(M[k, :, c])
        recall[k, c] = np.float(M[k, c, c])/np.sum(M[k, c, :])
for c in xrange(n_class):
    print c, '('+str(np.bincount(Y)[c])+')'
    if np.sum(M[:, :, c]) == 0:
        prec = 'NA'
        print '\tprecision:\tundefined (class never predicted)\t~('+str(int(np.sum(M[:, c, c])))+' / '+str(int(np.sum(M[:, :, c])))+')'
    else:
        prec = np.nanmean(precision[:, c])
        print '\tprecision:\t', np.nanmean(precision[:, c]), '\t~('+str(int(np.sum(M[:, c, c])))+' / '+str(int(np.sum(M[:, :, c])))+')'
    if np.sum(M[:, c, :]) == 0:
        rec  = 'NA'
        print '\trecall:\t\tundefined (class never observed)\t~('+str(int(np.sum(M[:, c, c])))+' / '+str(int(np.sum(M[:, c, :])))+')'
    else:
        rec = np.nanmean(recall[:, c])
        print '\trecall:\t\t', np.nanmean(recall[:, c]), '\t~('+str(int(np.sum(M[:, c, c])))+' / '+str(int(np.sum(M[:, c, :])))+')'
    if c == miss_class:
        co.write(params_path+' '+str(c)+' '+str(prec)+' '+str(rec)+'\n')
co.close()
