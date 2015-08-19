#!/bin/python
# Get score cutoff thresholds based on dev set

from bf2f import params
import re
import numpy as np
from sklearn.metrics import roc_curve
import sys

COND_PROB = False
#COND_PROB = True

print 'cond_prob', str(COND_PROB)

def get_accuracy(scores, threshold):
    # assume scores has col1 positive and col2 negative labels
    true_pos = np.sum(scores[:, 0] >= threshold)
    true_neg = np.sum(scores[:, 1] < threshold)
    total = 2*scores.shape[0]
    accuracy = np.float(true_pos + true_neg)/total
    return accuracy

# --- wordnet_simple --- $

try:
    params_path = sys.argv[1]
    assert '.txt' in params_path
except IndexError:
    sys.exit('must provide path to vectors')

# get dev data (note: operating on strings) (so not true)
devpath = '../data/dev_triples_simple.txt'
dev = np.array(map(lambda x: map(int, x.strip('\n').split(' ')), open(devpath,'r').readlines()))
print 'Dev data read from', devpath

# get parameters
pp = params(params_path)
print 'Parameters loaded from', params_path
if pp.relas == map(str, xrange(pp.R)):
    relamap = dict()
    for line in open('/cbio/grlab/home/hyland/data/nips13-dataset/Wordnet/simplified_vocab/relalist_simple.txt', 'r'):
        index = int(line.split()[0])
        rela = line.split()[1]
        relamap[rela] = index
        pp.relas[index] = rela

# run through dev data
n_dev = len(dev)/2
dev_scores = dict((rela, []) for rela in pp.relas)
dev_labels = dict((rela, []) for rela in pp.relas)

for i in xrange(0, len(dev)-1, 2):
    s_true = dev[i, 0]
    r_true = dev[i, 1]
    t_true = dev[i, 2]
    label = dev[i, 3]
    assert label == 1
    s_false = dev[i+1, 0]
    r_false = dev[i+1, 1]
    t_false = dev[i+1, 2]
    label = dev[i+1, 3]
    assert label == -1
    assert r_true == r_false
    r = r_true
    del(r_true)
    del(r_false)
    energy_true = pp.E_triple((s_true, r, t_true))
    # i know this because it is true for the dev sets
    switch = 'V'
    energy_false = pp.E_triple((s_false, r, t_false))
    rela_specific_array = dev_scores[pp.relas[r]]
    if COND_PROB:
        denom = np.sum(np.exp(-pp.E_axis((s_true, r, t_true), switch)))
        rela_specific_array.append([np.exp(-energy_true)/denom, np.exp(-energy_false)/denom])
    else:
        rela_specific_array.append([-energy_true, -energy_false])

# --- now get some cutoffs --- #
thresholds = dict((r, (0, 0)) for r in dev_scores.keys())
for (rela, specific_array) in dev_scores.iteritems():
    n = 100
    accs = []
    scores = np.array(specific_array)
    if len(scores) > 0:
        for threshold in np.linspace(np.min(scores), np.max(scores), n):
            accuracy = get_accuracy(scores, threshold)
            accs.append([threshold, accuracy])
        accs = np.array(accs)
        max_accuracy = np.max(accs[:, 1])
        max_index = np.argmax(accs[:, 1])
        threshold = accs[max_index, 0]
        thresholds[rela] = (threshold, max_accuracy, len(scores))
    else:
        thresholds[rela] = (0, 0, 0)

print '\nResults:'
co_path = re.sub('.txt', '_rela_thresholds_triples.txt', params_path)
co = open(co_path,'w')
co.write('rela n threshold accuracy\n')
weighted_acc = 0
nuh = 0
for (rela, (threshold, accuracy, nn)) in thresholds.iteritems():
    print '%.2f' % accuracy, str(nn).zfill(4), '%.3f' %threshold,  rela
    co.write(str(rela)+' '+str(nn)+' '+str(threshold)+' '+str(accuracy)+'\n')
    weighted_acc += accuracy*nn
    nuh += nn
print '\nWeighted accuracy:', weighted_acc/nuh
co.write('all '+str(nuh)+' NA '+str(weighted_acc/nuh)+'\n')
co.close()
