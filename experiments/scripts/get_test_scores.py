#!/bin/python
# get scores on test set, after loading thresholds from previous dev set

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
    #yo
    params_path = sys.argv[1]
    assert '.txt' in params_path
except IndexError:
    sys.exit('must provide path to vectors')

wordlist_path = '../data/wordlist_simple.txt'
relalist_path = '../data/relalist_simple.txt'

# get test data (note: operating on strings) (so not true)
testpath = '../data/test_triples_simple.txt'
test = np.array(map(lambda x: map(int, x.strip('\n').split(' ')), open(testpath,'r').readlines()))
print 'Dev data read from', testpath

# get parameters
pp = params(params_path)
print 'Parameters loaded from', params_path
if pp.relas == map(str, xrange(pp.R)):
    relamap = dict()
    for line in open(relalist_path, 'r'):
        index = int(line.split()[0])
        rela = line.split()[1]
        relamap[rela] = index
        pp.relas[index] = rela

# run through test data
n_test = len(test)/2
test_scores = dict((rela, []) for rela in pp.relas)
test_labels = dict((rela, []) for rela in pp.relas)

word_dict = dict()
rela_dict = dict()
for line in open(wordlist_path):
    index = int(line.split()[0])
    word = line.split()[1]
    word_dict[word] = index
    assert pp.words[index] == word
for line in open(relalist_path):
    index = int(line.split()[0])
    rela = line.split()[1]
    rela_dict[rela] = index
    assert pp.relas[index] == rela

for i in xrange(0, len(test)-1, 2):
    s_true = test[i, 0]
    r_true = test[i, 1]
    t_true = test[i, 2]
    label = test[i, 3]
    assert label == 1
    s_false = test[i+1, 0]
    r_false = test[i+1, 1]
    t_false = test[i+1, 2]
    label = test[i+1, 3]
    assert label == -1
    assert r_true == r_false
    r = r_true
    del(r_true)
    del(r_false)
    energy_true = pp.E_triple((s_true, r, t_true))
    # i know this because it is true for the dev sets
    switch = 'V'
    energy_false = pp.E_triple((s_false, r, t_false))
    rela_specific_array = test_scores[pp.relas[r]]
    if COND_PROB:
        denom = np.sum(np.exp(-pp.E_axis((s_true, r, t_true), switch)))
        rela_specific_array.append([np.exp(-energy_true)/denom, np.exp(-energy_false)/denom])
    else:
        rela_specific_array.append([-energy_true, -energy_false])

# --- now load some cutoffs --- #
thresh_path = re.sub('.txt', '_rela_thresholds_triples.txt', params_path)
thresholds = dict()
ti = open(thresh_path,'r')
ti.readline()
for line in ti:
    if 'NA' in line:
        continue
    rela = line.split()[0]
    threshold = float(line.split()[2])
    thresholds[rela] = threshold
print 'Thresholds read from', thresh_path
ti.close()

print '\nResults:'
co_path = re.sub('.txt', '_test_accuracies.txt', params_path)
co = open(co_path,'w')
co.write('rela n accuracy\n')
overall = 0
nn = 0
for rela in test_scores:
    scores = -np.array(test_scores[rela])
    if len(scores) > 0:
        thresh = thresholds[rela]
        accuracy = get_accuracy(scores, thresh)
    else:
        accuracy = 0
    n = scores.shape[0]
    overall += accuracy*n
    nn += n
    print '%.2f' % accuracy, re.sub('0','_',str(n).zfill(4)), '%.3f' % thresholds[rela], str(rela)
    co.write(str(rela)+' '+str(n)+' '+str(accuracy)+'\n')

print '\nOverall accuracy (weighted mean):', overall/nn
co.write('all '+str(nn)+' '+str(overall/nn)+'\n')
co.close()
