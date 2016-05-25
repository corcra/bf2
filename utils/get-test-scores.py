#!/bin/python
# NOTE: this REQUIRES a valid cfg file.
# helper script to get accuracy on a test set (THAT'S IT!) 

import socher_task_fns as socher_task
import bf2f as bf2f
import cProfile
import re
import pdb as pdb
from subprocess import call
import sys

# use simplified (i.e. aligned with wiki) vocab?
simple_socher = False
# save?
save = True

# --- load options --- #
options = bf2f.options_dict()
try:
    options.load(sys.argv[1], verbose=True)
except IndexError:
    sys.exit('ERROR: requires cfg file.')

try:
    epoch = sys.argv[2]
except IndexError:
    sys.exit('ERROR: requires epoch')

# --- unroll options --- #
output_root = options['output_root']
wordlist = options['wordlist']
relalist = options['relalist']

# --- load parameters --- #
params_path = output_root+'_epoch'+epoch+'_XXX.txt'
pp = bf2f.params(params_path, options, (wordlist, relalist))
print 'loaded params from', params_path

# --- save results? --- #
if save:
    # note: appending
    socher_auc_test_path = output_root+'_auc_test.txt'
    socher_auc_test_file = open(socher_auc_test_path, 'a')
    socher_acc_test_path = output_root+'_acc_test.txt'
    socher_acc_test_file = open(socher_acc_test_path, 'a')

# --- load the evaluation data  --- #
if simple_socher:
    droot = '/cbio/grlab/home/hyland/data/nips13-dataset/Wordnet/simplified_vocab/'
    dev_path = droot + 'dev_triples_simple.txt'
    dev = bf2f.np.array(map(lambda x: map(int, x.strip('\n').split(' ')), open(dev_path,'r').readlines()))
    print 'Dev data read from', dev_path
    test_path = droot + 'test_triples_simple.txt'
    test = bf2f.np.array(map(lambda x: map(int, x.strip('\n').split(' ')), open(test_path,'r').readlines()))
    print 'test data read from', test_path
else:
    droot = '/cbio/grlab/home/hyland/data/bf2-output/nips2013/wordnet/socher_task_full/data/'
    dev_path = droot + 'dev_triples.txt'
    dev = bf2f.np.array(map(lambda x: map(int, x.strip('\n').split(' ')), open(dev_path,'r').readlines()[1:]))
    print 'Dev data read from', dev_path
    test_path = droot + 'test_triples.txt'
    test = bf2f.np.array(map(lambda x: map(int, x.strip('\n').split(' ')), open(test_path,'r').readlines()[1:]))
    print 'test data read from', test_path

# --- scoretype: energy --- #
# get dev and test scores
dev_scores_energy = socher_task.get_scores(pp, dev, 'energy')
test_scores_energy = socher_task.get_scores(pp, test, 'energy')
# get dev thresholds
thresholds_energy = socher_task.get_thresholds(dev_scores_energy)
# get AUC
auc_energy, fpr_energy, tpr_energy, th_energy = socher_task.get_roc_curve(test_scores_energy)
print 'energy', 'auc', '\t', auc_energy
# get test accuracies
overall, nn = 0, 0
for (rela, (threshold, _, _)) in thresholds_energy.iteritems():
    pos, neg = test_scores_energy[rela]
    n = len(pos) + len(neg)
    if n > 0:
        accuracy = socher_task.get_accuracy(pos, neg, threshold)
        print 'energy', rela, '\t', n, '\t', accuracy
        nn += n
        overall += accuracy*n
weighted_accuracy = overall/nn
print 'energy', 'all', '\t', nn, '\t', weighted_accuracy
if save:
    socher_acc_test_file.write(epoch+' '+str(weighted_accuracy)+ ' energy\n')
    socher_auc_test_file.write(epoch+' '+str(auc_energy)+' energy\n')

# --- scoretype: conditional_G --- #
dev_scores_conditional_G = socher_task.get_scores(pp, dev, 'conditional_G')
test_scores_conditional_G = socher_task.get_scores(pp, test, 'conditional_G')
# get dev thresholds
thresholds_conditional_G = socher_task.get_thresholds(dev_scores_conditional_G)
# get AUC
auc_conditional_G, fpr_conditional_G, tpr_conditional_G, th_conditional_G = socher_task.get_roc_curve(test_scores_conditional_G)
print 'conditional_G', 'auc', '\t', auc_conditional_G
# get test accuracies
overall, nn = 0, 0
for (rela, (threshold, _, _)) in thresholds_conditional_G.iteritems():
    pos, neg = test_scores_energy[rela]
    n = len(pos) + len(neg)
    if n > 0:
        accuracy = socher_task.get_accuracy(pos, neg, threshold)
        print 'conditional_G', rela, '\t', n, '\t', accuracy
        nn += n
        overall += accuracy*n
weighted_accuracy = overall/nn
print 'conditional_G', 'all', '\t', nn, '\t', weighted_accuracy
if save:
    socher_acc_test_file.write(epoch+' '+str(weighted_accuracy)+ ' conditional_G\n')
    socher_auc_test_file.write(epoch+' '+str(auc_conditional_G)+' conditional_G\n')
    socher_acc_test_file.close()
    socher_auc_test_file.close()
