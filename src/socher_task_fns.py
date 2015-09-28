#!/bin/python
# Get accuracy on the wordnet task from Socher et al

from bf2f import params
import re
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import sys

# --- functions --- #
def split_data(data, relas):
    """
    just splits the data into a rela-keyed dict
    (should correspond in size to all the score dicts)
    """
    rela_data = dict((rela, []) for rela in relas)
    for (s, r, t, label) in data:
        rela_data[relas[r]].append([s, r, t, label])
    for (rela, pre_array) in rela_data.iteritems():
        rela_data[rela] = np.array(pre_array)
    return rela_data

def get_accuracy(pos_scores, neg_scores, threshold):
    true_pos = np.sum(pos_scores >= threshold)
    true_neg = np.sum(neg_scores < threshold)
    total = len(pos_scores) + len(neg_scores)
    accuracy = np.float(true_pos + true_neg)/total
    return accuracy

def get_hits_at_N(conditionals, truth, N):
    """
    runs through a ndarry of conditional vectors (of length |R|),
    measures how often the true rela is in the top N highest probabilities
    """
    n = conditionals.shape[0]
    if type(truth) == int:
        truth = [truth]*n
    rela_ranks = np.argsort(conditionals, axis=1)
    top_N = rela_ranks[:, :N]
    hits_at_N = np.sum([truth[i] in top_N[i] for i in xrange(n)])
    return hits_at_N

def get_conditionals(params, data):
    """
    runs through a dataset like:
        s, r, t, label
    for each (s, r, t) returns the conditional p(r| s, t) - vector of length |R|
    and returns a dictionary of r:(pos_cond, neg_cond)
    where pos_cond is an array of vectors for label == 1
    and neg_scores is an array of vectors for label == -1
    """
    rela_specific_conditionals = dict((r, [[],[]]) for r in params.relas)
    for (s, r, t, label) in data:
        unnorm_probs = np.exp(-params.E_axis((s, r, t), 'G'))
        normalisation = np.sum(unnorm_probs)
        conditional = unnorm_probs/normalisation
        if label == 1:
            rela_specific_conditionals[params.relas[r]][0].append(conditional)
        elif label == -1:
            rela_specific_conditionals[params.relas[r]][1].append(conditional)
    # make into ndarrays
    for (r, (pos_cond, neg_cond)) in rela_specific_conditionals.iteritems():
        rela_specific_conditionals[r][0] = np.array(pos_cond)
        rela_specific_conditionals[r][1] = np.array(neg_cond)
    return rela_specific_conditionals

def get_scores(params, data, scoretype='conditional_G'):
    """
    runs through a dataset like:
         s, r, t, label
    getting a score associated with each (s, r, t)
    and returning a dictionary of r:(pos_scores, neg_scores)
    where pos_scores is an array of scores for label == 1
    and neg_scores is an array of scores for label == -1
    """
    rela_specific_scores = dict((r, ([],[])) for r in params.relas)
    for (s, r, t, label) in data:
        energy = params.E_triple((s, r, t))
        if scoretype == 'energy':
            score = -energy
        elif scoretype == 'conditional_G':
            denom = np.sum(np.exp(-params.E_axis((s, r, t), 'G')))
            score = np.exp(-energy)/denom
        elif scoretype == 'conditional_V':
            denom = np.sum(np.exp(-params.E_axis((s, r, t), 'V')))
            score = np.exp(-energy)/denom
        else:
            sys.exit('ERROR: unknown scoretype '+scoretype)
        if label == 1:
            rela_specific_scores[params.relas[r]][0].append(score)
        elif label == -1:
            rela_specific_scores[params.relas[r]][1].append(score)
        else:
            sys.exit('ERROR: cannot parse label '+str(label))
    return rela_specific_scores

def get_thresholds(dev_scores):
    print 'Getting rela-specific score cutoffs from dev set.'
    thresholds = dict((r, (0, 0)) for r in dev_scores.keys())
    for (rela, score_array) in dev_scores.iteritems():
        accs = []
        pos_scores = np.array(score_array[0])
        neg_scores = np.array(score_array[1])
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            minscore = min(np.min(pos_scores), np.min(neg_scores))
            maxscore = max(np.max(pos_scores), np.max(neg_scores))
            for threshold in np.linspace(minscore, maxscore, 100):
                accuracy = get_accuracy(pos_scores, neg_scores, threshold)
                accs.append([threshold, accuracy])
            accs = np.array(accs)
            max_accuracy = np.max(accs[:, 1])
            max_index = np.argmax(accs[:, 1])
            threshold = accs[max_index, 0]
            thresholds[rela] = (threshold, max_accuracy,
                                len(pos_scores) + len(neg_scores))
        else:
            thresholds[rela] = (0, 0, 0)
    return thresholds

def get_roc_curve(dev_scores):
    print 'Getting ROC curve.'
    y_score = []
    y_true = []
    for (rela, score_array) in dev_scores.iteritems():
        pos_scores = score_array[0]
        neg_scores = score_array[1]
        y_score.extend(pos_scores)
        y_true.extend([1]*len(pos_scores))
        y_score.extend(neg_scores)
        y_true.extend([0]*len(neg_scores))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return auc, fpr, tpr, thresholds
