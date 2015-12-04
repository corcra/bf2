#!/bin/python
# Get rank and posterior probability (and means) for test/vali set etc.

from scipy.stats import rankdata
import sys
import numpy as np

def load_tasks(path, ENCODED=False):
    """
    From a single dev/test etc. file, load data to evaluate on.
    Returns a list, e.g. for T task:
        of [(S, R, _), [T1, T2, T3, T4]] etc. (Ti are the truths)
    So it returns three lists like this.
    
    ENCODED controls the format of the input; is it already in triples, or is
    in 'plaintext'?
    """
    S_task = []
    R_task = []
    T_task = []
    fi = open(path, 'r')
    header = fi.readline()
    for line in fi:
        which, s, r, t = line.strip('\n').split('\t')[:4]
        if which == 'S':
            if ENCODED:
                triple = (-1, int(r), int(t))
                truth = map(int, s.split('|'))
            else:
                triple = ('', r, t)
                truth = s.split('|')
            S_task.append([triple, truth])
        elif which == 'R':
            if ENCODED:
                triple = (int(s), -1, int(t))
                truth = map(int, r.split('|'))
            else:
                triple = (s, '', t)
                truth = r.split('|')
            R_task.append([triple, truth])
        elif which == 'T':
            if ENCODED:
                triple = (int(s), int(r), -1)
                truth = map(int, t.split('|'))
            else:
                triple = (s, r, '')
                truth = t.split('|')
            T_task.append([triple, truth])
        else:
            raise NotImplementedError
    return S_task, R_task, T_task

def convert_to_indices(tokens, relas, task, switch):
    """
    Given a task, and tokens/relas, convert to indices.
    """
    task_indices = []
    for ((s, r, t), truth) in task:
        try:
            if switch == 'S':
                si = -1
                ri = relas.index(r)
                ti = tokens.index(t)
            elif switch == 'R':
                si = tokens.index(s)
                ri = -1
                ti = tokens.index(t)
            elif switch == 'T':
                si = tokens.index(s)
                ri = relas.index(r)
                ti = -1
            else:
                raise NotImplementedError
        except IndexError:
            sys.exit('ERROR: missing one of', s, r, t)
        triplei = (si, ri, ti)
        truthi = []
        for q in truth:
            if switch == 'S':
                qi = tokens.index(q)
            elif switch == 'R':
                qi = relas.index(q)
            elif switch == 'T':
                qi = tokens.index(q)
            truthi.append(qi)
        task_indices.append([triplei, truthi])
    return task_indices

def get_ranks_probs(params, task, switch):
    """
    Given a task (see previous function for format) and some parameters, get
    the ranks and posteriors of *all* the true values.
    (switch tells us which task we're doing, passed to E_axis)

    Note, task is assumed to be encoded in indices.
    """
    ranks_probs = []
    for (triple, truth) in task:
        expmE = np.exp(-params.E_axis(triple, switch))
        probabilities = expmE/np.sum(expmE)
        ranks = []
        probs = []
        for q in truth:
            ranks.append(rankdata(-probabilities)[q]) 
            probs.append(probabilities[q])
        ranks_probs.append([triple, ranks, probs])
    return ranks_probs

def mean_ranks_probs(ranks_probs):
    """
    Get the mean reciprocral rank (MRR) and mean sumprobability etc. across tasks.
    
    NOTE: in cases where there are multiple correct answers, we:
        - sum them (for probabilities)
        - use the highest(/lowest) rank (for ranks)
    """
    mrr = 0
    sumprob = 0
    n = len(ranks_probs)
    if n == 0:
        print 'WARNING: missing data? No ranks or probability data.'
        return 'NA', 'NA'
    else:
        for (triple, ranks, probs) in ranks_probs:
            rank = min(ranks)
            prob = np.sum(probs)     # sum of probs, here
            mrr += 1.0/rank
            sumprob += prob
        mrr /= n
        sumprob /= n
        return mrr, sumprob

def get_mrr_sumprob(params, tasks_or_taskpath, ENCODED=False,
                     mrr_file=None, sumprob_file=None, epoch=None):
    """
    Full pipeline (folds in above functions)
    (has option not to reload the task data, in case that's already been done)
    """
    if type(tasks_or_taskpath) == str:
        # it's a path
        s_task, r_task, t_task = load_tasks(tasks_or_taskpath)
    elif type(tasks_or_taskpath) == tuple:
        # it's a tuple of tasks
        s_task, r_task, t_task = tasks_or_taskpath
    tokens = params.words
    relas = params.relas
    # S #
    if ENCODED:
        si_task = s_task
    else:
        si_task = convert_to_indices(tokens, relas, s_task, 'S')
    s_ranks_probs = get_ranks_probs(params, si_task, 'S')
    s_mrr, s_sumprob = mean_ranks_probs(s_ranks_probs)
    # R #
    if ENCODED:
        ri_task = r_task
    else:
        ri_task = convert_to_indices(tokens, relas, r_task, 'R')
    r_ranks_probs = get_ranks_probs(params, ri_task, 'R')
    r_mrr, r_sumprob = mean_ranks_probs(r_ranks_probs)
    # T #
    if ENCODED:
        ti_task = t_task
    else:
        ti_task = convert_to_indices(tokens, relas, t_task, 'T')
    t_ranks_probs = get_ranks_probs(params, ti_task, 'T')
    t_mrr, t_sumprob = mean_ranks_probs(t_ranks_probs)
    if not mrr_file is None and not sumprob_file is None:
        assert not epoch is None
        # mrrs
        mrr_file.write(str(epoch)+' '+str(s_mrr)+' '+'S\n')
        mrr_file.write(str(epoch)+' '+str(r_mrr)+' '+'R\n')
        mrr_file.write(str(epoch)+' '+str(t_mrr)+' '+'T\n')
        # sumprobs
        sumprob_file.write(str(epoch)+' '+str(s_sumprob)+' '+'S\n')
        sumprob_file.write(str(epoch)+' '+str(r_sumprob)+' '+'R\n')
        sumprob_file.write(str(epoch)+' '+str(t_sumprob)+' '+'T\n')
        mrr_file.flush()
        sumprob_file.flush()
        return True
    else:
        return (s_mrr, r_mrr, t_mrr), (s_sumprob, r_sumprob, t_sumprob)
