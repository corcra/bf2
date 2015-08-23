#!/bin/python
# author:       Stephanie Hyland (sh985@cornell.edu)
# date:         Jan, July 2014
# description:  Analysis of word vector representations which
#               relies on pairwise distances.
#               1. Get closest word(s) to a query.
#               2. Perform clustering,
#                   2.1 Hierarchical
#                   2.2 kmeans
#                   2.3 DBSCAN

import numpy as np
print 'Numpy version:', np.version.version
import pandas as pd
import scipy.cluster as spc
import scipy.spatial as sps
import random
import itertools
import sys
import fastcluster
import re
import time
import cPickle
from copy import copy, deepcopy
from sklearn.cluster import dbscan
from scipy.cluster.vq import kmeans2
from multiprocessing import Pool
from functools import partial

#  --- hyperparams --- #
VERBOSE = True

# --- i/o functions --- #
def get_words_vecs(data_path, norm=False):
    """
    data_path should be a path to a file formatted as follows:
    if .txt:
        first line is W d (vocab size, dimension)
        every other line is: word (string), vector (floats)
    if .npy:
        dictionary of word:vector 
    Note that in both cases, the redundant 1 is expected to be ommitted.
    returns:
        words (array of word strings)
        vectors (ndarray of W x (d+1))
        w2v (dict of word:vec)
    """
    if VERBOSE:
        print 'Splitting up words and vectors from', data_path
    excluded = []
    words = []
    vecs = []
    word2vec = dict()
    try:
        data = np.load(data_path)
        if VERBOSE:
            print '.npy detected, assuming data is a dictionary!'
        data_stream = data.item()
        word2vec = data_stream
        for word in data_stream:
            thisvec = np.array(data_stream[word], dtype=np.float32)
            if np.isfinite(np.sum(thisvec)):
                words.append(word)
                vecs.append(thisvec)
            else:
                excluded.append(word)
        vecs = np.array(vecs)
    except IOError:
        if VERBOSE:
            print 'No .npy detected, assuming data in plaintext!'
        datafile = open(data_path, 'rU')
        line1 = datafile.readline()
        nrow = int(line1.split()[0])
        d = int(line1.split()[1])
        # note, padding with ones occurs here
        vecs_temp = np.ones((nrow, d+1))
        l = 0
        for line in datafile:
            word = re.split('\t| ', line)[0].strip('\x00')
            thisvec = np.array(re.split('\t| ', line.strip())[1:],
                               dtype=np.float32)
            if np.isfinite(np.sum(thisvec)):
                words.append(word)
                vecs_temp[l,:-1] = thisvec
                word2vec[word] = thisvec
                l += 1
            else:
                excluded.append(word)
        datafile.close()
        vecs = vecs_temp[:l, :]
    if len(excluded) > 0:
        print 'Words excluded for containing NANs:', '\n'.join(excluded)
    if norm:
        vecs[:,:-1] = vecs[:,:-1]/np.linalg.norm(vecs[:,:-1],axis=1).reshape(len(vecs),1)
    return words, vecs, word2vec

def get_matrices(data_path):
    if VERBOSE:
        print 'Obtaining relationship matrices from', data_path
    # going to assume it's all just plaintext for now
    datafile = open(data_path, 'rU')
    line1 = datafile.readline()
    R = int(line1.split()[0])
    d = int(line1.split()[1])
    mats = np.zeros((R, d+1, d+1),dtype=np.float)
    rela2mat = dict()
    relas = []
    m = 0
    for line in datafile:
        sl = line.split()
        rela = sl[0]
        relas.append(rela)
        mat_vec = np.array(sl[1:],dtype=np.float)
        if len(mat_vec) == d*(d + 1):
            matrix = mat_vec.reshape(d, d+1)
            mats[m, :-1, :] = matrix
            mats[m, -1, -1] = 1
        elif len(mat_vec) == (d + 1)*(d + 1):
            matrix = mat_vec.reshape(d+1, d+1)
            mats[m, :, :] = matrix
        else:
            sys.exit('ERROR: Could not parse matrix.')
        rela2mat[rela] = mats[m, :, :]
        m += 1
    return relas, mats, rela2mat

def w2v_to_bf2(model_path, normalise=False):
    """
    Converts a .model object (word2vec object) into the .txt format I use.
    normalise transforms the w2v vectors to unit norm
    """
    assert '.model' in model_path
    from gensim.models.word2vec import Word2Vec
    model = Word2Vec.load(model_path)
    vocab = model.vocab
    C = model.syn1
    V = model.syn0
    d = C.shape[1]
    G = np.eye(d+1)
    # paths, files
    if normalise:
        C_out_path = re.sub('.model', '_normC.txt', model_path)
        G_out_path = re.sub('.model', '_normG.txt', model_path)
        V_out_path = re.sub('.model', '_normV.txt', model_path)
    else:
        C_out_path = re.sub('.model', '_C.txt', model_path)
        G_out_path = re.sub('.model', '_G.txt', model_path)
        V_out_path = re.sub('.model', '_V.txt', model_path)
    C_out = open(C_out_path, 'w')
    G_out = open(G_out_path, 'w')
    V_out = open(V_out_path, 'w')
    W = len(vocab)
    header = str(W) + ' ' + str(d) +'\n'
    C_out.write(header)
    G_out.write(header)
    V_out.write(header)
    # run through the vocabulary
    for (word, vocab_object) in vocab.iteritems():
        i = vocab_object.index
        C_vec = C[i, :]
        V_vec = V[i, :]
        if normalise:
            C_vec /= np.linalg.norm(C_vec)
            V_vec /= np.linalg.norm(V_vec)
        C_out.write(word+' '+' '.join(map(str, C_vec))+'\n')
        V_out.write(word+' '+' '.join(map(str, V_vec))+'\n')
    C_out.close()
    V_out.close()
    # trivial G
    G_out.write('w2v '+' '.join(map(str, G[:-1, :].reshape((d+1)*d)))+'\n')
    G_out.close()
    print 'Wrote', W, 'words to', C_out_path, 'and', V_out_path
    return True

# --- ??? functions --- #
def get_wordinfo(word_marker, words, vecs_all):
    """
    Given word_marker, figure out what it is (index, word, vector)
    and return the missing ones!
    """
    if type(word_marker) == int:
        _ = int(word_marker)
        # success? word_marker is an int!
        word_index = word_marker
        try:
            word = words[word_index]
            word_vec = vecs_all[word_index]
            word_vec = word_vec.reshape(1,len(word_vec))
        except IndexError:
            print 'No word corresponds to index', word_index
            word = None
            word_vec = None
    elif type(word_marker) == str:
        # word_marker is a string!
        try:
            word_index = words.index(word_marker)
            word = word_marker
            word_vec = vecs_all[word_index,:]
            word_vec = word_vec.reshape(1,len(word_vec))
        except ValueError:
            print 'Word', word_marker, 'not found!'
            word_index = None
            word = word_marker
            word_vec = None
    elif type(word_marker) == np.array() or type(word_marker) == list:
        # word_marker is a vector!
        word_index = 'UNKNOWN'
        word = 'UNKNOWN'
        word_vec = word_marker[:]
        word_vec = word_vec.reshape(1,len(word_vec))
    else:
        print 'ERROR: what on earth sort of word marker did you give me?'
        word_index, word, word_vec = None, None, None
    return word_index, word, word_vec

def square_to_condensed(i, j, n):
    """
    This function produces the index in a condensed distance matrix.
    """
    if i < j:
        lower, higher = i, j
    else: lower, higher = j, i
    index = n*lower - lower*(lower+1)*0.5 - lower - 1 + higher
    return int(index)

def word_dot(vecs_all, words, A, B):
    return np.dot(vecs_all[words.index(A)], vecs_all[words.index(B)])

def word_cos(vecs_all, words, A, B):
    """
    Cosine similarity between A and B.
    """
    cos_sim = sps.distance.cosine(vecs_all[words.index(A)],
                                  vecs_all[words.index(B)])
    return 1-cos_sim

# --- tree functions --- #
def has_child(start, node, n):
    """
    Check if a node has a child exactly n levels below it.
    """
    current = set([start])
    for i in range(n):
        next_right = [cur.get_right() for cur in current if not cur.is_leaf()]
        next_left = [cur.get_left() for cur in current if not cur.is_leaf()]
        current = set(next_right+next_left)
        print current
    if node in current:
        return True
    else:
        return False

# given a location in the tree, go up n nodes
def climb_tree_BreadthFirst(tree, node, n):
    queue = []
    queue.append(tree)
    levels = copy(dict((i, v) for (i, v) in zip(range(n), [None]*n)))
    while queue:
        #FIFO
        top = queue.pop(0)
        if has_child(top, node, n):
            # top is a parent at depth n of node!
            return top
        else:
            right = top.get_right()
            left = top.get_left()
            if not right.is_leaf():
                queue.append(right)
            if not left.is_leaf():
                queue.append(left)
    return 0

# given a possible parent and one child, return the other child (if parent)
def other_sibling(node, child1):
    if node.get_right() == child1:
        return node.get_left()
    elif node.get_left() == child1:
        return node.get_right()
    else:
        return 0

# recurse through the tree, retrieve the sibling of a node
def get_sibling_IterativePreorder(tree, node):
    stack = []
    top = tree
    while top:
        sibling = other_sibling(top, node)
        if sibling:
            #return top,sibling
            return sibling
        else:
            right = top.get_right()
            left = top.get_left()
            if not right.is_leaf():
                stack.append(right)
            if not left.is_leaf():
                stack.append(left)
            top = stack.pop()
    return 0

def get_leaves(tree, words):
    count = tree.get_count()
    goon = 'y'
    if count > 10:
        print 'There are', count, 'leaves in this tree!'
        goon = raw_input('continue? ')
    if goon == 'y':
        return tree.pre_order(lambda x: words[x.id])
    else: return None

# --- exploration functions --- #
def get_veclengths(words, C):
    """
    Map the norms of the words onto the words.
    """
    w2l = dict()
    l_list = []
    for (i, w) in enumerate(words):
        l = np.dot(C[i, :], C[i, :])
        w2l[w] = l
        l_list.append(l)
    return w2l, l_list

def get_closest(dist, topn, words, C, G, V, word_marker):
    """
    This will get the closest V to a C, e.g.
    word_marker should refer to a source
    and we will return likely targets,
    all given a relationship, via G.
    If no G, assume identity.
    Note that this may not be meaningful.
    returns topn_words
    """
    # transform C by G, then compare distances with V...
    try:
        GC = np.dot(G,C.T).T
    except TypeError:
        # G is None
        GC = C
    word_index, word, word_vec = get_wordinfo(word_marker, words, GC)
    if None in (word_index, word, word_vec):
        # failed at getting some information, abandon hope
        return None
    if VERBOSE:
        print ''
    # if a distance vector is supplied, use it...
    # please be careful that the distance vector is appropriate
    # (e.g. it must have been calculated wrt correct matrix)
    # otherwise just get distance between target word and all others!
    try:
        indices = range(len(words))
        indices.remove(word_index)
        d = np.array([dist[square_to_condensed(word_index, j, len(words))]
                      for j in indices])
    except (TypeError, ValueError) as e:
        # distance matrix does not exist
        #if VERBOSE:
        #    print 'No distance matrix detected, or unconventional word index provided!'
        d = sps.distance.cdist(word_vec,V,'cosine')
    # np 1.8 has argpartition, nicer!
    if '1.7' in np.version.version:
        if VERBOSE:
            print 'Using numpy version', np.version.version, '- avoiding argpartition!'
        if VERBOSE:
            print topn, 'closest word (indices) to', word, '(', word_index, ')'
        ordered = np.sort(d)[:topn+1]
        topn_words_indices = [np.where(d == val)[0][0] for val in ordered]
        topn_words = [words[np.where(d == val)[0][0]] for val in ordered]
    else:
        #print 'Using argpartition because new numpy!'
        topn_words_indices = np.argpartition(d, tuple(range(topn+1)))[0][:(topn+1)]
        topn_words = [words[index] for index in topn_words_indices]
    if VERBOSE:
        print ''
    # report the results
    if VERBOSE:
        print 'Top '+str(topn)+' closest results to '+word+':\n'
    for i in xrange(topn+1):
        if VERBOSE:
            print topn_words_indices[i], "\t", "%.5f" % d[:, topn_words_indices[i]][0], " ", topn_words[i]
        if i == 0 and VERBOSE:
            print ''
    print ''
    return True
    #return topn_words

def cut_dendrogram(Z, words, topy, t, criter='maxclust'):
    if VERBOSE:
        print 'Getting clusters from the dendrogram!'
        print 'Criterion:', criter, '\tparam :', t
    # note choice of inconsistent
    clusters = spc.hierarchy.fcluster(Z, t, criter)
    assignments = pd.DataFrame({'word':words, 'cluster':clusters})
    csizes, indices = eval_assignments(assignments, t, None)
    return assignments, csizes, indices

def q_cluster(assignments, n):
    print assignments[assignments['cluster'] == n]['word']
    return assignments[assignments['cluster'] == n]['word']

# --- evaluation functions --- #

def analogy_task(truthset_path, d, words, vecs):
    """
    This expects data in the form of either:
    1. mikolov semantic-syntactic test set
    2. a pickled dictionary of 'test type' and quadruples (e.g. derived from 1.)
    """
    if '.pk' in truthset_path:
        task_subtypes = cPickle.load(open(truthset_path, 'rb'))
    else:
        infile = open(truthset_path, 'r')
        task_subtypes = dict()
        for line in infile:
            if '//' in line:
                continue
            elif ':' in line:
                # new subtype defined!
                subtype = line.split()[1]
                task_subtypes[subtype] = []
            else:
                if subtype in task_subtypes:
                    tuple = line.lower().split()
                    # get their indices
                    try:
                        tuple_enc = [words.index(x) for x in tuple]
                        task_subtypes[subtype].append(tuple_enc)
                    except ValueError:
                        missing = []
                        for x in tuple:
                            if not x in words:
                                missing.append(x)
                        print 'Can\'t encode', line, '-missing', ', '.join(missing)
                else:
                    print 'Something went wrong!' # this is almost certainly due to file formatting
                    print subtype, line
                    return None
        cPickle.dump(task_subtypes, open(truthset_path+'.pk', 'wb'))
    print 'Semantic-syntactic tasks loaded!'
    scores = dict((k, 0) for k in task_subtypes.keys())
    for subtype in task_subtypes:
        task_score = 0
        # set of all the quadruples
        for task in task_subtypes[subtype]:
            A, B, C, D = task
            print 'Testing::', ' '.join(map(lambda x: words[x], task))
            # A is to B as C is to ...?
            propD_vec = vecs[B]-vecs[A]+vecs[C]
            closest = get_closest(None, 1, words, vecs, None, vecs, propD_vec)
            print '\t', ', '.join(closest)
            if D in closest[:10]:
                print 'Success!!'
                task_score += 1
        scores[subtype] = task_score
    print 'Scores acquired!'
    return scores

def get_spread(clusterX):
    cluster_mean = np.mean(clusterX, 0)
    distances = np.sum(np.abs(clusterX-cluster_mean)**2, axis=1)**(0.5)
    return np.std(distances)

def eval_assignments(assignments, nclust, topy=10):
    if VERBOSE:
        print 'There are', nclust, 'clusters!'
    csizes = np.array([len(assignments[assignments['cluster'] == i]) for i in xrange(nclust)])
    nsingleton = sum(csizes == 1)
    uniq_counts = list(set(csizes))
    topY = sorted(uniq_counts, reverse=True)[0:topy]
    indices = [np.where(csizes == i) for i in topY]
    if VERBOSE:
        print 'There are', sum(csizes == 1), 'clusters with one entity.'
        print 'Top clusters:'
        for i in range(len(topY)):
            print 'size:\t', topY[i], '\t#: ', len(indices[i][0]), '\twhich:\t', ', '.join(map(str,indices[i][0]))
    return csizes, indices

# --- clustering functions --- #
def kmeans_clust(vecs, words, K):
    if VERBOSE:
        print 'Running kmeans!'
    if np.mean(vecs[:,-1] == 1) == 1:
        # exclude the column of 1s
        vex = deepcopy(vecs[:, :-1])
    else:
        vex = deepcopy(vecs)
    # normalise (kmeans does euclidean distance, so this is required for cosine')
    vex /= np.linalg.norm(vex, axis=1).reshape(-1, 1)
    centroids, cluster_assignments = kmeans2(vex, K)
    assignments = pd.DataFrame({'word':words, 'cluster':cluster_assignments})
    csizes, indices = eval_assignments(assignments, K, None)
    return assignments, csizes, indices

def DBSCAN_clust(d, words, epsilon):
    if VERBOSE:
        print 'Running DBSCAN!'
    core, labels = dbscan(d, eps=epsilon, metric='precomputed')
    cluster_assignments = labels
    nclust = max(cluster_assignments)
    assignments = pd.DataFrame({'word':words, 'cluster':labels})
    csizes, indices = eval_assignments(assignments, nclust, None)
    return assignments, csizes, indices

def get_cluster_representatives(assignments, word2vec):
    """
    For each cluster, find the 'representative' word.
    Right now, this will be the word closest to the centre of the cluster. (by mean)
    """
    cluster_reps = dict()
    clusterlist = assignments['cluster']
    clusters = set(clusterlist)
    for cluster in clusters:
        words = list(assignments[assignments['cluster'] == cluster]['word'])
        vecs = [word2vec[word] for word in words]
        cluster_centre = np.mean(vecs, axis=0)
        distances = [sps.distance.cosine(cluster_centre, vec) for vec in vecs]
        mindist_index = np.argmin(distances)
        representative = words[mindist_index]
        cluster_reps[cluster] = representative
    reps = []
    for c in clusterlist:
        reps.append(cluster_reps[c])
    assignments['representative'] = reps
    return cluster_reps, assignments

def hierarchical_clust(d, cluster_method='average'):
    if VERBOSE:
        print 'Doing hierarchical clustering using fastcluster!'
    # some might say this function is redundant
    # d should be a distance vector
    Z = fastcluster.linkage(d, method=cluster_method)
    return Z

# --- distance functions --- #
def get_distances(C, V, distance_metric, max_nvecs=50000, matrix=None):
    """
    Get pairwise distances
    *** given a transformation matrix ***
    This is more for generating a big set (matrix) of pairwise distances.
    """
    nvecs = C.shape[0]
    d = C.shape[1]
    if not C.shape == V.shape:
        print 'ERROR: word vectors are not aligned.'
        return None
    if nvecs > max_nvecs:
        print 'We\'re not going to have enough memory for this. :( Subsetting!'
        subset = random.sample(xrange(nvecs), max_nvecs)
        C = C[subset]
        V = V[subset]
    else:
        subset = range(nvecs)
    if VERBOSE:
        print 'Getting distances!'
    try:
        if not matrix.shape[0] == d:
            print 'ERROR: Matrix is not aligned to word vector dimension.'
            return None
        else:
            # transform vectors wrt matrix (relationship)
            GC = np.dot(matrix,C.T).T
    except AttributeError:
        # NoneType has no attribute shape
        print 'No relationship matrix included, using identity.'
        # 'identity'
        GC = C
    return subset, sps.distance.cdist(V, GC, distance_metric)

# --- things pertaining to pairwise sentence distances --- #
def speed_sentence_distance(i1, i2):
    """
    Requires precalculated distance matrix.
    (the shared variable also needs to be set)
    (may have some memory issues here)
    """
    coords = map(list,itertools.product(i1,i2))
    dist = calculate_sentence_distance(coords)
    return dist

def sentence_distance(d, words, vecs, s1, s2, sigmasq=1.0):
    """
    Calculate the distance between two sentences (s1, s2)
    Assumes that the sentences are lists of EITHER strings, or integers.
    If strings, these are actual words and need to be mapped into integers. (using words)
    Integers can be used to query the pre-calculated distance matrix. (d)
    If no distance matrix, we calculate it on the fly.
    Note: the sentences do not need to be the same length... or even the same type !
    Note: using COSINE DISTANCE unless otherwise specified...
    """
    pairs = itertools.product(s1,s2)
    numpairs = 0
    distance = 0
    # this for loop is annoyingly long
    for pair in pairs:
        numpairs += 1
        # get the distance
        p1, p2 = pair
        if type(p1) == int:
            i1 = p1
        elif type(p1) == str or type(p1) == unicode:
            try:
                # this is a slow step
                i1 = words.index(p1)
            except ValueError:
                # don't know this word
                pair_dist = 0
                distance += np.exp(-(pair_dist*pair_dist)/sigmasq)
                continue
        else:
            print 'WTF???'
        if type(p2) == int:
            i2 = p2
        elif type(p2) == str or type(p2) == unicode:
            try:
                i2 = words.index(p2)
            except ValueError:
                # don't know this word
                pair_dist = 0
                distance += np.exp(-(pair_dist*pair_dist)/sigmasq)
                continue
        if not d == None:
            # already have pairwise distances
            if len(d.shape) == 2:
                # d is a matrix
                # NOTE d MUST CORRESPOND TO INDEXING OF THE WORDS
                pair_dist = d[i1,i2]
            elif len(d.shape) == 1:
                # d is a distance vector (:()
                n = d.shape[0]
                pair_dist = d[square_to_condensed(i1,i2,n)]
        else:
            # no pairwise distances
            # these vectors should exist becuse we always checked for invalid indices
            v1 = vecs[i1, :]
            v2 = vecs[i2, :]
            # OBSERVE CHOICE OF METRIC
            # this is a slow step
            pair_dist = sps.distance.cosine(v1,v2)
        distance += np.exp(-(pair_dist*pair_dist)/sigmasq)
    return distance/numpairs

# --- unsorted functions --- #
def order_sentences(head, neighbours, d, words, vecs, p):
    """
    This will take a list of neighbours to a sentence and order them
    according to sentence similarity.
    THIS NEEDS TO BE FAST, BUT IT ISN'T REALLY.
    p is a pool of processes...
    """
    partial_sentences = partial(sentence_distance, d, words, vecs, head)
    distances = p.map(partial_sentences, neighbours)
    ranking = np.argsort(distances)
    ordered_neighbours = [neighbours[r] for r in ranking]
    return ordered_neighbours

def do_cluster_for_a_t(Z, words, word2vec, t, criter, assignment_path):
    print 'Cutting dendrogram into ', t, 'clusters using', criter
    asi, cszies, topy = cut_dendrogram(Z, words, 20, criter, t)
    cluster_reps, asi_withreps = get_cluster_representatives(asi, word2vec)
    print 'Saving assignments into', assignment_path
    asi_withreps.to_pickle(assignment_path)
    return True

# --- ~~~ --- #
print 'explore_fns.py: Functions loaded!'
print 'HINT:'
print 'get_closest(dist, topn, words, C, G, V, word_marker)'
print 'get_distances(C, V, distance_metric, max_nvecs=50000, matrix=None)'
print 'hierarchical_clust(d, cluster_method=\'average\')'
print 'cut_dendrogram(Z, words, topy, t, criter=\'maxclust\')'
