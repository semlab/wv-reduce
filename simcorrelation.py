import argparse
import math
import os
import time
import numpy as np
import pandas as pd
import gensim
import matplotlib.pyplot as plt
from scipy import stats
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import datasets as ds
# TODO remove unused libraries



def sim_correlation(wordpairs, arg1, arg2):
    """
    :param ds_wordpairs: a list of tuples representing a dataset word pairs
    :param arg1: can be a list of scores or a gensim.models.keyedvectors.KeyedVectors
    :param arg2: can be a list of scores or a gensim.models.keyedvectors.KeyedVectors
    """
    sim_scores1 = [arg1.similarity(wordpair[0], wordpair[1]) for wordpair in wordpairs] if type(arg1) == gensim.models.keyedvectors.KeyedVectors else arg1
    sim_scores2 = [arg2.similarity(wordpair[0], wordpair[1]) for wordpair in wordpairs] if type(arg2) == gensim.models.keyedvectors.KeyedVectors else arg2
    s = stats.spearmanr(sim_scores1, sim_scores2)
    return s.correlation


def sim_correlations(wordpairs, sim_scores, kvecs_list):
    correlations = {}
    for kvecs in kvecs_list:
        correlation = sim_correlation(wordpairs, sim_scores, kvecs)
        dim = len(kvecs[0])
        correlations[dim] = correlation 
    return correlations



def plot_similarities(correlation_scores, filename='figs/correlations.png'):
    """
    Plot the word similarity correlation scores
    :param correlation_scores: nested dictionary storing correlation scores
        by model, then datasets, then trained ('train') or reduced ('reduced')
    """
    dimensions = list(range(0,550,50))
    n_models = len(correlation_scores)
    datasets = list(correlation_scores.keys())
    n_datasets = len(correlation_scores[datasets[0]]) if n_models > 0 else 0
    plot_styles = [{'marker':'o', 'linestyle':'-', 'color':'darkorange'},
                {'marker':'^', 'linestyle':'--', 'color':'green'}
               ]

    fig, axs = plt.subplots(n_datasets, n_models, figsize=(n_datasets*5,n_models*5))
    for idx_model, model in enumerate(correlation_scores):
        for idx_dataset, dataset in enumerate(correlation_scores[model]):
            for idx_method, method in enumerate(correlation_scores[model][dataset]):
                x = correlation_scores[model][dataset][method].keys()
                y = correlation_scores[model][dataset][method].values()
                #axs[idx_dataset, idx_model].grid(visible=True)#, axis='x')
                axs[idx_dataset, idx_model].plot(x, y, label=f'{dataset} {method}', **plot_styles[idx_method])
                axs[idx_dataset, idx_model].set_xlim(max(dimensions)+50, min(dimensions))
                axs[idx_dataset, idx_model].legend(loc='lower left')
                axs[idx_dataset, idx_model].set_title(f'{dataset} ({model})')
                # TODO grid, share axis? 
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    args = {}
    #load the vectors
    # TODO give the root path as an arg
    args['vectors_root'] = './out'
    print("loading glove vectors...")
    glove_kvecs_store = ds.load_keyedvectors(os.path.join(args['vectors_root'], 'glove'))#, no_header=True)
    print("loading cbow vectors...")
    cbow_kvecs_store = ds.load_keyedvectors(os.path.join(args['vectors_root'], 'cbow'))
    print("loading skipgram vectors...")
    skipgram_kvecs_store = ds.load_keyedvectors(os.path.join(args['vectors_root'], 'skipgram'))
    # vocab
    kvecs = KeyedVectors.load_word2vec_format(datapath(f'/home/gr0259sh/Projects/opensrc/word2vec/data/text8-wv-50.txt'), binary=False)
    vocab = kvecs.index_to_key
    # Loading datasets
    men_df = ds.load_men("data/MEN/MEN_dataset_natural_form_full", vocab=vocab)
    simlex_df = ds.load_simlex("data/SimLex-999/SimLex-999.txt", vocab=vocab)
    wordsim_df = ds.load_wordsim("data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt", vocab=vocab)
    men_pairs = [(w1, w2) for w1, w2 in zip(men_df['w1'], men_df['w2'])]
    simlex_pairs = [(w1, w2) for w1, w2 in zip(simlex_df['word1'], simlex_df['word2'])]
    wordsim_pairs = [(w1, w2) for w1, w2 in zip(wordsim_df['word1'], wordsim_df['word2'])]
    # Computing correlation scores
    men_scores = [float(score) for score in men_df['similarity']]
    simlex_scores = [float(score) for score in  simlex_df['SimLex999']]
    wordsim_scores = [float(score) for score in wordsim_df['similarity']]
    # loading word vectors
    glove_trained_kvecs = [glove_kvecs_store[dim]['train'] for dim in glove_kvecs_store]
    glove_reduced_kvecs_500 = [glove_kvecs_store[500]['pca'][reduced_dim] for reduced_dim in glove_kvecs_store[500]['pca']]
    skipgram_trained_kvecs = [skipgram_kvecs_store[dim]['train'] for dim in skipgram_kvecs_store]
    skipgram_reduced_kvecs_500 = [skipgram_kvecs_store[500]['pca'][reduced_dim] for reduced_dim in skipgram_kvecs_store[500]['pca']]
    cbow_trained_kvecs = [cbow_kvecs_store[dim]['train'] for dim in cbow_kvecs_store]
    cbow_reduced_kvecs_500 = [cbow_kvecs_store[500]['pca'][reduced_dim] for reduced_dim in cbow_kvecs_store[500]['pca']]

    correlation_scores = {}
    correlation_scores['cbow'] = {}
    correlation_scores['cbow']['men'] = {}
    correlation_scores['cbow']['men']['train'] = sim_correlations(men_pairs, men_scores, cbow_trained_kvecs)
    correlation_scores['cbow']['men']['pca'] = sim_correlations(men_pairs, men_scores, cbow_reduced_kvecs_500)
    correlation_scores['cbow']['simlex'] = {}
    correlation_scores['cbow']['simlex']['train'] = sim_correlations(simlex_pairs, simlex_scores, cbow_trained_kvecs)
    correlation_scores['cbow']['simlex']['pca'] = sim_correlations(simlex_pairs, simlex_scores, cbow_reduced_kvecs_500)
    correlation_scores['cbow']['wordsim'] = {}
    correlation_scores['cbow']['wordsim']['train'] = sim_correlations(wordsim_pairs, wordsim_scores, cbow_trained_kvecs)
    correlation_scores['cbow']['wordsim']['pca'] = sim_correlations(wordsim_pairs, wordsim_scores, cbow_reduced_kvecs_500)
    correlation_scores['skipgram'] = {}
    correlation_scores['skipgram']['men'] = {}
    correlation_scores['skipgram']['men']['train'] = sim_correlations(men_pairs, men_scores, skipgram_trained_kvecs)
    correlation_scores['skipgram']['men']['pca'] = sim_correlations(men_pairs, men_scores, skipgram_reduced_kvecs_500)
    correlation_scores['skipgram']['simlex'] = {}
    correlation_scores['skipgram']['simlex']['train'] = sim_correlations(simlex_pairs, simlex_scores, skipgram_trained_kvecs)
    correlation_scores['skipgram']['simlex']['pca'] = sim_correlations(simlex_pairs, simlex_scores, skipgram_reduced_kvecs_500)
    correlation_scores['skipgram']['wordsim'] = {}
    correlation_scores['skipgram']['wordsim']['train'] = sim_correlations(wordsim_pairs, wordsim_scores, skipgram_trained_kvecs)
    correlation_scores['skipgram']['wordsim']['pca'] = sim_correlations(wordsim_pairs, wordsim_scores, skipgram_reduced_kvecs_500)
    correlation_scores['glove'] = {}
    correlation_scores['glove']['men'] = {}
    correlation_scores['glove']['men']['train'] = sim_correlations(men_pairs, men_scores, glove_trained_kvecs)
    correlation_scores['glove']['men']['pca'] = sim_correlations(men_pairs, men_scores, glove_reduced_kvecs_500)
    correlation_scores['glove']['simlex'] = {}
    correlation_scores['glove']['simlex']['train'] = sim_correlations(simlex_pairs, simlex_scores, glove_trained_kvecs)
    correlation_scores['glove']['simlex']['pca'] = sim_correlations(simlex_pairs, simlex_scores, glove_reduced_kvecs_500)
    correlation_scores['glove']['wordsim'] = {}
    correlation_scores['glove']['wordsim']['train'] = sim_correlations(wordsim_pairs, wordsim_scores, glove_trained_kvecs)
    correlation_scores['glove']['wordsim']['pca'] = sim_correlations(wordsim_pairs, wordsim_scores, glove_reduced_kvecs_500)

    plot_similarities(correlation_scores)
    # TODO todo add args output