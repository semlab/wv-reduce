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

# Figure for the similarity correlation


def sim_correlation(wordpairs, arg1, arg2):
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



def plot_similarities(correlation_scores, filename='correlations.pdf'):
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
    # correlation scores
    men_df = ds.load_men("data/MEN/MEN_dataset_natural_form_full")
    simlex_df = ds.load_simlex("data/SimLex-999/SimLex-999.txt")
    wordsim_df = ds.load_wordsim("data/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt")
    men_pairs = [(w1, w2) for w1, w2 in zip(men_df['w1'], men_df['w2'])]
    simlex_pairs = [(w1, w2) for w1, w2 in zip(simlex_df['word1'], simlex_df['word2'])]
    wordsim_pairs = [(w1, w2) for w1, w2 in zip(wordsim_df['word1'], wordsim_df['word2'])

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

    plot_similarities(correlation_scores):
    # TODO todo add args output