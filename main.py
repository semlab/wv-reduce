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



def similarity_matrix_old(wordpairs, sim_scores, trained_vecs_paths, reduced_vecs_paths):
    score_mat = []
    
    for i, trained_vecs_path in enumerate(trained_vecs_paths):
        score_mat.append([])
        kvecs_trained = KeyedVectors.load_word2vec_format(datapath(os.path.join(trained_folder, trained_vecs_path)))
        
        for j, reduced_vecs_path in enumerate(reduced_vecs_paths):
            kvecs_reduced = KeyedVectors.load_word2vec_format(datapath(os.path.join(reduced_folder, reduced_vecs_path)))
            score = sim_correlation(wordpairs, kvecs_trained, kvecs_reduced)
            score_mat[i].append(score)
    return np.array(score_mat)



def similarity_matrix(wordpairs, sim_scores, kvecs_store):
    n_dim = len(kvecs_store.keys())
    score_mat = np.empty((n_dim, n_dim,))
    score_mat[:] = np.nan
    for trained_idx, trained_dim in enumerate(kvecs_store):
        trained_kvecs = kvecs_store[trained_dim]['train']
        score = sim_correlation(wordpairs, sim_scores, trained_kvecs)
        line = n_dim - 1 - trained_idx
        score_mat[line, line] = float(score)
        for reduced_idx, reduced_dim in enumerate(kvecs_store[trained_dim]['pca']):
            reduced_kvecs = kvecs_store[trained_dim]['pca'][reduced_dim]
            col = line + 1 + reduced_idx 
            score_mat[line, col] = sim_correlation(wordpairs, sim_scores, reduced_kvecs)
    return score_mat



def load_keyedvectors(root, name=None, dimensions=None):
    """
    Load the trained vectors of a particular model into a nested dictionary
    :param root: the root path of the trained and and reduced vectors of a 
        particular word vector model. the root path hierachy follows the 
    :param name: the name of the model
    :param dimensions: the list of vector dimensions for different trained 
        instances
    """
    # TODO explain the the hierarchy of the root path 
    if name is None: name = os.path.basename(root.rstrip('/'))
    if dimensions is None: dimensions = list(range(50, 550,50))
    kvecs_store = {}
    for idx, dimension in enumerate(dimensions):
        filepath_train = os.path.join(root, 'train', f'{name}-{dimension}.txt')
        if not os.path.exists(filepath_train): continue
        kvecs_store[dimension] = {}
        kvecs_store[dimension]['train'] = KeyedVectors.load_word2vec_format(filepath_train, no_header=True)
        if os.path.exists(os.path.join(root, 'pca')):
            kvecs_store[dimension]['pca'] = {}
            reduced_dims = [reduced_dim for reduced_dim in dimensions if reduced_dim < dimension]
            for reduced_dim in reduced_dims:
                filepath_reduced = os.path.join(root, 'pca', f'{name}-{dimension}-pca-{reduced_dim}.txt')
                if not os.path.exists(filepath_reduced): continue
                kvecs_store[dimension]['pca'][reduced_dim] = KeyedVectors.load_word2vec_format(filepath_reduced)#, no_header=True)
    return kvecs_store


def figure3(sim_matrix):
    """
    :param sim_matrix: matrix of similarity scores (for a particular dataset)
    """
    n_models = 3#len(correlation_scores)
    fig, axs = plt.subplots(n_models, 1, figsize=(5, 5*n_models))
    for idx_model, model in enumerate(sim_matrix):
        pcm = axs[idx_model].imshow(sim_matrix[model], cmap='autumn', interpolation='nearest')
        axs[idx_model].set_title(f'{model}')
        rdimensions = list(reversed(dimensions))
        axs[idx_model].set_xticks(np.arange(len(rdimensions)), rdimensions)
        axs[idx_model].set_yticks(np.arange(len(rdimensions)), rdimensions)
        fig.colorbar(pcm, ax=axs[idx_model])
    plt.savefig('figs/trained-reduced-correlations.pdf')
    #plt.show()


def figure1(correlation_scores, filename='correlations.pdf'):
    """
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
    #plt.show()



def figure2(analogy_scores, filename="analogies.pdf"):
    """
    :param analogy_scores: nested dictionary storing analogy scores
        by model then trained ('train') or reduced ('reduced')
    """
    dimensions = list(range(0,550,50))
    n_models = len(correlation_scores)
    plot_styles = [{'marker':'o', 'linestyle':'-', 'color':'darkorange'},
                {'marker':'^', 'linestyle':'--', 'color':'green'}
               ]
    fig, axs = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    for idx_model, model in enumerate(analogies_scores):
        for idx_method, method in enumerate(analogies_scores[model]):
            x = analogies_scores[model][method].keys()
            y = analogies_scores[model][method].values()
            #axs[idx_dataset, idx_model].grid(visible=True)#, axis='x')
            axs[idx_model].plot(x, y, label=f'{method}', **plot_styles[idx_method])
            axs[idx_model].set_xlim(max(dimensions)+50, min(dimensions))
            axs[idx_model].legend(loc='lower left')
            axs[idx_model].set_title(f'{model}')
            # TODO grid, share axis? 
    fig.tight_layout()
    plt.savefig('figs/analogies.pdf')
    #plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--vectors-root', required=True,
            help="Root path of the saved trained/reduced vectors")
    args = parser.parse_args()
    args_dic = {}
    args_dic['vectors_root'] = args.vectors_root
    return args_dic


if __name__ == "__main__":
    args = get_args()
    #load the vectors
    # TODO give the root path as an arg
    # '/home/gr0259sh/Projects/devel/exp2210/out/glove'
    glove_kvecs_store = load_keyedvectors(os.path.join(args['vectors_root'], 'glove')
    cbow_kvecs_store = load_keyedvectors(os.path.join(args['vectors_root'], 'cbow')
    skipgram_kvecs_store = load_keyedvectors(os.path.join(args['vectors_root'], 'skipgram')
