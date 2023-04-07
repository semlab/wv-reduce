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



def analogy_scores(qa_df, kvectors): 
    top_answers = [kvectors.most_similar(positive=[seed2, question], negative=[seed1], topn=1) 
                    for seed1, seed2, question in zip(qa_df['seed1'], qa_df['seed2'], qa_df['question'])]
    corrects = [top_answer[0][0] == answer for top_answer, answer in zip(top_answers, qa_df['answer'])]
    scores = list(map(int, corrects))
    return scores


def analogy_accuracies(qa_df, kvecs_list):
    accuracies = {}
    for kvecs in kvecs_list:
        dim = len(kvecs[0])
        scores = analogy_scores(qa_df, kvecs)
        accuracy = sum(scores)/len(scores)
        accuracies[dim] = accuracy
    return accuracies


def plot_analogies(analogy_scores, filename="analogies.pdf"):
    """
    Plot figure for analogies.
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

    qa_df = ds.load_word2vec_qa("./data/questions-words.txt")
    semantic_qa = ['capital-common-countries', 'capital-world', 'currency',
       'city-in-state', 'family']
    syntactic_qa = ['gram1-adjective-to-adverb',
       'gram2-opposite', 'gram3-comparative', 'gram4-superlative',
       'gram5-present-participle', 'gram6-nationality-adjective',
       'gram7-past-tense', 'gram8-plural', 'gram9-plural-verbs']
    qa_semantic_df = qa_df[qa_df['category'].isin(semantic_qa)]
    qa_syntactic_df = qa_df[qa_df['category'].isin(syntactic_qa)]
    # loading word vectors
    glove_trained_kvecs = [glove_kvecs_store[dim]['train'] for dim in glove_kvecs_store]
    glove_reduced_kvecs_500 = [glove_kvecs_store[500]['pca'][reduced_dim] for reduced_dim in glove_kvecs_store[500]['pca']]
    skipgram_trained_kvecs = [skipgram_kvecs_store[dim]['train'] for dim in skipgram_kvecs_store]
    skipgram_reduced_kvecs_500 = [skipgram_kvecs_store[500]['pca'][reduced_dim] for reduced_dim in skipgram_kvecs_store[500]['pca']]
    cbow_trained_kvecs = [cbow_kvecs_store[dim]['train'] for dim in cbow_kvecs_store]
    cbow_reduced_kvecs_500 = [cbow_kvecs_store[500]['pca'][reduced_dim] for reduced_dim in cbow_kvecs_store[500]['pca']]

    # analogies scores
    analogies_scores = {}
    analogies_scores['cbow'] = {}
    analogies_scores['cbow']['train'] = analogy_accuracies(qa_df, cbow_trained_kvecs)
    analogies_scores['cbow']['pca'] = analogy_accuracies(qa_df, cbow_reduced_kvecs_500)
    analogies_scores['skipgram'] = {}
    analogies_scores['skipgram']['train'] = analogy_accuracies(qa_df, skipgram_trained_kvecs)
    analogies_scores['skipgram']['pca'] = analogy_accuracies(qa_df, skipgram_reduced_kvecs_500)
    analogies_scores['glove'] = {}
    analogies_scores['glove']['train'] = analogy_accuracies(qa_df, glove_trained_kvecs)
    analogies_scores['glove']['pca'] = analogy_accuracies(qa_df, glove_reduced_kvecs_500)
    # TODO call analogy  figure
    plot_analogies(analogies_scores)