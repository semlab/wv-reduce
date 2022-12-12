import gensim
from gensim.models import KeyedVectors
from gensim.models import datapath

from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE


def load_word2vec_qa(filepath):
    datad = dict()
    datad['categories'] = []
    datad['seeds1'] = []
    datad['seeds2'] = []
    datad['questions'] = []
    datad['answers'] = []
    with open(filepath) as f:
        category = None
        for line in f:
            split = line.split()
            if split[0] == ':':
                category = split[1]
            elif len(split) == 4 and category is not None:
                datad['categories'].append(category)
                datad['seeds1'].append(split[0])
                datad['seeds2'].append(split[1])
                datad['questions'].append(split[2])
                datad['answers'].append(split[3])
    return datad




def load_vectors(filename):
    """Load pretrained word vectors"""
    word_vectors = {}
    with open(filename) as f:
        bad_lines_start = []
        for line in f:
            split = line.split()
            try:
                word_vectors[split[0]] = np.array([float(x) for x in split[1:]])
            except ValueError:
                #bad_lines_start.append(split[0])
                print(f"Warning bad line start with: '{split[0]}'")
    return word_vectors


def reduce(keyedvectors, model, to_size):
    """
    Dimensionality reduction of word vectors.
    keyedvectors: word vectors
    model: dimensionality reduction model
    """
    assert keyedvectors.vector_size > to_size, f"The target size ({to_size}) should be less than the original vectors size ({keyedvectors.vector_size})"
    X = [keyedvectors[idx] for idx in range(len(keyedvectors))]
    X_fitted = model.fit_transform(X)
    kvecs = KeyedVectors(vector_size=to_size, count=len(keyedvectors))
    kvecs.add_vectors(keyedvectors.index_to_key, list(X_fitted))
    return kvecs



def mrr(answers, correct, weights=None):
    """
    Mean Reciprocal Rank (MRR) from a list of guesses
    :param answers: list of guesses
    :param correct: the correct answer
    :param weights: list of the guesses weight. Default is None and the MRR is
        computed unweighted
    :returns: the MRR
    """
    score = 0
    for idx, answer in enumerate(answers):
        if answer == correct:
            score = 1/idx
            if weights is not None:
                score += .01 * weights[idx]
            break
    return score


def qa_mrr(qa_df, keyedvectors, weighted=False):
    """
    Mean Reciprocal Rank for dataframe based on dataset of word2vec analogies
    """
    vocab = keyedvectors.index_to_key
    df = qa_df[(qa_df['seeds1'].isin(vocab)) 
         & (qa_df['seeds2'].isin(vocab)) 
         & (qa_df['questions'].isin(vocab)) 
         & (qa_df['answers'].isin(vocab))]
    answers_vecs = [seed1 - seed2 + question 
                for seed1, seed2, question in zip(
                    keyedvectors[df['seeds1']], 
                    keyedvectors[df['seeds2']], 
                    keyedvectors[df['questions']])
               ]
    answers_list = [keyedvectors.similar_by_vector(vec, topn=10) for vec in answers_vecs]
    scores = []
    for answers_weights, correct in zip(answers_list, df['answers']):
        answers, weights = map(list, zip(*answers_weights))
        if weighted:
            scores.append(mrr(answers, correct, weights))
        else:
            scores.append(mrr(answers, correct))
    return scores



