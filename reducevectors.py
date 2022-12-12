import os
import argparse
import gensim
from gensim.models import KeyedVectors
#from gensim.test.utils import datapath

from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE




def dim_reduce(keyedvectors, model, to_size):
    """
    Dimensionality reduction of word vectors.
    keyedvectors: word vectors
    model: dimensionality reduction model
    """
    assert len(keyedvectors[0]) > to_size, "The target size: " +str(to_size) + " should be less than the original vectors size: " + str(len(keyedvectors[0]))
    X = [keyedvectors[idx] for idx in range(len(keyedvectors))]
    X_fitted = model.fit_transform(X)
    assert len(X) == len(X_fitted)
    kvecs = KeyedVectors(vector_size=to_size, count=len(keyedvectors))
    #kvecs.add_vectors(keyedvectors.index_to_key, list(X_fitted))
    for idx in range(len(keyedvectors)):
        kvecs.add_vector(keyedvectors.index_to_key[idx], X_fitted[idx])
    assert len(keyedvectors) == len(kvecs)
    return kvecs


# TODO change 'size' arg to 'dim'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, 
            help="Word vectors file to be reduced")
    parser.add_argument('-o', '--output', help="output filename reduced")
    parser.add_argument('-s', '--size', nargs="+", required=True, type=int, 
            help="Target sizes (separated by space)")
    parser.add_argument('--no-header', action='store_true', 
            help="Set if the pretrained vector file have a header. Note word2vec has, GloVe doesn't")
    args = parser.parse_args()
    args_dic = {}
    args_dic["input"] = args.input
    args_dic["output"] = args.output
    args_dic["size"] = args.size
    args_dic["no_header"] = args.no_header
    return args_dic

if __name__ == "__main__":
    args = parse_args()
    if args['output'] is None:
        args['output'] = os.path.basename(args["input"]) 
        args['output'] = os.path.splitext(args['output'])[0]
    for size in args["size"]:
        model = PCA(n_components=size)
        kvecs = KeyedVectors.load_word2vec_format(args["input"], 
                no_header=args["no_header"], binary=False)
        if size < len(kvecs[0]):
            reduced_kvecs = dim_reduce(kvecs, model, size)
            reduced_kvecs.save_word2vec_format(f"{args['output']}-pca-{size}.txt")
        else:
            print(f"Skipping... PCA dimension: {size} should be less than keyedvectors dimension: {len(kvecs[0])}")


