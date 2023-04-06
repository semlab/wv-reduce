



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


def load_men(filepath, vocab=None):
    men_df = pd.read_csv(filepath, sep=" ", header=None)
    men_df.rename(columns={0: 'w1', 1:'w2', 2:'similarity'}, inplace=True)
    print(f'Loaded: {men_df.shape}')
    if vocab is not None:
        men_df = men_df[(men_df['w1'].isin(vocab))  & (men_df['w2'].isin(vocab))]
        print(f'Filtered: {men_df.shape}')
    return men_df


def load_simlex(filepath, vocab=None):
    simlex_df = pd.read_csv(filepath, sep="\t")
    print(f'Loaded: {simlex_df.shape}')
    if vocab is not None:
        simlex_df = simlex_df[(simlex_df['word1'].isin(vocab)) & (simlex_df['word2'].isin(vocab))]
        print(f'Filtered: {simlex_df.shape}')
    return simlex_df


def load_wordsim(filepath, vocab=None):
    wordsim_df = pd.read_csv(filepath, sep='\t', header=None)
    wordsim_df.rename(columns={0: 'word1', 1:'word2', 2:'similarity'}, inplace=True)
    print(f'Loaded: {wordsim_df.shape}')
    if vocab is not None:
        wordsim_df = wordsim_df[(wordsim_df['word1'].isin(vocab)) & (wordsim_df['word2'].isin(vocab))]
        print(f'Filtered: {wordsim_df.shape}')
    return wordsim_df