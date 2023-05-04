import json
import pandas as pd

def compute_deltas(init_score, scores):
    """
    :param init_score: initial score
    :param scores: dictionary
    """
    deltas = [100*(score - init_score)/init_score_500 for score in scores.values()]
    return deltas


def compute_dataset_deltas(dataset, scores_dict):
    """
    To be used for correlation
    """
    # init score in ['train']['500']
    deltas = {}
    deltas['cbow_train'] = compute_deltas(
        scores_dict['cbow'][dataset]['train']['500'],
        scores_dict['cbow'][dataset]['train']
    )
    deltas['cbow_pca'] = compute_deltas(
        scores_dict['cbow'][dataset]['train']['500'],
        scores_dict['cbow'][dataset]['pca']
    )
    deltas['skipgram_train'] = compute_deltas(
        scores_dict['skipgram'][dataset]['train']['500'],
        scores_dict['skipgram'][dataset]['train']
    )
    deltas['skipgram_pca'] = compute_deltas(
        scores_dict['skipgram'][dataset]['train']['500'],
        scores_dict['skipgram'][dataset]['pca']
    )
    deltas['glove_train'] = compute_deltas(
        scores_dict['glove'][dataset]['train']['500'],
        scores_dict['glove'][dataset]['train']
    )
    deltas['glove_pca'] = compute_deltas(
        scores_dict['glove'][dataset]['train']['500'],
        scores_dict['glove'][dataset]['pca']
    )
    # removing score for 500
    deltas['cbow_train'] = deltas['cbow_train'][:-1]
    deltas['skipgram_train'] = deltas['skipgram_train'][:-1]
    deltas['glove_train'] = deltas['glove_train'][:-1]
    return deltas


def latex_table(deltas, row_idx=None, col_idx=None, caption=None):
    if row_idx is None:
        row_idx= list(range(50,500,50))
    if col_idx is None:
        col_idx = pd.MultiIndex.from_arrays([
            ["CBOW", "CBOW", "Skipgram", "Skipgram", "GloVe","GloVe"],
            ["Trained", "Reduced","Trained", "Reduced","Trained", "Reduced"]
        ])
    df = pd.DataFrame( #deltas
        {
            ("CBOW", "Trained"): deltas['cbow_train'], 
            ("CBOW", "Reduced"): deltas['cbow_pca'],
            ("Skipgram","Trained"): deltas['skipgram_train'], 
            ("Skipgram","Reduced"): deltas['skipgram_pca'],
            ("GloVe","Trained"): deltas['glove_train'], 
            ("GloVe","Reduced"): deltas['glove_pca']}
        , columns=col_idx, index=row_idx)
    df.sort_index(ascending=False, inplace=True)
    styler = df.style
    styler.background_gradient(cmap="inferno", 
        #subset="Equity", 
        vmin=0, vmax=100)
    latex = df.to_latex(caption=caption, 
        index=True, 
        float_format="{:.1f}".format)
    return latex

    

if __name__ == "__main__":
    
    analogies_scores = None
    with open("analogies_scores.json") as score_file:
        print(type(score_file))
        analogies_scores = json.load(score_file)

    init_score_500 = analogies_scores['cbow']['train']['500']
    dimensions = list(range(50,500))
    dimensions.reverse()
    deltas = {}
    deltas['cbow_train'] = compute_deltas(
        analogies_scores['cbow']['train']['500'],
        analogies_scores['cbow']['train']
    )
    deltas['cbow_train'] = deltas['cbow_train'][:-1]
    deltas['cbow_pca'] = compute_deltas(
        analogies_scores['cbow']['train']['500'],
        analogies_scores['cbow']['pca']
    )
    deltas['skipgram_train'] = compute_deltas(
        analogies_scores['skipgram']['train']['500'],
        analogies_scores['skipgram']['train']
    )
    deltas['skipgram_train'] = deltas['skipgram_train'][:-1]
    deltas['skipgram_pca'] = compute_deltas(
        analogies_scores['skipgram']['train']['500'],
        analogies_scores['skipgram']['pca']
    )
    deltas['glove_train'] = compute_deltas(
        analogies_scores['glove']['train']['500'],
        analogies_scores['glove']['train']
    )
    deltas['glove_train'] = deltas['glove_train'][:-1]
    deltas['glove_pca'] = compute_deltas(
        analogies_scores['glove']['train']['500'],
        analogies_scores['glove']['pca']
    )
    latex = latex_table(deltas)
    print(latex)



    print("\n\n**** Correlations ***")
    correlations_scores = None
    with open("correlations-scores.json") as score_file:
        print(type(score_file))
        correlations_scores = json.load(score_file)
    men_deltas = compute_dataset_deltas("men", correlations_scores)
    men_latex = latex_table(men_deltas)
    print("\n\n**** MEN ***")
    print(men_latex)
    simlex_deltas = compute_dataset_deltas("simlex", correlations_scores)
    simlex_latex = latex_table(simlex_deltas)
    print("\n\n**** Simlex ***")
    print(simlex_latex)
    wordsim_deltas = compute_dataset_deltas("wordsim", correlations_scores)
    wordsim_latex = latex_table(wordsim_deltas)
    print("\n\n**** Wordsim ***")
    print(wordsim_latex) 
