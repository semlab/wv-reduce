

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
    plt.savefig('figs/analogies.pdf')
    #plt.show()


if __name__ == "__main__":
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