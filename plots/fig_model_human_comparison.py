import pickle
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

importance_aggregator = 'max'
xai_method = 'lrp'

metric = 'cohen'
selected_classes = ['physician', 'dentist', 'all']
plt.close()
fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=[5, 5])
cbar_ax = fig.add_axes([.91, .33, .03, .33])
fontsize=10

for imodel, model in enumerate(['roberta', 'gpt2', 't5']):
    df = {}
    results_dir = f'../results/{model}-base-biosbias'

    scores_contrastive = pickle.load(
                    open(join(results_dir, f'human-model/contrastive_f1_cohen_{xai_method}_{importance_aggregator}.pkl'),
                         'rb'))

    scores_noncontrastive = pickle.load(
        open(
            join(results_dir, f'human-model/non-contrastive_f1_cohen_{xai_method}_{importance_aggregator}.pkl'),
            'rb'))

    # metrics = ['cohen']
    for current_class in selected_classes:
        df[current_class] = pd.DataFrame(columns=['model_contr', 'model_non-contr'])
        df[current_class].loc['hum_contr'] = [scores_contrastive[metric]['contrastive'][current_class],
                                              scores_contrastive[metric]['non-contrastive'][current_class]]
        df[current_class].loc['hum_non-contr'] = [scores_noncontrastive[metric]['contrastive'][current_class],
                                                  scores_noncontrastive[metric]['non-contrastive'][
                                                      current_class]]

    for ii, current_class in enumerate(selected_classes):
        # if metric in ['f1', 'cohen']:
        sns.heatmap(df[current_class], ax=axs[imodel, ii], vmin=0.1, vmax=0.55, annot=True,
                    cmap='mako', cbar=True if imodel+ii==4 else False, cbar_ax=cbar_ax,
                    square=True)

        if imodel==2:
            axs[2, ii].set_xticklabels(['contr.', 'non-contr.'], fontsize=fontsize-1, rotation=0)
            axs[0, ii].tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)
            axs[2, ii].tick_params(bottom=False, labelbottom=False)
            axs[0, ii].set_xticklabels(['contr.', 'non-contr.'], fontsize=fontsize - 1, rotation=0)
            axs[0, ii].set_xlabel("model")
            axs[0, ii].xaxis.set_label_position('top')


        axs[imodel, 0].set_yticklabels(['contr.', 'non-contr.'], fontsize=fontsize-1, rotation=90)
        axs[imodel, 0].set_ylabel("human")
        # axs[0, ii].tick_params()

axs[0, 0].set_title(selected_classes[0], fontsize=fontsize)
axs[0, 1].set_title('RoBERTa-base \n' + selected_classes[1], fontsize=fontsize)
axs[0, 2].set_title(selected_classes[2], fontsize=fontsize)
axs[1, 1].set_title('GPT2-base', fontsize=fontsize)
axs[2, 1].set_title('T5-base', fontsize=fontsize)
plt.subplots_adjust(wspace=0, hspace=0.35)
plt.savefig("model_human_comparison.png", dpi=300, bbox_inches='tight')

