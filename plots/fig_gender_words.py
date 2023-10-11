import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

cm = plt.get_cmap('Set3')
colors = cm.colors

# Load Data
root = "../data/rationales-biosbias"

lrp = pd.read_csv(f"{root}/analysis_gender_lrp.csv")
gi = pd.read_csv(f"{root}/analysis_gender_gi.csv")
gradnorm = pd.read_csv(f"{root}/analysis_gender_gi_norm.csv")

pos = pd.read_csv(f"{root}/analysis_pos_lrp.csv")
n_words = {}
labels = ['nurse', 'psychologist', 'physician', 'dentist', 'surgeon']

for xai in ['lrp', 'gi', 'gi_norm']:
    n_words[xai] = {}
    pos = pd.read_csv(f"{root}/analysis_pos_{xai}.csv")
    pos = pos.rename(columns={'class': 'labels'})
    for model in pos.model.unique():
        n_words[xai][model] = {}
        for label in labels:
            n_words[xai][model][label] = int(5*len(pos.query('model==@model and labels==@label'))/2)
            assert (int(5 * len(pos.query('model==@model and labels==@label and contrastive==True'))) == int(
                5 * len(pos.query('model==@model and labels==@label and contrastive==False'))))

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=[5, 6.5], sharex=True, sharey=True)
cbar_ax = fig.add_axes([.91, .25, .03, .5])
contr_title = {'contrastive': 'contr.',
               'non-contrastive': 'non-contr.'}
xai_title = {'lrp': 'LRP',
             'gi': 'GxI',
             'gi_norm': 'gradNorm'}
for icontr, contr in enumerate(['contrastive', 'non-contrastive']):
    labels = ['nurse', 'psychologist', 'physician', 'dentist', 'surgeon']
    ii = 0
    df_plot = pd.DataFrame(columns=labels.extend(['xai_method', 'model']))
    for xai_method, df in zip(['lrp', 'gi', 'gi_norm'], [lrp, gi, gradnorm]):
        df = df.rename(columns={'class': 'labels'})
        for modelbase in ['roberta', 'gpt2', 't5']:
            for modelsize in ['small', 'base', 'large']:
                model = modelbase + '-' + modelsize
                df_plot.loc[ii, 'xai_method'] = xai_title[xai_method]
                df_plot.loc[ii, 'model'] = model
                for label in df['labels'].unique():
                    denominator = n_words[xai_method][model][label]
                    if contr=='contrastive':
                        df_plot.loc[ii, label] = float(df.query(
                            'model==@model and labels==@label and contrastive==True').value_counts().sum()) / denominator
                    else:
                        df_plot.loc[ii, label] = float(df.query(
                            'model==@model and labels==@label and contrastive==False').value_counts().sum()) / denominator
                ii += 1
    # vmin = df_plot[labels[:-2]].values.min()
    # vmax = df_plot[labels[:-2]].values.max()
    vmin=0
    vmax=0.25
    for imodelbase, modelbase in enumerate(['roberta', 'gpt2', 't5']):
        for imodelsize, modelsize in enumerate(['base']):
            model = modelbase + '-' + modelsize
            sns.heatmap(data=df_plot.query("model==@model")[['nurse', 'surgeon', 'xai_method']].set_index('xai_method').transpose(),
                        annot=True,
                        square=True,
                        vmin=vmin,
                        vmax=vmax,
                        fmt=".2f",
                        ax=axs[imodelbase, icontr],
                        cbar=True if imodelbase==1 and icontr==1 else False,
                        cbar_ax=cbar_ax,
                        cmap='mako')

            if imodelbase==0:
                axs[imodelbase, icontr].set_title(f'{contr}\n{model}')
            else:
                axs[imodelbase, icontr].set_title(f'{model}')
        axs[imodelbase, icontr].set_xlabel("")

plt.subplots_adjust(wspace=0.05, hspace=0.025)
plt.savefig(f'gendered_words.png', dpi=300, bbox_inches='tight')
plt.close()
