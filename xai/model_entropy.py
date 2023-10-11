import pickle
import pandas as pd
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re
import scipy
from os.path import join


def main(root_dir_, dataset, xai_method):
    cm = plt.get_cmap('Set1')
    colors = cm.colors

    models = ["gpt2-small", "gpt2-base", "gpt2-large",
              "roberta-small", "roberta-base", "roberta-large",
              "t5-small", "t5-base", "t5-large"]
    model2idx = {k:v for v, k in enumerate(models)}
    print("\n", dataset.upper(), ",", xai_method.upper())
    print("-" * 100)

    entropy_scores = {modelname: {'contrastive': [], 'non-contrastive': []}
                      for modelname in models}
    entropy_scores_plot = {"model": [], "score": [], "contrastive": []}
    file_not_found = []

    for idr, modelname in enumerate(models):
        print("\n" + modelname)
        root_dir = join(root_dir_, modelname + "-" + dataset)
        # Load dicts with importance scores
        relevance_dict = {}
        relevance_dict_val = {}
        try:
            relevance_dict['contrastive'] = pickle.load(
                open(join(root_dir, f'relevance_contrastive_{xai_method}_{modelname}.pkl'), 'rb'))
            relevance_dict['non-contrastive'] = pickle.load(
                open(join(root_dir, f'relevance_true_{xai_method}_{modelname}.pkl'), 'rb'))
            relevance_dict_val['contrastive'] = pickle.load(
                open(join(root_dir, f'relevance_validation_contrastive_{xai_method}_{modelname}.pkl'), 'rb'))
            relevance_dict_val['non-contrastive'] = pickle.load(
                open(join(root_dir, f'relevance_validation_true_{xai_method}_{modelname}.pkl'), 'rb'))
        except FileNotFoundError:
            file_not_found.append(modelname)
            continue

        relevance_dict['contrastive'] = relevance_dict['contrastive'].append(relevance_dict_val['contrastive'])
        relevance_dict['non-contrastive'] = relevance_dict['non-contrastive'].append(relevance_dict_val['non-contrastive'])

        for mode in ["non-contrastive", "contrastive"]:
            for ix, xx in relevance_dict[mode].iterrows():
                scores = xx['attention']
                e = scipy.stats.entropy(np.abs(scores))
                # e = scipy.stats.entropy(np.clip(scores, a_min=0., a_max=None))
                entropy_scores[modelname][mode].append(e)

                entropy_scores_plot["model"].append(model2idx[modelname])
                entropy_scores_plot["score"].append(e)
                entropy_scores_plot["contrastive"].append(mode == "contrastive")

        entropy_df = pd.DataFrame.from_dict(entropy_scores[modelname])
        e_ttest = scipy.stats.ttest_ind([x for x in entropy_df['contrastive'] if np.isnan(x) == False],
                                        [x for x in entropy_df['non-contrastive'] if np.isnan(x) == False])
        print(f"Non-contrastive\tmean: {entropy_df['non-contrastive'].mean()} std: {entropy_df['non-contrastive'].std()}")
        print(f"Contrastive\tmean: {entropy_df['contrastive'].mean()} std: {entropy_df['contrastive'].std()}")
        print(f"T-test: {e_ttest}")
        # odf.iloc[idr] = {"model": modelname,
        #                  "contrastive": entropy_df['contrastive'].mean(),
        #                  "non-contrastive": entropy_df['non-contrastive'].mean()}

        # ax.hlines(x=idr, xmin=idr-0.5, xmax=idr+1.5, color='black', linestyle='--')
        # ax.hlines(x=idr, xmin=idr-0.5, xmax=idr+1.5, color='black', linestyle='--')

    f, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.5, len(models) - len(file_not_found) + 0.5)
    plot_width = 0.7
    df_all = pd.DataFrame.from_dict(entropy_scores_plot)
    ax = sns.violinplot(data=df_all,
                        x="model", y="score", hue="contrastive", palette="Set1",
                        inner='point', cut=0, width=plot_width)

    h = [Patch(facecolor=colors[0], edgecolor='k', label='contrastive'),
         Patch(facecolor=colors[1], edgecolor='k', label='non-contrastive')]
    ax.legend(handles=h, ncol=1, fontsize=16, loc='lower right')

    ax.set_xticklabels([name for name in models if name not in file_not_found], rotation=10)
    ax.set_ylabel('entropy')
    ax.set_title(f"{dataset} ({xai_method.upper()})")
    # plt.show()

    plt.savefig(f"../bin/entropy_{dataset}_{xai_method}.png")


if __name__ == "__main__":
    root = "/Users/sxk199/mnt/nlp/data/humans-contrastiveXAI"
    dataset = ["biosbias", "dbpedia-animals"]
    xai_methods = ["lrp"]  #"gi", "gi_norm"]
    for d in dataset:
        for m in xai_methods:
            main(root, d, m)