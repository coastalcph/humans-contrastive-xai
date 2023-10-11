import pickle
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(nrows=3, ncols=2, sharey=True, sharex=True)

for idataset, dataset in enumerate(['biosbias', 'dbpedia-animals']):
    df={}
    for ixai, xai_method in enumerate(['gi', 'gi_norm', 'lrp']):
        df[xai_method] = pd.DataFrame(columns=['roberta', 'gpt2', 't5'])
        for isize, size in enumerate(['small', 'base', 'large']):
            for imodel, model in enumerate(['roberta', 'gpt2', 't5']):
                results_dir = f'../results/{model}-{size}-{dataset}/model-model'
                df[xai_method].loc[size, model] = np.around(np.mean(pickle.load(open(join(results_dir, f'correlation_{xai_method}.pkl'), "rb"))), decimals=2)
        # sns.heatmap(data=df[xai_method], annot=True, vmax=1, vmin=0.3, fmt='.2f', ax=axs[ixai, idataset])
        print(xai_method, '\n', df[xai_method], '\n')
# plt.show()
