import pickle

import pandas as pd
from scipy.stats import spearmanr, entropy
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xai.xai_utils.tokenization_utils import merge_roberta_tokens
from utils import set_up_dir
import os
import scipy
from os.path import join
import click


@click.command()
@click.option('--modelname', default='gpt2')
@click.option('--xai_method', default='lrp')
@click.option('--correct_only', is_flag=True, help='if only correctly classified samples are considered')
@click.option('--plotting', is_flag=True)
@click.option('--source_path', default='./results')
def main(modelname, xai_method, correct_only, source_path, plotting):

    try:
        data_contrast = pickle.load(
            open(join(source_path, f'relevance_contrastive_{xai_method}_{modelname}.pkl'), 'rb'))
        data_contrast_val = pickle.load(
            open(join(source_path, f'relevance_validation_contrastive_{xai_method}_{modelname}.pkl'), 'rb'))
        data_contrast = pd.concat([data_contrast, data_contrast_val])

        data_true = pickle.load(open(join(source_path, f'relevance_true_{xai_method}_{modelname}.pkl'), 'rb'))
        data_true_val = pickle.load(
            open(join(source_path, f'relevance_validation_true_{xai_method}_{modelname}.pkl'), 'rb'))
        data_true = pd.concat([data_true, data_true_val])

        data_foil = pickle.load(open(join(source_path, f'relevance_foil_{xai_method}_{modelname}.pkl'), 'rb'))
        data_foil_val = pickle.load(
            open(join(source_path, f'relevance_validation_foil_{xai_method}_{modelname}.pkl'), 'rb'))
        data_foil = pd.concat([data_foil, data_foil_val])

        res_dir = os.path.join(source_path, 'model-model')
        set_up_dir(res_dir)

    except FileNotFoundError:
        print('run "run_bios_exp" first')

    if 'biosbias' in source_path:
        class_names = ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician']
    elif 'animals' in source_path:
        class_names = ['Amphibian', 'Arachnid', 'Bird', 'Crustacean', 'Fish', 'Insect', 'Mollusca', 'Reptile']

    label2idx = {k: i for i, k in enumerate(class_names)}
    idx2label = {v: k for k, v in label2idx.items()}

    # correlation = np.empty([len(data['label'])])
    correlation = np.empty([len(data_true)])
    entropy_vals = {key: [] for key in ['true', 'contrastive']}
    mse = np.empty([len(data_true)])

    if 'gpt2' in modelname:
        start_end = [None, None]
    elif 't5' in modelname:
        start_end = [None, -1]
    elif 'roberta' in modelname:
        start_end = [1, -1]
    else:
        raise NotImplementedError

    ii = 0

    logit_diff = np.empty([len(data_true)])
    correlation_dict = {
        'correct': [],
        'misclassified': []
    }
    logit_dict = {
        'correct': [],
        'misclassified': []
    }

    for (row_true, subdf_true), (row_contrast, subdf_contrast) in zip(data_true.iterrows(), data_contrast.iterrows()):

        assert subdf_true['y_true'] == subdf_contrast['y_true']
        assert subdf_true['y_pred'] == subdf_contrast['y_pred']

        if correct_only and subdf_true['y_true'] != subdf_true['y_pred']:
            ii += 1
            logit_diff[ii] = np.nan
            continue

        relevance = subdf_true['attention']
        contrastive = subdf_contrast['attention']
        rho, p = spearmanr(relevance[start_end[0]:start_end[1]], contrastive[start_end[0]:start_end[1]])
        probs = scipy.special.softmax(subdf_contrast['logits'])
        correlation[ii] = rho

        if 't5' not in modelname:
            logit_diff[ii] = probs[subdf_contrast['y_true']] - probs[subdf_contrast['foil']]
            if subdf_true['y_true'] == subdf_true['y_pred']:
                logit_dict['correct'].append(probs[subdf_contrast['y_true']] - probs[subdf_contrast['foil']])
                correlation_dict['correct'].append(rho)
            else:
                logit_dict['misclassified'].append(probs[subdf_contrast['y_true']] - probs[subdf_contrast['foil']])
                correlation_dict['misclassified'].append(rho)

        entropy_vals['true'].append(entropy(relevance[start_end[0]:start_end[1]]))
        entropy_vals['contrastive'].append(entropy(contrastive[start_end[0]:start_end[1]]))
        if len(relevance) < 3:
            continue
        try:
            mse[ii] = mean_squared_error(
                relevance[start_end[0]:start_end[1]] / np.sum(relevance[start_end[0]:start_end[1]]),
                contrastive[start_end[0]:start_end[1]] / np.sum(contrastive[start_end[0]:start_end[1]]))
        except ValueError:
            import pdb;pdb.set_trace()
        ii += 1

    pickle.dump(correlation, open(join(res_dir, f'correlation_{xai_method}.pkl'), 'wb'))
    pickle.dump(entropy_vals, open(join(res_dir, f'entropy_{xai_method}.pkl'), 'wb'))
    pickle.dump(mse, open(join(res_dir, f'mse_{xai_method}.pkl'), 'wb'))

    if 't5' not in modelname:
        rho2 = spearmanr(logit_dict['correct'] + logit_dict['misclassified'],
                         correlation_dict['correct'] + correlation_dict['misclassified'])[0]
        print(modelname, xai_method, f"rho={rho2:.2f}")
        plt.scatter(logit_dict['correct'], correlation_dict['correct'], label='correct')
        plt.scatter(logit_dict['misclassified'], correlation_dict['misclassified'], label='misclassified')
        plt.legend()
        plt.xlabel("difference in prob. prediction vs. foil")
        plt.ylabel("correlation contr. vs. non-contr. explanation")
        plt.title(f"rho={rho2:.2f}")
        plt.savefig(os.path.join(res_dir, f'scatter_{xai_method}'), dpi=300)
        plt.close()

    if plotting:

        min_idx = np.argsort(correlation)

        for idx in [20,68, 220, 1839, 1840]:
        # for idx in range(len(data_true)):
            misclassified = False
            threshold = 10

            label = label2idx[data_true.iloc[idx]['y_pred'][0]] if 't5' in modelname else int(
                data_true.iloc[idx]['y_pred'])
            foil = label2idx[data_contrast.iloc[idx]['foil'][0]] if 't5' in modelname else int(
                data_contrast.iloc[idx]['foil'])

            factor = int(np.ceil(len(data_true.iloc[idx]['tokens'])/threshold))
            factor = 1.5 * factor if factor == 3 else factor
            fig, axs = plt.subplots(3, 1, figsize=(15, factor))

            if 't5' not in modelname:
                logits = data_true.iloc[idx]['logits']
                probs = scipy.special.softmax(logits)

            if label != int(data_true.iloc[idx]['y_true']):
                misclassified = True

            if label == foil:
                plt.close()
                continue

            # Normalize standard explanation by the maximal relevance score across all explanations
            r_normalization = np.max(np.abs([
                data_foil.iloc[idx]['attention'][start_end[0]:start_end[1]],
                data_true.iloc[idx]['attention'][start_end[0]:start_end[1]]
            ]))

            if 't5' in modelname:
                words, relevance = merge_roberta_tokens(data_true.iloc[idx]['tokens'],
                                                        data_true.iloc[idx]['attention'],
                                                        sep_token="▁")
            else:
                words, relevance = merge_roberta_tokens(data_true.iloc[idx]['tokens'], data_true.iloc[idx]['attention'])

            append = [0 for _ in range(threshold - len(words[start_end[0]:start_end[1]]) % threshold)]
            tok_append = ['PAD' for _ in range(len(append))]
            reshape = (-1, threshold)

            R = np.array(relevance[start_end[0]:start_end[1]])  # / r_normalization
            R_plot = np.array(R.tolist() + append)
            sns.heatmap(np.array(R_plot).reshape(reshape),
                        annot=np.array(words[start_end[0]:start_end[1]] + tok_append)[np.newaxis, :].reshape(reshape),
                        fmt='', ax=axs[0], cmap='vlag', vmin=-r_normalization, vmax=r_normalization,
                        annot_kws={"size": 10},
                        cbar=False)
            axs[0].set_xticks([])
            axs[0].set_yticks([])

            if 't5' not in modelname:
                title = 'prediction {} p={}'.format(idx2label[label], '{:0.2f}'.format(probs[label]))
                axs[0].set_title(title, fontsize=12)
                words, relevance_foil = merge_roberta_tokens(data_true.iloc[idx]['tokens'],
                                                             data_foil.iloc[idx]['attention'])

            else:
                title = 'prediction: {}'.format(idx2label[label])
                axs[0].set_title(title, fontsize=12)
                words, relevance_foil = merge_roberta_tokens(data_true.iloc[idx]['tokens'],
                                                             data_foil.iloc[idx]['attention'],
                                                             sep_token="▁")

            # Plot foil
            append = [0 for _ in range(threshold - len(words[start_end[0]:start_end[1]]) % threshold)]
            tok_append = ['PAD' for _ in range(len(append))]
            reshape = (-1, threshold)

            R = np.array(relevance_foil[start_end[0]:start_end[1]])  # / r_normalization
            R_plot = np.array(R.tolist() + append)
            sns.heatmap(np.array(R_plot).reshape(reshape),
                        annot=np.array(words[start_end[0]:start_end[1]] + tok_append)[np.newaxis, :].reshape(reshape),
                        fmt='', ax=axs[1], cmap='vlag',
                        vmin=-r_normalization, vmax=r_normalization,
                        annot_kws={"size": 10}, cbar=False)
            axs[1].set_xticks([])
            axs[1].set_yticks([])

            if 't5' not in modelname:
                title = 'foil {} p={}'.format(idx2label[foil], '{:0.2f}'.format(probs[foil]))
                axs[1].set_title(title, fontsize=12)
                words, contrastive = merge_roberta_tokens(data_true.iloc[idx]['tokens'],
                                                          data_contrast.iloc[idx]['attention'])

            else:
                title = 'foil: {}'.format(idx2label[foil])
                axs[1].set_title(title, fontsize=12)
                words, contrastive = merge_roberta_tokens(data_true.iloc[idx]['tokens'],
                                                          data_contrast.iloc[idx]['attention'],
                                                          sep_token="▁")


            # Normalize contrastive explanation by its maximal relevance score
            r_normalization = np.max(np.abs(contrastive[start_end[0]:start_end[1]]))
            R = np.array(contrastive[start_end[0]:start_end[1]])  # / r_normalization
            R_plot = np.array(R.tolist() + append)
            sns.heatmap(np.array(R_plot).reshape(reshape),
                        annot=np.array(words[start_end[0]:start_end[1]] + tok_append)[np.newaxis, :].reshape(reshape),
                        fmt='', ax=axs[2], cmap='vlag', vmin=-r_normalization, vmax=r_normalization,
                        annot_kws={"size": 10}, cbar=False)

            axs[2].set_xticks([])
            axs[2].set_yticks([])
            axs[2].set_title(f'contrastive: {idx2label[label]} vs. {idx2label[int(foil)]}', fontsize=12)

            if misclassified:
                plt.savefig(os.path.join(res_dir, f'{xai_method}_{str(idx)}_{correlation[idx]:.2f}_wrong.png'),
                            bbox_inches='tight', dpi=300)
            else:
                plt.savefig(os.path.join(res_dir, f'{xai_method}_{str(idx)}_{correlation[idx]:.2f}.png'),
                            bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == '__main__':
    main()
