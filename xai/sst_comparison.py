import pandas as pd
import click
import pickle
from os.path import join
import re
from tqdm import tqdm
import numpy as np
from xai.xai_utils.tokenization_utils import merge_subwords_words
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def rat_num(len_sen, rationale):
    try:
        rationale = [el - 1 for el in list(map(int, rationale.split(",")))] if len(rationale.split(",")) > 1 else [
            int(float(rationale)) - 1]
        return [1 if idx in rationale else 0 for idx in range(len_sen)]
    except ValueError:
        import pdb;
        pdb.set_trace()


@click.command()
@click.option('--modelname', default="coastalcph/roberta-base-sst2", help='path to finetuned model on hf')
@click.option('--model_type', default="roberta", help='Model type')
@click.option('--importance_aggregator', default="max", help='aggregation method (max, mean, sum)')
@click.option('--results_dir_base', default='./results')
@click.option('--xai_method', default='lrp')
# @click.option('--dataset_name', default="sst2", help='name of dataset_name')
def main(modelname, model_type, importance_aggregator, results_dir_base, xai_method):
    df_results = {}

    for dataset_name in ['sst2', 'dynasent']:

        results_dir = join(results_dir_base, f'roberta-base-{dataset_name}')

        # loading human annotations
        if dataset_name == 'sst2':
            annotation = pd.read_csv('./data/rationales-demographics/SST_annot_before_exclusions.csv') \
                .query("label==original_label").query("attentioncheck=='PASSED'")
            df_gaze = pd.read_pickle('./data/SST/sst_importance.pkl')
        elif dataset_name == 'dynasent':
            annotation = pd.read_csv('./data/rationales-demographics/dynasent_annot_before_exclusions.csv') \
                .query("label==original_label").query("attentioncheck=='PASSED'")
        else:
            raise NotImplementedError
        annotation['len_sen'] = annotation['sentence'].apply(lambda x: len(x.split(" ")))
        annotation['rationale_numeric'] = annotation.apply(lambda x: rat_num(x.len_sen, x.rationale_index), axis=1)
        annotation['normalized_text'] = annotation['sentence'].apply(lambda x: re.sub('[^a-z]', '', "".join(x).lower()))

        # loading model explanations
        if 'gpt2-biosbias' in modelname:
            modelbase = "-".join(modelname.split("/")[1].split("-")[:1])
        else:
            modelbase = "-".join(modelname.split("/")[1].split("-")[:2])
        relevance_dict = {}
        relevance_dict_val = {}
        relevance_dict['non-contrastive'] = pickle.load(
            open(join(results_dir, f'relevance_true_{xai_method}_{modelbase}.pkl'), 'rb')).drop_duplicates(
            subset=['tokens'])
        relevance_dict['contrastive'] = pickle.load(
            open(join(results_dir, f'relevance_contrastive_{xai_method}_{modelbase}.pkl'), 'rb')).drop_duplicates(
            subset=['tokens'])
        relevance_dict_val['contrastive'] = pickle.load(
            open(join(results_dir, f'relevance_validation_contrastive_{xai_method}_{modelbase}.pkl'),
                 'rb')).drop_duplicates(subset=['tokens'])
        relevance_dict_val['non-contrastive'] = pickle.load(
            open(join(results_dir, f'relevance_validation_true_{xai_method}_{modelbase}.pkl'), 'rb')).drop_duplicates(
            subset=['tokens'])
        relevance_dict['contrastive'] = pd.concat([relevance_dict['contrastive'], relevance_dict_val['contrastive']])
        relevance_dict['non-contrastive'] = pd.concat(
            [relevance_dict['non-contrastive'], relevance_dict_val['non-contrastive']])

        # Importance aggregator
        if importance_aggregator == 'max':
            importance_aggregator_name = importance_aggregator
            importance_aggregator_func = np.max
        elif importance_aggregator == 'mean':
            importance_aggregator_name = importance_aggregator
            importance_aggregator_func = np.mean
        elif importance_aggregator == 'sum':
            importance_aggregator_name = importance_aggregator
            importance_aggregator_func = np.sum
        else:
            raise NotImplementedError

        merging_error = 0
        missing = 0
        valid_samples = 0
        valid_gaze_samples = 0

        df_results[dataset_name] = pd.DataFrame(columns=['gaze', 'annotations', 'non-contrastive', 'contrastive'])
        y_pred_old = []
        y_pred_cont_old = []
        y_gaze_cont_old = []
        y_pred_cont_gaze_old = []
        for mode in relevance_dict:
            corr = []
            y_gaze_cont = []
            y_true = []
            y_pred = []
            y_gaze = []
            y_true_gaze = []
            y_pred_gaze = []
            y_pred_cont = []
            y_gaze_cont = []
            y_pred_cont_gaze = []
            for ix, xx in tqdm(relevance_dict[mode].iterrows(), desc='iterating through datapoints'):
                normalized_text = re.sub('[^a-z]', '', "".join(
                    xx['tokens']).replace('</s>', "").replace('<s>', '').lower())
                if len(annotation.query('normalized_text==@normalized_text')) > 0:
                    ann_tmp = annotation.query('normalized_text==@normalized_text')
                    rat_tmp = np.zeros([len(ann_tmp), len(ann_tmp.iloc[0]['rationale_numeric'])])
                    for ii in range(len(ann_tmp)):
                        rat_tmp[ii] = ann_tmp.iloc[ii]['rationale_numeric']
                    gold_rationale = (np.mean(rat_tmp, axis=0) > 0.5).astype(int)
                    words, word_importances = merge_subwords_words(xx['tokens'], ann_tmp.iloc[0]['sentence'].split(" "),
                                                                   xx['attention'],
                                                                   model_type=model_type,
                                                                   importance_aggregator=importance_aggregator_func)
                    if words[-1] == "." or words[-1] == '?':
                        words = words[:-1]
                        last_importance = word_importances[-2] + word_importances[-1]
                        word_importances = word_importances[:-1]
                        word_importances[-1] = last_importance

                    if len(words) == len(gold_rationale):
                        idx_sorted = np.argsort(word_importances)[::-1][:sum(gold_rationale)]
                        model_rationale = np.zeros(len(words), dtype=int)
                        model_rationale[idx_sorted] = 1

                        valid_samples += 1
                        # Append Rationales to Lists
                        y_true.extend(gold_rationale)
                        y_pred.extend(model_rationale)
                        y_pred_cont.extend(word_importances)
                    else:
                        merging_error += 1
                        continue
                        # print(f' \n error when merging {normalized_text}')

                    if dataset_name == 'sst2':
                        try:
                            text_id = ann_tmp.iloc[0]['originaldata_id']
                            gaze_tmp = df_gaze.query("text_id==@text_id").iloc[0]['relfix']
                            if len(gold_rationale) == len(gaze_tmp):
                                idx_sorted = np.argsort(gaze_tmp)[::-1][:sum(gold_rationale)]
                                gaze_rationale = np.zeros(len(words), dtype=int)
                                gaze_rationale[idx_sorted] = 1

                                valid_gaze_samples += 1
                                y_gaze.extend(gaze_rationale)
                                y_gaze_cont.extend(gaze_tmp)
                                y_true_gaze.extend(gold_rationale)
                                y_pred_gaze.extend(model_rationale)
                                y_pred_cont_gaze.extend(word_importances)

                                corr.append(spearmanr(gaze_tmp, word_importances)[0])
                        except IndexError:
                            pass


                else:
                    missing += 1
                    # print(f"couldn't find {normalized_text}")
                    continue

            from sklearn.metrics import f1_score, cohen_kappa_score

            print(f'Valid samples: {valid_samples}')
            print(mode, '\n')
            print('comparison model - annotations')
            print('f1:', f1_score(y_true, y_pred))
            print('cohen:', cohen_kappa_score(y_true, y_pred), '\n')

            df_results[dataset_name].loc[mode, 'annotations'] = np.around(cohen_kappa_score(y_true, y_pred), decimals=2)
            df_results[dataset_name].loc['annotations', mode] = np.around(cohen_kappa_score(y_true, y_pred), decimals=2)

            if dataset_name == 'sst2':
                print('comparison gaze - annotations')
                print('f1:', f1_score(y_gaze, y_pred_gaze))
                print('cohen:', cohen_kappa_score(y_gaze, y_pred_gaze), '\n')

                df_results[dataset_name].loc['gaze', mode] = np.around(cohen_kappa_score(y_gaze, y_pred_gaze),
                                                                       decimals=2)
                df_results[dataset_name].loc[mode, 'gaze'] = np.around(cohen_kappa_score(y_gaze, y_pred_gaze),
                                                                       decimals=2)

                print('comparison model - gaze')
                print('f1:', f1_score(y_true_gaze, y_gaze))
                print('cohen:', cohen_kappa_score(y_true_gaze, y_gaze), '\n')

                df_results[dataset_name].loc['annotations', 'gaze'] = np.around(cohen_kappa_score(y_true_gaze, y_gaze),
                                                                                decimals=2)
                df_results[dataset_name].loc['gaze', 'annotations'] = np.around(cohen_kappa_score(y_true_gaze, y_gaze),
                                                                                decimals=2)

                print('gaze cont vs model cont', spearmanr(y_gaze_cont, y_pred_cont_gaze))

            y_pred_old = y_pred if len(y_pred_old) == 0 else y_pred_old
            y_pred_cont_old = y_pred_cont if len(y_pred_cont_old) == 0 else y_pred_cont_old

        print('model cont vs model cont', spearmanr(y_pred_cont_old, y_pred_cont))

        df_results[dataset_name].loc['contrastive', 'non-contrastive'] = np.around(
            cohen_kappa_score(y_pred, y_pred_old),
            decimals=2)
        df_results[dataset_name].loc['non-contrastive', 'contrastive'] = np.around(
            cohen_kappa_score(y_pred, y_pred_old),
            decimals=2)

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[6, 4], sharey=True)
    #
    fig = plt.figure(figsize=[2.5, 1.5])
    vmin = np.nanmin(df_results['sst2'].values)
    df_plot = df_results['sst2'].fillna(0).loc[
        ['gaze', 'annotations', 'contrastive'], ['annotations', 'contrastive', 'non-contrastive']]
    mask = np.triu(np.ones_like(df_results['sst2'], dtype=bool))
    sns.heatmap(df_plot,
                annot=True,
                fmt='.2f',
                square=True,
                vmin=vmin,
                cmap='mako',
                cbar=False,
                mask=~mask[1:, 1:])
    plt.title("SST2 Cohen's Kappa")
    plt.yticks([0.5, 1.5, 2.5], ['gaze', 'annotations', 'contr.lrp'])
    plt.xticks([0.5, 1.5, 2.5], ['annotations', 'contr.lrp', 'non-contr.lrp'])

    plt.savefig('sst.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=[2, 1])
    # cbar_ax = fig.add_axes([.91, .3, .01, .4])
    df_plot = df_results['dynasent'].fillna(0).loc[
        ['annotations', 'contrastive'], ['contrastive', 'non-contrastive']]
    mask = np.triu(np.ones_like(df_results['dynasent'], dtype=bool))[:-1, :-2]
    sns.heatmap(df_plot,
                annot=True,
                fmt='.2f',
                square=True,
                vmin=vmin,
                cmap='mako',
                mask=~mask)
    plt.title("Dynasent Cohen's Kappa")
    plt.yticks([0.5, 1.5], ['annotations', 'contr.lrp'])
    plt.xticks([0.5, 1.5], ['contr.lrp', 'non-contr.lrp'])

    plt.savefig('dynasent.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=[6, 3], sharey=True)
    cbar_ax = fig.add_axes([.8, .2, .03, .6])

    # fig = plt.figure(figsize=[4, 3])
    vmin = np.nanmin(df_results['sst2'].values)
    df_plot = df_results['sst2'].fillna(0).loc[
        ['annotations', 'contrastive', 'non-contrastive'], ['contrastive', 'non-contrastive', 'gaze']]
    mask = np.triu(np.ones_like(df_results['sst2'], dtype=bool))
    sns.heatmap(df_plot,
                ax=axs[0],
                annot=True,
                fmt='.2f',
                square=True,
                vmin=vmin,
                cbar=False,
                cmap='mako',
                mask=~mask[1:, 1:])
    axs[0].set_title("SST2")
    axs[0].set_xticks([0.5, 1.5, 2.5], ['contr.lrp', 'non-contr.lrp', 'gaze'], rotation=30, ha='right')
    # plt.yticks([0.5, 1.5, 2.5], ['gaze', 'annotations', 'contr.lrp'])
    # plt.xticks([0.5, 1.5, 2.5], ['annotations', 'contr.lrp', 'non-contr.lrp'])

    # plt.savefig('sst.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # elif dataset_name == 'dynasent':
    #     fig = plt.figure(figsize=[3, 2])
    df_plot = df_results['dynasent'].fillna(0).loc[
        ['annotations', 'contrastive', 'non-contrastive'], ['contrastive', 'non-contrastive', 'gaze']]
    df_plot['gaze'] = np.nan
    mask = np.triu(np.ones_like(df_results['dynasent'], dtype=bool))[:, :-1]
    sns.heatmap(df_plot,
                ax=axs[1],
                annot=True,
                fmt='.2f',
                square=True,
                vmin=vmin,
                cbar_ax=cbar_ax,
                cmap='mako',
                mask=~mask)
    axs[1].set_title("Dynasent    ")
    axs[1].set_xticks([0.5, 1.5, 2.5], ['contr.lrp', 'non-contr.lrp', ""], rotation=30, ha='right')
    axs[1].set_yticks([0.5, 1.5, 2.5], ['annotations', 'contr.lrp', 'non-contr.lrp'])
    plt.suptitle("Cohen's Kappa")
    plt.savefig('sentiment45.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
