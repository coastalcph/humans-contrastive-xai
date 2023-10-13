from os.path import join
import click
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


@click.command()
@click.option('--modelname', default='coastalcph/roberta-base-sst2')
@click.option('--dataset_name', default='sst2')
@click.option('--results_dir', default='./results')
@click.option('--xai_method', default='lrp')
def main(modelname, dataset_name, results_dir, xai_method):
    print(results_dir)
    try:
        # model-model evaluation
        correlation = pickle.load(open(join(results_dir, f'model-model/correlation_{xai_method}.pkl'), 'rb'))
        entropy = pickle.load(open(join(results_dir, f'model-model/entropy_{xai_method}.pkl'), 'rb'))
        mse = pickle.load(open(join(results_dir, f'model-model/mse_{xai_method}.pkl'), 'rb'))

        df = pd.DataFrame(columns=['correlation', 'entropy (non-contrastive)', 'entropy (contrastive)', 'mse'])
        df.loc[0] = [np.mean(correlation), np.mean(entropy['true']), np.mean(entropy['contrastive']), np.mean(mse)]

        print(f'correlation: {np.around(np.mean(correlation), decimals=2)}')
        print(f'mse: {np.around(np.mean(mse), decimals=2)}')

        with open(join(results_dir, f'model-model/metrics_{xai_method}.txt'), 'w') as f:
            f.write('correlation')
            f.write(str(np.around(np.mean(correlation), decimals=2)))
            f.write('\n')
            f.write('mse')
            f.write(str(np.around(np.mean(mse), decimals=2)))
            f.write('\n\n')

    except FileNotFoundError:
        print('model-model evaluation files not found - run xai_comparison first')

    try:
        # human-human evaluation
        for ii, aggregation_method in enumerate(['majority', 'full_overlap', 'any']):
            labels, f1_scores, cohen_kappas, fleiss_kappas = \
                pickle.load(open(join(results_dir, f'human-human/{aggregation_method}_scores.pkl'), 'rb'))

            # label support
            label_support = {
                "psychologist": 2200,
                "surgeon": 1280,
                "nurse": 1638,
                "dentist": 1533,
                "physician": 1349,
                "all": 7800
            }

            cohen_kappa_contrastive = pickle.load(open("./results/contrastive_biosbias_rationales_scores.pkl", "rb"))
            cohen_kappa_standard = pickle.load(open("./results/standard_biosbias_rationales_scores.pkl", "rb"))

            df = pd.DataFrame(columns=labels)
            df.loc['Label support'] = [int(i) for i in label_support.values()]
            df.loc['macro F1'] = [0.93, 0.88, 0.85, 0.98, 0.80, 0.89]
            # df.loc['F1'] = [float(i) for i in f1_scores.values()]
            df.loc["Across settings"] = [float(i) for i in cohen_kappas.values()]
            df.loc["Fleiss kappa"] = [float(i) for i in fleiss_kappas.values()]
            df.loc["Contrastive"] = [float(cohen_kappa_contrastive[label]) for label in labels]
            df.loc["Non-contrastive"] = [float(cohen_kappa_standard[label]) for label in labels]

            try:
                fig, ax = plt.subplots(figsize=[7, 6.5], ncols=1, nrows=2, gridspec_kw={'height_ratios': [3, 2]},
                                       sharex=True)
                cbar_ax = fig.add_axes([.91, .02, .02, .25])
                vmin = df.loc[["Non-contrastive", "Contrastive", "Across settings"]].values.min()
                sns.heatmap(df.loc[["Non-contrastive", "Contrastive", "Across settings"]],
                            cmap='mako', annot=True, vmin=vmin, vmax=1, fmt='g', square=True,
                            cbar_ax=cbar_ax, ax=ax[0])

                sns.heatmap(
                    df.loc[['macro F1', "Label support"]],
                    cmap='mako', annot=True, vmin=vmin, vmax=1, fmt='g', square=True,
                    cbar_ax=cbar_ax, ax=ax[1])

                ax[0].set_title('Inter-annotator agreement')
                ax[1].set_title('RoBERTa large')
                ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=0)
                fig.savefig(join(results_dir, f'human-human/{aggregation_method}.png'), dpi=300, bbox_inches='tight')
                plt.close()

            except TypeError:
                print('error')
                pass
            print(aggregation_method)
            print(df.to_string())

            with open(join(results_dir, f'human-human/metrics_{aggregation_method}.txt'), 'w') as f:
                f.write(aggregation_method)
                f.write('\n')
                f.write(df.to_string())
                f.write('\n')
                try:
                    f.write(df.style.to_latex())
                except AttributeError:
                    f.write(df.to_latex(float_format="%.2f"))
                f.write('\n\n')

    except FileNotFoundError:
        print('human-human evaluation files not found - run xai_comparison first')

    try:
        for ii, importance_aggregator in enumerate(['max']):
            scores_contrastive = pickle.load(
                open(join(results_dir, f'human-model/contrastive_f1_cohen_{xai_method}_{importance_aggregator}.pkl'),
                     'rb'))

            scores_noncontrastive = pickle.load(
                open(
                    join(results_dir, f'human-model/non-contrastive_f1_cohen_{xai_method}_{importance_aggregator}.pkl'),
                    'rb'))

            metric = 'cohen'
            selected_classes = ['physician', 'dentist', 'all']
            df = {}
            fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=[12, 4])

            for current_class in selected_classes:
                df[current_class] = pd.DataFrame(columns=['model_contr', 'model_non-contr'])
                df[current_class].loc['hum_contr'] = [scores_contrastive[metric]['contrastive'][current_class],
                                                      scores_contrastive[metric]['non-contrastive'][current_class]]
                df[current_class].loc['hum_non-contr'] = [scores_noncontrastive[metric]['contrastive'][current_class],
                                                          scores_noncontrastive[metric]['non-contrastive'][
                                                              current_class]]

            for ii, current_class in enumerate(selected_classes):
                # if metric in ['f1', 'cohen']:
                sns.heatmap(df[current_class], ax=axs[ii], vmin=0.1, vmax=0.55, annot=True,
                            cmap='mako',
                            square=True)

                axs[ii].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                axs[ii].set_title(current_class)

            with open(join(results_dir, f'human-model/metrics_{xai_method}.txt'), 'w') as f:
                for current_class in df:
                    df_string = df[current_class].round(2).to_string()
                    f.write(metric)
                    f.write('\n')
                    f.write(df_string)
                    f.write('\n\n')
                    try:
                        f.write(df[current_class].round(2).style.to_latex())
                    except AttributeError:
                        f.write(df[metric][current_class].round(2).to_latex())
                    f.write('\n\n')


            plt.savefig(join(results_dir, f'human-model/{importance_aggregator}_{xai_method}.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()

    except FileNotFoundError:
        print('human-model evaluation files not found - run compute_human_model_alignment.py first')


if __name__ == '__main__':
    main()
