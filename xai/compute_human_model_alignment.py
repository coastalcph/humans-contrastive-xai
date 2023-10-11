import pickle
import re

from utils import set_up_dir
import numpy as np
import click
from os.path import join
from tqdm import tqdm
from xai.xai_utils.tokenization_utils import merge_subwords_words
from xai.xai_utils.prodigy_annotations_utils import aggregate_annotations, EXCLUDED_ANNOTATORS

@click.command()
@click.option('--modelname', default="coastalcph/gpt2-small-biosbias", help='path to finetuned model on hf')
@click.option('--model_type', default="gpt2", help='Model type')
@click.option('--importance_aggregator', default="max", help='aggregation method (max, mean, sum)')
@click.option('--dataset_name', default="biosbias", help='name of dataset_name')
@click.option('--annotations_filename', default='standard_biosbias_rationales')
@click.option('--correct_only', is_flag=True)
@click.option('--results_dir', default='./results')
@click.option('--xai_method', default='lrp')
def main(modelname, model_type, importance_aggregator, dataset_name, annotations_filename, correct_only, results_dir, xai_method):

    res_dir = join(results_dir, 'human-model')
    set_up_dir(res_dir)

    # Load aggregated human annotations
    ANNOTATIONS, _ = aggregate_annotations(annotations_filename=annotations_filename, aggregation_method='majority',
                                           exclude_annotators=EXCLUDED_ANNOTATORS)

    # Normalize annotation names
    ANNOTATIONS = {re.sub('[^a-z]', '', key.lower()): value for key, value in ANNOTATIONS.items()}

    # Load dicts with importance scores
    relevance_dict = {}
    relevance_dict_val = {}
    if 'gpt2-biosbias' in modelname:
        modelbase = "-".join(modelname.split("/")[1].split("-")[:1])
    else:
        modelbase = "-".join(modelname.split("/")[1].split("-")[:2])
    relevance_dict['contrastive'] = pickle.load(
        open(join(results_dir, f'relevance_contrastive_{xai_method}_{modelbase}.pkl'), 'rb'))
    relevance_dict['non-contrastive'] = pickle.load(
        open(join(results_dir, f'relevance_true_{xai_method}_{modelbase}.pkl'), 'rb'))
    relevance_dict_val['contrastive'] = pickle.load(
        open(join(results_dir, f'relevance_validation_contrastive_{xai_method}_{modelbase}.pkl'), 'rb'))
    relevance_dict_val['non-contrastive'] = pickle.load(
        open(join(results_dir, f'relevance_validation_true_{xai_method}_{modelbase}.pkl'), 'rb'))

    relevance_dict['contrastive'] = relevance_dict['contrastive'].append(relevance_dict_val['contrastive'])
    relevance_dict['non-contrastive'] = relevance_dict['non-contrastive'].append(relevance_dict_val['non-contrastive'])


    # Importance aggregator
    if importance_aggregator == 'max':
        importance_aggregator_name = importance_aggregator
        importance_aggregator = np.max
    elif importance_aggregator == 'mean':
        importance_aggregator_name = importance_aggregator
        importance_aggregator = np.mean
    elif importance_aggregator == 'sum':
        importance_aggregator_name = importance_aggregator
        importance_aggregator = np.sum
    else:
        raise NotImplementedError

    results = {'f1': {key: {} for key in relevance_dict.keys()},
               'cohen': {key: {} for key in relevance_dict.keys()},
               'roc_auc_score': {key: {} for key in relevance_dict.keys()}
               }

    class_names = ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician']
    label2idx = {k: i for i, k in enumerate(class_names)}

    selected_classes = ['all', 'nurse', 'dentist']
    
    for mode in relevance_dict:
        # for xx in tqdm(range(len(relevance_dict[mode]['label'])), desc='iterating through datapoints'):
        for current_class in selected_classes:
            y_true = []
            y_pred = []
            valid_samples = 0
            current_label = label2idx[current_class] if current_class != 'all' else None
            subdf = relevance_dict[mode].query('y_true==@current_label') if current_class!='all' else relevance_dict[mode]
            for ix, xx in tqdm(subdf.iterrows(), desc='iterating through datapoints'):
                subwords = xx['tokens']
                ypred = xx["y_pred"]
                ytrue = xx["y_true"]
                if correct_only and ypred != ytrue:
                    continue

                importance = xx['attention']

                # Check if we have annotations for this text
                try:
                    normalized_text = re.sub('[^a-z]', '', xx['data']['text'].lower())
                    annotations = ANNOTATIONS[normalized_text]
                except KeyError:
                    continue

                # Normalize Gold SpaCy Tokens
                tokens = [token for token, score in annotations]
                # Annotated Human Rationales
                gold_rationale = np.asarray([score for tok, score in annotations])
                # Merge Roberta Tokens into SpaCy Tokens
                words, word_importances = merge_subwords_words(subwords, tokens, importance,
                                                               model_type=model_type,
                                                               importance_aggregator=importance_aggregator)
                # Take Top-K scores, K equals the nuber of annotated tokens
                idx_sorted = np.argsort(word_importances)[::-1][:sum(gold_rationale)]
                # Make a Binary Model Rationale
                model_rationale = np.zeros(len(words), dtype=int)
                model_rationale[idx_sorted] = 1
                if len(gold_rationale) != len(model_rationale):
                    continue
                valid_samples += 1
                # Append Rationales to Lists
                y_true.extend(gold_rationale)
                y_pred.extend(model_rationale)

            from sklearn.metrics import f1_score, cohen_kappa_score, roc_auc_score

            print(f'Valid samples: {valid_samples}')

            results['f1'][mode][current_class] = f1_score(y_true, y_pred)
            results['cohen'][mode][current_class] = cohen_kappa_score(y_true, y_pred)
            results['roc_auc_score'][mode][current_class] = roc_auc_score(y_true, y_pred)

    if 'contrastive' in annotations_filename:
        pickle.dump(results, open(join(res_dir, f'contrastive_f1_cohen_{xai_method}_{importance_aggregator_name}.pkl'), 'wb'))
    else:
        pickle.dump(results, open(join(res_dir, f'non-contrastive_f1_cohen_{xai_method}_{importance_aggregator_name}.pkl'), 'wb'))


if __name__ == "__main__":
    main()
