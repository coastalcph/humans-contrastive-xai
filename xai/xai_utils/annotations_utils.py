import json
import spacy
from datasets import load_dataset
# Load  SpaCy model
NLP_ENGINE = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
import numpy as np
from data import DATA_DIR
from .tokenization_utils import merge_subwords_words
EXCLUDED_ANNOTATORS = ['bios_batch_1_v2-12345', 'bios_batch_4-[prolific-id]', 'bios_batch_4-607717eaae6e81fa5a889d7f',
                       'bios_batch_4_repeated-5a8367b9190420000155ec2a', 'bios_batch_3-[612cecf6ebfcc62494f287eb]']
EXCLUDED__SCREEN_ANNOTATORS = ['5c85963b2421a3000142f159', '6131d01bc956974e057e9f72', '60fc0e57aed2db659b10be32',
                               '60ef2e52a188d9859f92bb58', 'id', 'ilias', '611d088b4a63306f79950cc9']


def read_annotations(setting: str, label_name: str='all'):
    """
    Read annotations from jsonl file
    :param annotations_filename: filename of JSONL annotations' file
    :param label_name: label name to filter annotations
    :param exclude_annotators: ids of annotators to exclude
    :return: Dict of rationales per example, Dict of metadata per example
    """
    annotations = {}
    dataset = load_dataset('coastalcph/medical-bios', 'rationales', split='test')
    label_id = dataset.features['label'].names.index(label_name) if label_name != 'all' else None
    if label_id is not None:
        dataset = dataset.filter(lambda example: example['label'] == label_id)
    for example in dataset:
        for idx in range(len(example['annotations'])):
            if example['text'] in annotations:
                annotations[example['text']][idx] = [(token, annotation) for token, annotation in zip(example['words'], example['annotations' if setting == 'standard' else 'contrastive_annotations'][idx])]
            else:
                annotations[example['text']] = {idx: [(token, annotation) for token, annotation in zip(example['words'], example['annotations' if setting == 'standard' else 'contrastive_annotations'][idx])]}
    return annotations


def aggregate_annotations(setting: str, aggregation_method='majority', label_name: str='all'):
    """
    Aggregate annotations from jsonl file
    :param setting: standard or contrastive
    :param aggregation_method: Method to aggregate annotations. Options: 'majority', 'mean', 'full_overlap', 'any'
    :param label_name: label name to filter annotations
    :return: List of aggregated rationales per example
    """
    dataset = load_dataset('coastalcph/medical-bios', 'rationales', split='test')
    label_id = dataset.features['label'].names.index(label_name) if label_name != 'all' else None
    if label_id is not None:
        dataset = dataset.filter(lambda example: example['label'] == label_id)
    aggregated_rationales = {}
    for document in dataset:
        key = document['text']
        aggregated_rationales[key] = []
        tokens = document['words']
        for annotation in document['annotations' if setting == 'standard' else 'contrastive_annotations']:
            aggregated_rationales[key].append(annotation)
        aggregated_rationales[key] = np.asarray(aggregated_rationales[key])
        if aggregation_method == 'majority':
            aggregated_rationales[key] = (np.mean(aggregated_rationales[key], axis=0) > 0.5).astype(int)
        elif aggregation_method == 'mean':
            aggregated_rationales[key] = np.mean(aggregated_rationales[key], axis=0)
        elif aggregation_method == 'any':
            aggregated_rationales[key] = (np.mean(aggregated_rationales[key], axis=0) > 0).astype(int)
        elif aggregation_method == 'full_overlap':
            aggregated_rationales[key] = (np.mean(aggregated_rationales[key], axis=0) == 1).astype(int)
        aggregated_rationales[key] = [(token, rationale) for token, rationale in zip(tokens, aggregated_rationales[key])]

    return aggregated_rationales


def preview_rationales(rationales):
    """
    Preview rationales
    :param rationales:  List of rationales
    :return: None
    """
    annotation = False
    for idx, (key, rationale) in enumerate(rationales.items()):
        print(f'Example {idx + 1}:', end=' ')
        for token, score in rationale:
            if score == 1 and annotation is False:
                print('[ ' + token, end=' ')
                annotation = True
            elif score == 0 and annotation is True:
                print('] ' + token, end=' ')
                annotation = False
            else:
                print(token, end=' ')
        print('\n' + '-' * 150)


def align_subwords_text(text, subwords, importance_scores, model_type='roberta', importance_aggregator=np.max):
    """
    Align sub-words and importance scores to raw text
    :param text: Raw text of example
    :param subwords: List of sub-words
    :param importance_scores: List Importance scores per sub-word
    :param model_type: Type of model used for tokenization, e.g., 'roberta', 'bert'
    :param importance_aggregator: Aggregation method for importance scores. Options: np.max, np.mean, np.sum
    :return:
    """
    # Tokenize text with SpaCy
    tokens = [token.text for token in NLP_ENGINE(text)]
    # Align sub-words to tokens
    adjusted_tokens, adjusted_importance = \
        merge_subwords_words(subwords, tokens, importance_scores,
                             model_type=model_type, importance_aggregator=importance_aggregator)

    return adjusted_tokens, adjusted_importance


if __name__ == '__main__':
    # Load human rationales
    annotations, annotations_metadata = read_annotations('standard', label_name='all')

    # Aggregate human rationales
    rationales = aggregate_annotations('standard', aggregation_method='majority')
    # Preview rationales
    preview_rationales(rationales)

    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset('coastalcph/medical-bios', 'rationales', split='test')

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    count_broken = 0
    # Preview sub-words to words alignment
    for idx, example in enumerate(dataset):
        subwords = tokenizer.tokenize(example['text'])
        fake_importance_scores = [1] * len(subwords)
        adjusted_tokens, adjusted_importance = align_subwords_text(example['text'], subwords, fake_importance_scores,
                                                                   model_type='roberta',
                                                                   importance_aggregator=np.max)
        if sum([1 if len(token) > 30 else 0 for token in adjusted_tokens]) != 0:
            count_broken += 1
            print(f'Example {idx + 1}: {example["text"]}')
            print(f'Example {idx + 1}: ', adjusted_tokens)
            print('-' * 150)
    print(count_broken)


def get_human_label_foil_lookup():    
    import re
    
    _, annotations_metadata_standard = read_annotations("standard", label_name='all')

    _, annotations_metadata_contrastive = read_annotations("contrastive", label_name='all')
        
    label_foil_lookup = {}
    
    for key in annotations_metadata_contrastive.keys():
        
        assert annotations_metadata_standard[key]['label'] == annotations_metadata_contrastive[key]['label']
        label = annotations_metadata_contrastive[key]['label']
        foil = annotations_metadata_contrastive[key]['foil']
        
        key_normalized = re.sub('[^a-z]', '', key.lower())
        label_foil_lookup[key_normalized] = {'label':label, 'foil':foil}
        
    return label_foil_lookup
