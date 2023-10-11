import os.path

from data import DATA_DIR
from datasets import load_dataset, concatenate_datasets
import re


def filter_out_sst2(original_dataset):
    # open Laura's rationales-demographics annotated dataset
    excl_dataset = load_dataset(os.path.join(DATA_DIR, 'rationales-demographics'),
                                data_files={"train": "SST_annot_before_exclusions.csv"}, cache_dir=None)
    doc_set = [re.sub(r'\s', '', sentence.lower()) for sentence in excl_dataset['train']['sentence']]

    # open Ander's rationales-demographics annotated dataset
    documents = []
    with open(os.path.join(DATA_DIR, 'rationales-multilingual', 'Multilingual_Interpretability_EN.csv'),
              'r', encoding='utf-8', errors='ignore') as input_file:
        for row in input_file:
            try:
                documents.append(re.sub(r'\s', '', row.split(';')[1].lower()))
            except:
                pass

    # filter out samples with identical sentences
    doc_set += documents
    doc_set = set(doc_set)
    modified_dataset = original_dataset.filter(lambda example: re.sub(r'\s', '', example["sentence"]) not in doc_set,
                                               load_from_cache_file=False)
    return modified_dataset


def filter_out_dynasent(original_dataset):
    # open Laura's rationales-demographics annotated dataset
    excl_dataset = load_dataset(os.path.join(DATA_DIR, 'rationales-demographics'),
                                data_files={"train": "dynasent_annot_before_exclusions.csv"}, cache_dir=None)
    doc_set = [re.sub(r'\s', '', sentence.lower()) for sentence in excl_dataset['train']['sentence']]

    # filter out samples with identical sentences
    doc_set = set(doc_set)
    modified_dataset = original_dataset.filter(lambda example: re.sub(r'\s', '', example["sentence"]) not in doc_set,
                                               load_from_cache_file=False)
    return modified_dataset


def fix_dynasent(original_dataset):
    LABELS = {'negative': 0, 'positive': 1}
    modified_dataset = original_dataset.filter(lambda example: example['gold_label'] != 'neutral',
                                               load_from_cache_file=False)
    modified_dataset = modified_dataset.map(
        lambda example: {'sentence': example['sentence'], 'label': LABELS[example['gold_label']]},
        load_from_cache_file=False)
    modified_dataset = modified_dataset.remove_columns(
        [column_name for column_name in modified_dataset.column_names if column_name not in ['sentence', 'label']])

    return modified_dataset


def get_char_scores(words, word_scores, sentence, tokenizer):
    inputs = tokenizer(sentence, return_offsets_mapping=True)
    # assign character-level rationale scores to enable use of offset mapping
    char_scores = ' '.join([''.join([str(score)] * len(word)) for word, score in zip(words, word_scores)])
    inputs = tokenizer(sentence, return_offsets_mapping=True)
    inputs['input_ids_scores'] = []
    _ = [inputs['input_ids_scores'].append(char_scores[off[0]:off[1]]) for off in inputs['offset_mapping']]
    # Proc character-level scores back to list of integers
    try:
        inputs['input_ids_scores'] = [0 if len(x) == 0 else int(x[0]) for x in inputs['input_ids_scores']]
    except:
        import pdb;
        pdb.set_trace()
    return inputs


def load_sst2_rationales(model_path: str, rationales_preview=False):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def align_words_rationales(example):
        words = example['sentence'].split()
        example['label'] = 1 if example['label'] == 'positive' else 0
        example['words'] = words
        rationale_indices = [int(index) for index in
                             example['rationale_index'].replace('[', '').replace(']', '').split(',')]
        example['word_scores'] = [1 if token_idx in rationale_indices else 0 for token_idx, token in enumerate(words)]

        example['input_ids'] = [sub_word for word in example['words'] for sub_word in tokenizer.tokenize(word)]
        #except AttributeError:
        #    import pdb;pdb.set_trace()
        example['rationale_ids'] = [token_score for token_score, word in zip(example['word_scores'], example['words'])
                                    for _ in tokenizer.tokenize(word)]
        example['tokenized_inputs'] = get_char_scores(words, example['word_scores'], example['sentence'], tokenizer)
        return example

    # open Laura's rationales-demographics annotated dataset
    dataset = load_dataset(os.path.join(DATA_DIR, 'rationales-demographics'),
                           data_files={"train": "SST_annot_after_processing.csv"}, cache_dir=None)['train']
    dataset = dataset.filter(lambda example: example['label'] != 'no sentiment', load_from_cache_file=False)

    dataset = dataset.remove_columns([column_name for column_name in dataset.column_names if
                                      column_name not in ['sentence', 'label', 'rationale_index', 'rationale']])
    dataset = dataset.map(align_words_rationales, load_from_cache_file=False)

    if rationales_preview:
        for sample in dataset:
            rationale = ' '.join([f'[{token}]' if token_score else token for token_score, token in
                                  zip(sample['word_scores'], sample['words'])])
            print(rationale.replace('] [', ' '))
            print(sample['rationale'])
            print(f'Label: {sample["label"]}')
            print('-' * 100)

    dataset = dataset.remove_columns([column_name for column_name in dataset.column_names if
                                      column_name not in ['sentence', 'label', 'words', 'word_scores', 'input_ids',
                                                          'rationale_ids', 'tokenized_inputs']])

    return dataset


def load_dynasent_rationales(model_path: str, rationales_preview=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def align_words_rationales(example):
        words = example['sentence'].split()
        sentence = ' '.join(words)  # harmonize sentences in comparison to example['sentence']
        example['label'] = 1 if example['label'] == 'positive' else 0
        example['words'] = words
        rationale_indices = [int(index) for index in
                             example['rationale_index'].replace('[', '').replace(']', '').split(',')]
        example['word_scores'] = [1 if token_idx in rationale_indices else 0 for token_idx, token in enumerate(words)]
        example['input_ids'] = [sub_word for word in example['words'] for sub_word in tokenizer.tokenize(word)]
        example['rationale_ids'] = [token_score for token_score, word in zip(example['word_scores'], example['words'])
                                    for _ in tokenizer.tokenize(word)]

        example['tokenized_inputs'] = get_char_scores(words, example['word_scores'], sentence, tokenizer)

        return example

    # open Laura's rationales-demographics annotated dataset
    dataset = load_dataset(os.path.join(DATA_DIR, 'rationales-demographics'),
                           data_files={"train": "dynasent_annot_after_processing.csv"}, cache_dir=None)['train']
    dataset = dataset.filter(lambda example: example['label'] != 'no sentiment', load_from_cache_file=False)

    dataset = dataset.remove_columns([column_name for column_name in dataset.column_names if
                                      column_name not in ['sentence', 'label', 'rationale_index', 'rationale']])
    dataset = dataset.map(align_words_rationales, load_from_cache_file=False)

    if rationales_preview:
        for sample in dataset:
            rationale = ' '.join([f'[{token}]' if token_score else token for token_score, token in
                                  zip(sample['word_scores'], sample['words'])])
            print(rationale.replace('] [', ' '))
            print(sample['rationale'])
            print(f'Label: {sample["label"]}')
            print('-' * 100)

    dataset = dataset.remove_columns([column_name for column_name in dataset.column_names if
                                      column_name not in ['sentence', 'label', 'words', 'word_scores', 'input_ids',
                                                          'rationale_ids', 'tokenized_inputs']])

    return dataset


def load_biosbias_rationales(model_path: str, rationales_preview=False):
    label_names = ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def extract_rationales(example):

        if example['spans'] is not None:
            tokens = [token['text'] for token in example['tokens']]
            tokens_scores = [0] * len(tokens)

            #  if 'abortion provider' in ' '.join(tokens):
            #      import pdb;pdb.set_trace()

            for span in example['spans']:
                if span['token_start'] == span['token_end']:
                    tokens_scores[span['token_start']] = 1
                else:
                    tokens_scores[span['token_start']:span['token_end']] = [1] * (
                        # span['token_start'] - span['token_end'])
                            span['token_end'] - span['token_start'] + 1)

            question_end = tokens.index("\n\n") + 2
            assert sum(tokens_scores[:question_end]) == 0.
            tokens = tokens[question_end + 1:]
            tokens_scores = tokens_scores[question_end + 1:]

            words = tokens
            sentence = ' '.join(words)

            example['words'] = tokens
            example['word_scores'] = tokens_scores
            example['input_ids'] = [sub_word for word in example['words'] for sub_word in tokenizer.tokenize(word)]

            example['rationale_ids'] = [token_score for token_score, word in
                                        zip(example['word_scores'], example['words']) for _ in tokenizer.tokenize(word)]
            example['tokenized_inputs'] = get_char_scores(words, example['word_scores'], sentence, tokenizer)

        else:
            example['words'] = []
            example['word_scores'] = []
            example['input_ids'] = []
            example['rationale_ids'] = []
            example['tokenized_inputs'] = {}

        example['label'] = label_names.index(example['title'])
        example['second_best'] = label_names.index(example['2nd best prediction'])

        return example

    dataset = \
    load_dataset('json', data_files=os.path.join(DATA_DIR, 'rationales-biosbias/bios_filtered_annotations.jsonl'),
                 cache_dir=None)['train']
    dataset = dataset.filter(lambda example: example['spans'] is not None, load_from_cache_file=False)
    dataset = dataset.map(extract_rationales, load_from_cache_file=False)
    dataset = dataset.remove_columns([column_name for column_name in dataset.column_names if
                                      column_name not in ['words', 'word_scores', 'input_ids', 'rationale_ids', 'label',
                                                          'second_best', 'tokenized_inputs']])
    dataset = dataset.filter(lambda example: len(example['words']) > 0, load_from_cache_file=False)

    if True:
        easy_dataset = \
        load_dataset('json', data_files=os.path.join(DATA_DIR, 'rationales-biosbias/bios_easy_annotations.jsonl'),
                     cache_dir=None)['train']
        easy_dataset = easy_dataset.map(extract_rationales, load_from_cache_file=False)
        easy_dataset = easy_dataset.remove_columns([column_name for column_name in easy_dataset.column_names if
                                                    column_name not in ['words', 'word_scores', 'input_ids',
                                                                        'rationale_ids', 'label', 'second_best',
                                                                        'tokenized_inputs']])
        easy_dataset = easy_dataset.filter(lambda example: len(example['words']) > 0, load_from_cache_file=False)

        dataset = concatenate_datasets([dataset, easy_dataset])

    if rationales_preview:
        for sample in dataset:
            rationale = ' '.join([f'[{token}]' if token_score else token for token_score, token in
                                  zip(sample['word_scores'], sample['words'])])
            print(rationale.replace('] [', ' '))
            print(f'Label: {sample["title"]}')
            print('-' * 100)

    return dataset


if __name__ == "__main__":
    # original_dataset = load_dataset('sst2', split="train")
    # modified_dataset = filter_out_sst2(original_dataset=original_dataset)
    # rationale_dataset = load_sst2_rationales('roberta-base', rationales_preview=True)
    #
    # original_dataset = load_dataset('dynabench/dynasent', 'dynabench.dynasent.r1.all', split="train")
    # original_dataset = fix_dynasent(original_dataset)
    # modified_dataset = filter_out_dynasent(original_dataset=original_dataset)
    # rationale_dataset = load_dynasent_rationales('roberta-base', rationales_preview=True)

    load_biosbias_rationales('roberta-base', rationales_preview=True)
