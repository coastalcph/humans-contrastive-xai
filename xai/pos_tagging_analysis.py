import re
import spacy
import json
import pickle
import numpy as np
import pandas as pd
from os.path import join
from collections import Counter, OrderedDict
from xai.xai_utils.annotations_utils import read_annotations, aggregate_annotations
from xai.xai_utils.tokenization_utils import merge_subwords_words
from datasets import load_dataset

def preview_rationales_2_print(rationales):
    """
    Preview rationales
    :param rationales:  List of rationales
    :return: List of rationales to preview
    """
    result = []
    annotation = False
    for idx, (key, rationale) in enumerate(rationales.items()):
        line = f'{idx + 1}: '
        for token, score in rationale:
            if score == 1 and annotation is False:
                line += '[ ' + token + ' '
                annotation = True
            elif score == 0 and annotation is True:
                line += '] ' + token + ' '
                annotation = False
            else:
                line += token + ' '
        result.append(line+'\n')
    return result


def analyze_annotator_rationales(gender_wordlist, class_names):
    nlp = spacy.load('en_core_web_sm')
    preview = []
    preview_tagging_rationales_pos = []
    labels = load_dataset('coastalcph/medical-bios', 'rationales', split='test')['label']

    for setting in ["standard", "contrastive"]:
        rationale_length = []
        annotations = read_annotations(setting, label_name='all')
        rationales = aggregate_annotations(setting, aggregation_method='majority')
        labels = {text: label for label, text in zip(labels, annotations)}

        examples = sorted(list(annotations.keys()))
        rationales = OrderedDict(sorted(rationales[0].items()))
        print(f"\nAnalysing {setting}. {len(examples)} examples.")

        docs = nlp.pipe(examples)
        tagging_pos = {}
        tagging_tag = {}
        for i, doc in enumerate(docs):
            tags_pos = {}
            tags_tag = {}
            assert len(doc) == len(list(rationales.values())[i])  # If an exception raised, we will need to normalize tokenizations
            for token in doc:
                tags_pos[token.text] = token.pos_
                tags_tag[token.text] = token.tag_
                # tags_.append({token.text: (token.pos_, token.tag_, token.dep_)})
            tagging_pos[doc.text] = tags_pos
            tagging_tag[doc.text] = tags_tag

        tagging_rationales_pos = {}
        gender = {k: [] for i, k in enumerate(class_names)}
        for example, rationale in rationales.items():
            tagging_ = tagging_pos[example]
            rationale_selection = list(filter(lambda x: x[1] == 1, rationale))
            # tagging_rationales_pos[example] = [tagging_[w[0]] for w in rationale_selection]
            tagging_rationales_pos[example] = []
            for w in rationale_selection:
                tagging_rationales_pos[example].append(tagging_[w[0]])
                text=w[0].lower()
                if text in list(gender_wordlist.keys()) and gender_wordlist[text] != 'n':
                    y_true = labels[example]
                    gender[y_true].append(gender_wordlist[text])
            rationale_length.append(len(rationale_selection))

        print(f"Average rationale length: {np.mean(rationale_length)}")
        # Flatten list to count on simple pos tagging (not taking into account the order)
        tagging_rationales_values = [w for sequence in tagging_rationales_pos.values() for w in sequence]
        print("POS: The simple UPOS part-of-speech tag.")
        c = Counter(tagging_rationales_values)
        print(
            [(i, np.round(c[i] / len(tagging_rationales_values) * 100.0, 2)) for i, count in c.most_common()]
        )
        for k, v in gender.items():
            print(k, Counter(gender[k]))

        tagging_rationales_tag = {}
        i=0
        for example, rationale in rationales.items():
            tagging_ = tagging_tag[example]
            rationale_selection = list(filter(lambda x: x[1] == 1, rationale))
            tagging_rationales_tag[example] = [tagging_[w[0]] for w in rationale_selection]
        # Flatten list to count on simple pos tagging (not taking into account the order)
        tagging_rationales_values = [w for sequence in tagging_rationales_tag.values() for w in sequence]
        print("Tag: The detailed part-of-speech tag.")
        c = Counter(tagging_rationales_values)
        print(
            [(i, np.round(c[i] / len(tagging_rationales_values) * 100.0, 2)) for i, count in c.most_common()]
        )
        preview.append(preview_rationales_2_print(rationales))
        preview_tagging_rationales_pos.append(tagging_rationales_pos)

    # with open("../data/rationales-biosbias/preview_biosbias_rationales.txt", "w") as f:
    #     for i in range(len(preview[0])):
    #         f.write(preview[0][i])
    #         f.write(", ".join(list(preview_tagging_rationales_pos[0].values())[i]) + "\n")
    #         f.write(preview[1][i])
    #         f.write(", ".join(list(preview_tagging_rationales_pos[1].values())[i]) + "\n")
    #         f.write("\n")


def analyze_model_rationales(root_dir_, gender_wordlist, dataset, xai_method,
                             correct_only, topk, class_names, examples_annotated_only):
    models = ["gpt2-small", "gpt2-base", "gpt2-large",
              "roberta-small", "roberta-base", "roberta-large",
              "t5-small", "t5-base", "t5-large"]
    print(dataset.upper(), ",", xai_method.upper())
    print("-"*100)
    nlp = spacy.load('en_core_web_sm')

    if examples_annotated_only:
        # Read annotations
        annotations = read_annotations("standard", label_name='all')
        annotations = [re.sub('[^a-z]', '', text.lower()) for text in list(annotations.keys())]

        rationales_ = aggregate_annotations("contrastive", aggregation_method='majority')
        rationales_length = OrderedDict()
        for text, v in sorted(rationales_[0].items()):
            rationales_length[re.sub('[^a-z]', '', text.lower())] = list(filter(lambda x: x[1] == 1, v))

    label2idx = {k: i for i, k in enumerate(class_names)}
    idx2label = {i: k for i, k in enumerate(class_names)}
    pos2df = []
    gender2df = []
    for modelname in models:
        print("\n" + modelname + "-" + dataset)
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
            continue

        relevance_dict['contrastive'] = relevance_dict['contrastive'].append(relevance_dict_val['contrastive'])
        relevance_dict['non-contrastive'] = relevance_dict['non-contrastive'].append(relevance_dict_val['non-contrastive'])
        examples_with_annotations = {}
        # current_label = label2idx[current_class] if current_class != 'all' else None
        for mode in ["non-contrastive", "contrastive"]:
            print(mode)
            subdf = relevance_dict[mode]

            for ix, xx in subdf.iterrows():
                if modelname.startswith("t5"):
                    ypred = label2idx[xx["y_pred"][0]]
                else:
                    ypred = xx["y_pred"].item()
                ytrue = xx["y_true"]
                if correct_only and ypred != ytrue:
                    continue
                topk_max_args = []

                # Merge Roberta Tokens into SpaCy Tokens
                if modelname.startswith('gpt2'):
                    model_type = 'gpt2'
                elif modelname.startswith('roberta'):
                    model_type = 'roberta'
                else:
                    model_type = 't5'
                doc = nlp(xx['data']['text'])

                spacy_tokens = [token.text for token in doc]
                spacy_pos = [token.pos_ for token in doc]
                words, word_importances = merge_subwords_words(xx['tokens'], spacy_tokens, xx['attention'],
                                                               model_type=model_type,
                                                               importance_aggregator=np.max)
                if len(words) != len(spacy_tokens):
                    continue
                if dataset == "biosbias" and examples_annotated_only:
                    # Check if we have annotations for this text
                    norm = re.sub('[^a-z]', '', xx['data']['text'].lower())
                    if norm in annotations:
                        if topk > 0:
                            topk_max_args = np.argpartition(word_importances, -topk)[-topk:]
                            examples_with_annotations[xx['data']['text']] = topk_max_args
                else:
                    if len(xx['attention']) > topk:
                        topk_max_args = np.argpartition(word_importances, -topk)[-topk:]
                        examples_with_annotations[xx['data']['text']] = topk_max_args

                # Look into gendered words
                for id_ in topk_max_args:
                    word = words[id_].lstrip('Ġ').lower()
                    if word in gender_wordlist and gender_wordlist[word] != 'n':
                        gender2df.append([xai_method, modelname, mode=='contrastive',
                                          idx2label[ytrue], word, gender_wordlist[word]])
                # Look into POS
                pos2df.append([xai_method, modelname, mode=='contrastive',
                               idx2label[ytrue],
                               [spacy_tokens[j] for j in topk_max_args],
                               [spacy_pos[j] for j in topk_max_args]])

        print(f"correct_only={correct_only}. {modelname}: {len(examples_with_annotations)} examples.")
    df = pd.DataFrame(data=gender2df,
                      columns=["xai", "model", "contrastive", "class", "word", "gender"])
    df.to_csv(f"../data/rationales-biosbias/analysis_gender_{xai_method}.csv", index=False)
    df = pd.DataFrame(data=pos2df,
                      columns=["xai", "model", "contrastive", "class", "words", "top5POS"])
    df.to_csv(f"../data/rationales-biosbias/analysis_pos_{xai_method}.csv", index=False)


if __name__ == "__main__":

    humans = False
    class_names = ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician']
    gender_wordlist_ = json.load(open("../data/gendered_words.json", "r"))
    gender_wordlist = {line["word"].lower(): line["gender"] for line in gender_wordlist_}
    if humans:
        analyze_annotator_rationales(gender_wordlist, class_names)
    else:
        root = "../results"
        dataset = ["biosbias", "dbpedia-animals"]
        xai_method = ["lrp", "gi", "gi_norm"]
        for m in xai_method:
            analyze_model_rationales(root, gender_wordlist, dataset[0], m, True, 1, class_names, False)
