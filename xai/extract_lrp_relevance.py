from os.path import join, isfile
from tqdm import tqdm
import click

import pandas as pd
import torch
from torch import nn, Tensor
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from xai.xai_roberta import RobertaForSequenceClassificationXAI
from xai.xai_gpt2 import GPT2ForSequenceClassificationXAI
from xai.xai_t5 import T5ForConditionalGenerationXAI

from xai.xai_utils import plot_conservation, compute_lrp_explanation
from xai.prodigy_annotations_utils import get_human_label_foil_lookup

from train_models.data_helpers import filter_out_sst2, load_sst2_rationales, filter_out_dynasent, fix_dynasent, \
    load_dynasent_rationales
from utils import set_up_dir, AUTH_TOKEN
from typing import Any
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

class Zero(nn.Identity):
    """A layer that just returns Zero-Embeddings"""

    def __init__(self, dim=768, *args: Any, **kwargs: Any) -> None:
        self.dim = dim
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.zeros((input.shape[0], input.shape[1], self.dim)).to(input.device)


# python extract_lrp_relevance.py --model coastalcph/roberta-base-sst2 --dataset sst2 --lang en --case lrp --mode true
# python extract_lrp_relevance.py --modelname michelecafagna26/t5-base-finetuned-sst2-sentiment --dataset_name sst2 --lang en --case lrp --mode true

# python xai/extract_lrp_relevance.py --modelname coastalcph/t5-base-biosbias --dataset_name biosbias  --case lrp --mode true

#python xai/extract_lrp_relevance.py --modelname coastalcph/roberta-base-biosbias --dataset_name biosbias  --case lrp --mode true --human_labels True

#python xai/extract_lrp_relevance.py --modelname coastalcph/gpt2-large-biosbias --dataset_name biosbias  --case lrp --mode true --human_labels False


@click.command()
@click.option('--modelname', default='coastalcph/roberta-base-sst2')
@click.option('--dataset_name', default='sst2')
@click.option('--dataset_split', default='test', help='split (validation, test)')
@click.option('--res_folder', default='./results')
@click.option('--case', default='lrp')
@click.option('--mode', default='true')
@click.option('--human_labels', default=False)
def main(modelname, dataset_name, dataset_split, res_folder, case, mode, human_labels):
    if 'gpt2-small' in modelname:
        model_case = 'gpt2-small'
        model_class = GPT2ForSequenceClassificationXAI
    elif 'gpt2-large' in modelname:
        model_case = 'gpt2-large'
        model_class = GPT2ForSequenceClassificationXAI
    elif 'gpt2-base' in modelname:
        model_case = 'gpt2-base'
        model_class = GPT2ForSequenceClassificationXAI
    elif 'roberta-small' in modelname:
        model_case = 'roberta-small'
        model_class = RobertaForSequenceClassificationXAI
    elif 'roberta-base' in modelname:
        model_case = 'roberta-base'
        model_class = RobertaForSequenceClassificationXAI
    elif 'roberta-large' in modelname:
        model_case = 'roberta-large'
        model_class = RobertaForSequenceClassificationXAI
    elif 't5-small' in modelname:
        model_case = 't5-small'
        model_class = T5ForConditionalGenerationXAI
    elif 't5-base' in modelname:
        model_case = 't5-base'
        model_class = T5ForConditionalGenerationXAI
    elif 't5-large' in modelname:
        model_case = 't5-large'
        model_class = T5ForConditionalGenerationXAI
    else:
        raise NotImplementedError

    filename_out = f"relevance_{dataset_split}_{mode}_{case}_{model_case}.pkl" \
        if dataset_split != 'test' \
        else f"relevance_{mode}_{case}_{model_case}.pkl"

    filename_out = filename_out.replace('.pkl', '_annotated_only.pkl') if human_labels else filename_out
    
    
    res_folder = join(res_folder, modelname.split('/')[1])
    set_up_dir(res_folder)

    print(join(res_folder, filename_out))

    if isfile(join(res_folder, filename_out)):
        print(f'{filename_out} already exists')
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if 't5' in modelname:
            model = T5ForConditionalGeneration.from_pretrained(modelname, use_auth_token=AUTH_TOKEN)
            tokenizer = T5Tokenizer.from_pretrained(modelname, use_auth_token=AUTH_TOKEN)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(modelname, use_auth_token=AUTH_TOKEN)
            tokenizer = AutoTokenizer.from_pretrained(modelname, use_auth_token=AUTH_TOKEN)

        # Load explainable model

        if case == 'lrp':
            # Load lrp model with modified forward pass
            model_xai = model_class(model.config, lrp = True)
            return_gradient_norm = False
        elif case == 'lrp_norm':
            model_xai = model_class(model.config, lrp = True)
            return_gradient_norm = True
        elif case == 'gi':
            if 't5' in model_case:
                model_xai = model_class(model.config, lrp = False)
            else:    
                # Copy model
                model_xai = AutoModelForSequenceClassification.from_pretrained(modelname, use_auth_token=AUTH_TOKEN)
            return_gradient_norm = False
        elif case == 'gi_norm':
            if 't5' in model_case:
                model_xai = model_class(model.config, lrp = False)
            else:  
                model_xai = AutoModelForSequenceClassification.from_pretrained(modelname, use_auth_token=AUTH_TOKEN)
            return_gradient_norm = True

        state_dict_ = model.state_dict()

        # Set biases to zero to sanity check that conservation is preserved
        if False:
            keys = list(state_dict_.keys())
            for k, v in state_dict_.items():
                if '.bias' in k or '_bias' in k:
                    state_dict_[k] = 0. * v
                    print(k)

        _ = model_xai.load_state_dict(state_dict_)
        _ = model.load_state_dict(state_dict_)

        model_xai.eval()
        model_xai.to(device)

        model.eval()
        model.to(device)

        # check outputs are closeby
        # inputs_ids_test = torch.tensor(np.array([1, 2, 3, 4, 100])).to(device).unsqueeze(0)
        # outs1 = model(inputs_ids_test)
        # o1 = outs1.logits.detach().cpu().numpy().squeeze()
        # o2 = model_xai(inputs_ids_test)['logits'].squeeze().detach().cpu().numpy()
        # assert np.allclose(o1,o2, rtol=1e-04, atol=1e-05) == True

        if dataset_name == 'sst2':
            original_dataset = load_dataset('sst2', split="train")
            modified_dataset = filter_out_sst2(original_dataset=original_dataset)
            rationale_dataset = load_sst2_rationales(model_case, rationales_preview=False)
            labels = list(set(rationale_dataset['label']))  # "negative" (0) or positive (1).
            key = 'sentence'

        elif dataset_name == 'dynasent':
            original_dataset = load_dataset('dynabench/dynasent', 'dynabench.dynasent.r1.all', split="train")
            original_dataset = fix_dynasent(original_dataset)
            modified_dataset = filter_out_dynasent(original_dataset=original_dataset)
            rationale_dataset = load_dynasent_rationales(model_case, rationales_preview=False)
            labels = list(set(rationale_dataset['label']))  # "negative" (0) or positive (1).
            key = 'sentence'

        elif dataset_name == 'biosbias':
            original_dataset = load_dataset("coastalcph/xai_fairness_benchmark", dataset_name,
                                            use_auth_token=AUTH_TOKEN)
            rationale_dataset = original_dataset[dataset_split]
            labels = list(set(rationale_dataset['label']))  # "negative" (0) or positive (1).
            key = 'text'

        elif dataset_name == 'dbpedia-animals':
            original_dataset = load_dataset('coastalcph/dbpedia-datasets', 'animals', use_auth_token=AUTH_TOKEN)
            rationale_dataset = original_dataset['test']
            labels = list(set(rationale_dataset['label']))  # "negative" (0) or positive (1).
            key = 'text'

        if 't5' in modelname:
            if dataset_name == 'biosbias':
                labels_names = ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician']
                label2idx = {k: v for k, v in enumerate(labels_names)}
                label2id = {k: tokenizer(k).input_ids[:-1] for k in labels_names}
                generation_ids_dict = {k: v[0] if len(v) == 1 else v for k, v in label2id.items()}

            elif dataset_name == 'dbpedia-animals':
                #labels_names = ['Amphibian', 'Arachnid', 'Bird', 'Crustacean', 'Fish', 'Insect', 'Mollusca', 'Reptile']
                labels_names = ['<extra_id_{}>'.format(i) for i in range(8)]
                label2idx = {k:v for k,v in enumerate(labels_names)}
                label2id = {k:tokenizer(k).input_ids[:-1] for k in labels_names}
                generation_ids_dict = {k:v[0] if len(v)==1 else v for k,v in label2id.items()}

        else:
            generation_ids_dict = None
            
            
        # Get label and foil from human annotations
        label_foil_lookup = get_human_label_foil_lookup() if human_labels else None

        if human_labels:
            assert dataset_name == 'biosbias'
            labels_names = ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician']
            labelname2idx = {v:k for k, v in enumerate(labels_names)}
        
        
        
        def preprocess_function(examples, padding, examples_key):
            # Tokenize the texts
            # Padding and truncation switched off
            batch = tokenizer(
                examples[examples_key],
                padding=padding,
                max_length=128,
                truncation=True,
            )
            return batch

        tokenized_dataset = rationale_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=None,
            load_from_cache_file=False,
            fn_kwargs={"padding": False,
                       "examples_key": key
                       },
        )

        if mode == 'contrastive':
            df = pd.DataFrame(
                columns=['id', 'tokens', 'attention', 'y_pred', 'y_true', 'logits', 'data', 'foil'])
        else:
            df = pd.DataFrame(columns=['id', 'tokens', 'attention', 'y_pred', 'y_true', 'logits', 'data'])

        Rs = []
        Ls = []

        # Bypass embeddings
        if 'gpt2' in model_case:
            token_embeddings = model_xai.transformer.wte
            # bypass position embeddings
            position_embeddings = model_xai.transformer.wpe
            model_xai.transformer.wpe = Zero(model.config.hidden_size)

            model_components = {'embeddings': token_embeddings,
                                'wpe': position_embeddings,
                                'encoder': model_xai.transformer,
                                'classifier': model_xai.score}

        elif 'roberta' in model_case:
            token_embeddings = model_xai.roberta.embeddings
            model_xai.roberta.embeddings = nn.Identity()

            model_components = {'embeddings': token_embeddings,
                                'encoder': model_xai.roberta.encoder,
                                'classifier': model_xai.classifier}

        elif 't5' in model_case:
            # Handled inside the model
            model_components = {'model': model_xai}

        for ii in tqdm(range(len(tokenized_dataset))):

            data = tokenized_dataset[ii] \
                if dataset_name in ['biosbias', 'dbpedia-animals'] else tokenized_dataset[ii]['tokenized_inputs']

            # Input sentence without padding
            if False:
                attention_mask = torch.tensor(data['attention_mask'])
                inputs_ = torch.tensor(data['input_ids'])[attention_mask == 1].unsqueeze(0)
                tokens = tokenizer.convert_ids_to_tokens(inputs_.squeeze())

            inputs_ = data['input_ids']
            tokens = tokenizer.convert_ids_to_tokens(inputs_)
            inputs_ = torch.tensor(inputs_).unsqueeze(0)

            # assert len(data['input_ids']) == len(data['input_ids_scores'])

            example_id = ii
            y_true = tokenized_dataset[ii]['label']

            if human_labels:
                text_normalized = re.sub('[^a-z]', '', data['text'].lower())                
                if text_normalized  in label_foil_lookup:
                    human_label = label_foil_lookup[re.sub('[^a-z]', '', data['text'].lower())]['label']
                    human_foil = label_foil_lookup[re.sub('[^a-z]', '', data['text'].lower())]['foil']
                    human_label_dict = {'foil': human_foil, 'label': human_label} 
                    
                    # Overwrite y_true to human label
                    y_true = labelname2idx[human_label_dict['label']]
                    foil = labelname2idx[human_label_dict['foil']]

                    
                else:
                    print('Skip, text was not annotated:', example_id)
                    continue
            else:
                human_label_dict = None
                
            
            
            if 't5' in model_case:
                if False:
                    extra_input_ids = tokenizer("sentiment: ", return_tensors="pt")
                    inputs_conditional = torch.concat([extra_input_ids.input_ids[:, :-1], inputs_], 1).to(device)

                else:
                    inputs_conditional = inputs_.to(device)

                preds = model.generate(inputs_conditional)
                decoded_preds = tokenizer.batch_decode(sequences=preds, skip_special_tokens=True)
                y_pred = decoded_preds
            else:
                try:
                    output = model(input_ids=inputs_.to(device), output_attentions=False)
                except:
                    print(f'skip datapoint {ii}: {tokens}')
                    continue
                logits = output['logits'].squeeze().detach().cpu()
                logits_sort = torch.flip(logits.argsort(), dims=(0,)).squeeze()
                assert np.argmax(logits.numpy()) == logits_sort[0]
                y_pred = logits_sort[0]

            model_inputs = {'input_ids': inputs_.to(device)}

            
        
            # Extract explanation for target
            if 't5' in model_case:
                # Handle logit_func in generate function (-> xai/xai_t5.py)
                logit_func = None
                model_xai.xai_generation = {'label2idx': label2idx,
                                            'generation_ids_dict': generation_ids_dict,
                                            'mode': mode,
                                            'human_labels': human_label_dict
                                            }
            else:
                if mode == 'true':
                    # Explain true label
                    if human_labels:
                        logit_func =  lambda x: x[:,y_true]
                    else:
                        logit_func =lambda x: x[:, y_pred] 

                 #   import pdb;pdb.set_trace()

                elif mode == 'foil':
                    # Explain true label
                    if not human_labels:
                        foil = logits_sort[1] if y_pred == y_true else logits_sort[0]
                       
                    logit_func = lambda x: x[:, foil]

                elif mode == 'contrastive':
                    # Explain contrast of "label - foil" (foil here defined as all other labels)
                    # Define foil in label space
                    if dataset_name in ['sst2', 'dynasent']:
                        assert human_labels==False # not defined for these datasets
                        foil = [y for y in labels if y is not int(y_pred)]
                        logit_func = lambda x: x[:, y_pred] - sum([x[:, k] for k in foil])
                    else:
                        
                        if human_labels:
                            mask = torch.zeros((1, len(labels))).to(device)
                            mask[:, foil] = -1.
                            mask[:, y_true] = 1.
                            logit_func = lambda x: (mask * x).sum()
                        else:
                            foil = logits_sort[1]
                            mask = torch.ones((1, len(labels))).to(device)
                            mask[:, foil] = -1.
                            mask[:, logits_sort[2:]] = 0.
                            logit_func = lambda x: (mask * x).sum()

            try:
                relevance, selected_logit, logits = compute_lrp_explanation(model_components,
                                                                            model_inputs,
                                                                            logit_function=logit_func,
                                                                            model_case=model_case,
                                                                            return_gradient_norm=return_gradient_norm,
                                                                            tokenizer=tokenizer,
                                                                            generation_ids_dict=generation_ids_dict)

            except AssertionError:
                print(f'skip datapoint {ii}: {tokens}')
                continue

            if 't5' in model_case and mode in ['contrastive', 'foil']:
                foil = model_xai.foil

            # if mode == 'contrastive':
            #     R_all = []
            #     for y_ in range(len(labels)):
            #         relevance_, _, _ = compute_lrp_explanation(model_components,
            #                                                    model_inputs,
            #                                                    logit_function=lambda x: x[:, y_],
            #                                                    model_case=model_case,
            #                                                    return_gradient_norm=return_gradient_norm)
            #         R_all.append(relevance_)
            #     R_all = np.array(R_all)

            if False:
                # just to check if models give same outputs
                print(model(**model_inputs).logits)
                print(logits)
                print()

            Ls.append(selected_logit)
            Rs.append(relevance.sum())

            if mode == 'contrastive':
                df.loc[ii] = [example_id, tokens, relevance, y_pred, y_true, logits, data, foil]  #
            elif mode == 'foil':
                df.loc[ii] = [example_id, tokens, relevance, y_pred, foil, logits, data]
            else:
                df.loc[ii] = [example_id, tokens, relevance, y_pred, y_true, logits, data]

        df.to_pickle(join(res_folder, filename_out))

        filename_conservation = "conservation_{}_{}_{}.png".format(mode, case, model_case)
        filename_conservation = filename_conservation.replace('.png', '_annotated_only.png') if human_labels else filename_conservation

        plot_conservation(Ls, Rs, join(res_folder, filename_conservation))


if __name__ == '__main__':
    main()
