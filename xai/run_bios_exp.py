
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk, load_dataset
import torch
import numpy as np
from xai.xai_roberta import RobertaForSequenceClassificationXAI
from xai.xai_utils import plot_conservation, plot_conservation_all, compute_lrp_explanation
from utils import set_up_dir, AUTH_TOKEN
import click
import torch.nn as nn
from xai.xai_utils import plot_sentence

from datasets import load_dataset
from train_models.data_helpers import load_biosbias_rationales
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle
import os

res_dir = '../results/17042023_bios'
set_up_dir(res_dir)
plotting = True
device = 'cuda'


# Load model, data and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("coastalcph/roberta-base-biosbias", use_auth_token=AUTH_TOKEN)
rationale_dataset = load_biosbias_rationales('roberta-base')
tokenizer = AutoTokenizer.from_pretrained('roberta-base')

model_xai = RobertaForSequenceClassificationXAI(model.config)
state_dict_ = model.state_dict()

_ = model_xai.load_state_dict(state_dict_)
_ = model.load_state_dict(state_dict_)


# Bypass embeddings
model_xai.roberta.embeddings_ = model.roberta.embeddings
model_xai.roberta.embeddings = nn.Identity()
model_components = {'embeddings': model_xai.roberta.embeddings_,
                    'encoder': model_xai.roberta.encoder, 
                    'classifier': model_xai.classifier}


class_names = ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician']
label2idx = {k:i for i,k in enumerate(class_names)}
idx2label = {v:k for k,v in label2idx.items()}

model_xai.eval()
model_xai.to(device)

model.eval()
model.to(device)


data = {'Rs': [],
        'As': [],
        'label': [],
        'logits': [],
        'words': []        
       }


ii=0
with PdfPages(os.path.join(res_dir, "plots.pdf")) as pdf:

    for X in rationale_dataset: 
        text = ' '.join(X['words'])
        y_true = X['label'] 
        inputs = tokenizer(text, return_tensors = 'pt').to(device)
        outputs = model(input_ids = inputs['input_ids'])

        logits = outputs['logits']
        logits_sort = torch.flip(logits.argsort(),dims=(1,)).squeeze()

        foil = logits_sort[1:].tolist() 

        print([idx2label[int(k)] for k in logits_sort])

        logits_ = logits.detach().cpu().numpy().squeeze()

        print('logits', logits_[y_true], [logits_[j] for j in foil])

        model_inputs = {'input_ids' : inputs['input_ids']}

        
        mask = torch.ones((1,len(class_names))).to(device)
        mask[:, foil[0]] = -1 
        mask[:, foil[1:]] = 0.
     
     #   mask[:, foil]=-1

        logit_funcs = {'true' : lambda x : x[:, logits_sort[0]] ,
                        'contrast':  lambda x :  (mask*x).sum()
                      }

        Rall = [] 
        strings  = {}
        
        # Extract explanation
        for i, case in enumerate(class_names):

            logit_func = lambda x : x[:, i] 
            strings[case] = case

            if case == idx2label[y_true]:
                c1 = strings[case]
                strings[case] += '*'

            if case == idx2label[foil[0]]:
                c2 = strings[case]
                strings[case] += ' (c)'

            relevance, selected_logit, logits = compute_lrp_explanation(model_components, 
                                                                        model_inputs, 
                                                                        logit_function = logit_func) 
            y_pred = np.argmax(logits.squeeze().detach().cpu().numpy())
            Rall.append(relevance)

        contrast_title = c1 + ' - ' + c2
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        words = [t.replace('Ġ','') for t in tokens]


        # Extra for GT
        rationale = np.array(X['tokenized_inputs']['input_ids_scores'])
        assert len(words) == len(rationale)


        if plotting:
            f, axs = plt.subplots(5+2,1, figsize=(22,6))
            plot_sentence(words, rationale, H0=150, W0=100, fax=(f, axs[0]))
            axs[0].set_title('ground truth')
             
        # Extra for contrast
        logit_func = logit_funcs['contrast']
        relevance, selected_logit, logits = compute_lrp_explanation(model_components, 
                                                                model_inputs, 
                                                                logit_function = logit_func) 
        Rall.append(relevance)

        if plotting: 
            # Plot explanations for all classes 
            for i, case in enumerate(class_names):
                relevance = Rall[i]
                R = relevance/np.max(np.abs(Rall))
                plot_sentence(words, R, H0=140, W0=110, fax=(f, axs[i+1]))
                axs[i+1].set_title(strings[case])

            # Plot contrast
            relevance = Rall[-1]
            R = relevance/np.max(np.abs(relevance))
            plot_sentence(words, R, H0=150, W0=100, fax=(f, axs[i+2]))
            axs[i+2].set_title(contrast_title)

            pdf.savefig(f)
            plt.close()

        # collect data
        data['Rs'].append(Rall)
        data['As'].append(rationale)
        data['label'].append(y_true)
        data['logits'].append(logits.detach().cpu().numpy().squeeze())
        data['words'].append(words)

        ii+=1
       
        
        print()
        
pickle.dump(data, open(os.path.join(res_dir, 'data.p'), 'wb'))
        
# Evaluation
from sklearn.metrics import roc_auc_score

T = []
T_contrast = []
for ii in range(len(data['Rs'])):
    labels = data['As'][ii]
    lrp  = data['Rs'][ii][data['label'][ii]]
    lrp_contrast  = data['Rs'][ii][-1]
    tokens = data['words'][ii]    
    
    try:
        T.append(roc_auc_score(labels, lrp))
        T_contrast.append(roc_auc_score(labels, lrp_contrast))
    except:
        print('Empty annotation', ii)
        

print('Target: {:0.2f} | Contrast: {:0.2f}'.format(np.mean(T)*100, np.mean(T_contrast)*100 ))

f,ax = plt.subplots(1,1)
ax.scatter(T, T_contrast)
ax.set_title('ROC-AUC score: explanation -> annotation')
ax.set_xlabel('AUC for standard lrp')
ax.set_ylabel('AUC for contrastive lrp')
f.savefig(os.path.join(res_dir, 'roc_auc.png'), dpi=200)
plt.close()



