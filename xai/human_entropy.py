import os
import pandas as pd
import scipy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import scipy

from xai.prodigy_annotations_utils import read_annotations, aggregate_annotations, \
    EXCLUDED_ANNOTATORS, EXCLUDED__SCREEN_ANNOTATORS

STANDARD_FILENAME = "standard_biosbias_rationales"
CONTRASTIVE_FILENAME = "contrastive_biosbias_rationales"

annotations_standard, annotations_metadata_standard = read_annotations(STANDARD_FILENAME, label_name='all',
                                                     exclude_annotators=EXCLUDED_ANNOTATORS + EXCLUDED__SCREEN_ANNOTATORS)

annotations_contrastive, annotations_metadata_contrastive = read_annotations(CONTRASTIVE_FILENAME, label_name='all',
                                                     exclude_annotators=EXCLUDED_ANNOTATORS + EXCLUDED__SCREEN_ANNOTATORS)


rationales_standard,_ = aggregate_annotations(STANDARD_FILENAME, aggregation_method='majority',
                                   exclude_annotators=EXCLUDED_ANNOTATORS + EXCLUDED__SCREEN_ANNOTATORS)

rationales_contrastive,_ = aggregate_annotations(CONTRASTIVE_FILENAME, aggregation_method='majority',
                                   exclude_annotators=EXCLUDED_ANNOTATORS + EXCLUDED__SCREEN_ANNOTATORS)


keys = [re.sub('[^a-z]', '', text.lower()) for text in list(rationales_standard.keys())]

entropy_scores = {'human': {'contrastive': [], 'non-contrastive': [], 'max':[], 'min':[]}}



for  data, name in [(rationales_standard, 'non-contrastive'), (rationales_contrastive, 'contrastive')]:
    
    data = {re.sub('[^a-z]', '', text.lower()):v for text,v in data.items()}
    
    for k in keys:
        rs = data[k]
        scores = np.array([r[1] for r in rs])
        
        e = scipy.stats.entropy(scores)
        entropy_scores['human'][name].append(e)
        
        # Just compute min/max entrppy values once
        if name == 'contrastive':
            
            e_max = scipy.stats.entropy(np.ones_like(scores))
            e_min = scipy.stats.entropy(np.eye(len(scores))[0])
            
            entropy_scores['human']['max'].append(e_max)
            entropy_scores['human']['min'].append(e_min)
            

entropy_df = pd.DataFrame.from_dict(entropy_scores['human'])

ax = sns.violinplot(data=entropy_df[['contrastive', 'non-contrastive']], inner='point', cut=0)

entropy_min = np.mean(entropy_df['min'])
entropy_max = np.mean(entropy_df['max'])

ax.hlines(y=entropy_min,  xmin=-0.5, xmax=1.5, color='black', linestyle='--')
ax.hlines(y=entropy_max,  xmin=-0.5, xmax=1.5, color='black', linestyle='--')
ax.set_ylabel('entropy')
plt.show()

e_ttest = scipy.stats.ttest_ind([x for x in entropy_df['contrastive'] if np.isnan(x)==False],
                                [x for x in entropy_df['non-contrastive']if np.isnan(x)==False])
print(e_ttest)