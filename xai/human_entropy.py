import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import scipy

from xai.xai_utils.annotations_utils import read_annotations, aggregate_annotations

annotations_standard = read_annotations("standard", label_name='all')

annotations_contrastive = read_annotations("contrastive", label_name='all')


rationales_standard = aggregate_annotations("standard", aggregation_method='majority')

rationales_contrastive = aggregate_annotations("contrastive", aggregation_method='majority')


keys = [re.sub('[^a-z]', '', text.lower()) for text in list(rationales_standard.keys())]

entropy_scores = {'human': {'contrastive': [], 'non-contrastive': [], 'max':[], 'min':[]}}


for  data, name in [(rationales_standard, 'non-contrastive'), (rationales_contrastive, 'contrastive')]:
    
    data = {re.sub('[^a-z]', '', text.lower()): v for text, v in data.items()}
    
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

e_ttest = scipy.stats.ttest_ind([x for x in entropy_df['contrastive'] if np.isnan(x) == False],
                                [x for x in entropy_df['non-contrastive']if np.isnan(x) == False])
print(e_ttest)