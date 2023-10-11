import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def rat_num(len_sen, rationale):
    try:
        rationale = [el - 1 for el in list(map(int, rationale.split(",")))] if len(rationale.split(",")) > 1 else [
            int(float(rationale)) - 1]
        return [1 if idx in rationale else 0 for idx in range(len_sen)]
    except ValueError:
        import pdb;
        pdb.set_trace()


correlation_humans = []
correlation_rat_lrp = []
correlation_fix_lrp = []
correlation_fix_flow = []

annotation = pd.read_csv('./data/rationales-demographics/SST_annot_before_exclusions.csv') \
    .query("label==original_label").query("attentioncheck=='PASSED'")
annotation['len_sen'] = annotation['sentence'].apply(lambda x: len(x.split(" ")))
annotation['rationale_numeric'] = annotation.apply(lambda x: rat_num(x.len_sen, x.rationale_index), axis=1)

pad_token = '[PAD]'
sep_token = '[SEP]'
special_tokens = ['[PAD]', '[CLS]', '[SEP]', '</s>', '<s>', '<pad>']
modelname = 'mbert'

df_attention = pd.read_pickle('./data/SST_lrp/sst_importance.pkl')

df_attention = df_attention.set_index('text_id')
df_attention['rationale_numeric'] = None
df_attention['rationale_binary'] = None
df_attention['flow_accumulated'] = None
df_attention['rationale_ratio'] = None
df_attention['random'] = None

model_importance = ['attention_0', 'attention_5', 'attention_11', 'flow_11', 'lrp', 'relfix', 'random']

for id, subdf in annotation.groupby('originaldata_id'):
    if id not in df_attention.index:
        continue
    assert subdf.iloc[0]['len_sen'] == len(df_attention.loc[id, 'tokens'])

    subdf = subdf.query("original_label==label")
    df_attention.at[id, 'rationale_numeric'] = np.mean(subdf['rationale_numeric'].tolist(), axis=0)
    df_attention.at[id, 'rationale_binary'] = [1 if rat >= .5 else 0 for rat in
                                               df_attention.loc[id, 'rationale_numeric']]
    df_attention.at[id, 'random'] = np.random.rand(subdf.iloc[0]['len_sen'])
    correlation_humans.append(spearmanr(df_attention.loc[id, 'rationale_numeric'], df_attention.loc[id, 'relfix'])[0])
    correlation_rat_lrp.append(spearmanr(df_attention.loc[id, 'rationale_numeric'], df_attention.loc[id, 'lrp'])[0])
    correlation_fix_lrp.append(spearmanr(df_attention.loc[id, 'lrp'], df_attention.loc[id, 'relfix'])[0])
    correlation_fix_flow.append(spearmanr(df_attention.loc[id, 'flow_11'], df_attention.loc[id, 'relfix'])[0])

fig, ax = plt.subplots()
ax.bar([0, 1, 2, 3], [np.mean(correlation_humans), np.mean(correlation_rat_lrp),
                      np.mean(correlation_fix_lrp), np.mean(correlation_fix_flow)],
       )
ax.set_xticks([0, 1, 2, 3], labels=["gaze/annot", "lrp/annot", "lrp/gaze", "flow/gaze"])
ax.set_title("correlations")
plt.savefig("correlations", dpi=300)
