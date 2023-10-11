import pandas as pd
import numpy as np
from tokenization_utils import merge_hyphens, merge_subwords, merge_symbols
import sys

sys.path.append('/Users/wnd306/OneDrive - University of Copenhagen/KU/rationales-eyetracking/task_gaze_transformers')
from eval_utils import get_human_avg_df

sys.path.append(
    '/Users/wnd306/OneDrive - University of Copenhagen/KU/rationales-eyetracking/task_gaze_transformers/dataloader')
from data_loader_zuco import ZucoLoader


def rat_num(len_sen, rationale):
    try:
        rationale = [el - 1 for el in list(map(int, rationale.split(",")))] if len(rationale.split(",")) > 1 else [
            int(float(rationale)) - 1]
        return [1 if idx in rationale else 0 for idx in range(len_sen)]
    except ValueError:
        import pdb;
        pdb.set_trace()


df = pd.read_pickle('/Users/wnd306/OneDrive - University of Copenhagen/PapersConferences/2022ACL/data/'
                    'dfs_bert_finetuned_sst/bert_sst_without_zuco_bert-base-uncased_finetuned/'
                    'bert_sst_without_zuco_bert-base-uncased_finetuned_bert_evaluation.pkl').query('labels!=1')
df['labels'] = df['labels'].astype(int)
df = df.query('labels==ypred')

sst = pd.read_pickle("/Users/wnd306/OneDrive - University of Copenhagen/Data/Text/sst_pos-neg.pkl")

df_lrp = pd.read_pickle('../data/SST_lrp/df_fine_bert_lrp.pkl')

zuco_files = '/Users/wnd306/OneDrive - University of Copenhagen/PapersConferences/2022ACL/data/preprocessed'
ZM = ZucoLoader(zuco1_prepr_path=zuco_files, zuco2_prepr_path=None)
df_human_tsr = ZM.get_zuco_task_df(zuco_version=1, task='SR')
df_human_avg_tsr = get_human_avg_df(df_human_tsr, ignore_zeros=False).query('labels!=0')
df_human_avg_tsr['tokens'] = df_human_avg_tsr['words'].apply(lambda x: [el for el in x])

pad_token = '[PAD]'
sep_token = '[SEP]'
special_tokens = ['[PAD]', '[CLS]', '[SEP]', '</s>', '<s>', '<pad>']
modelname = 'mbert'

df_attention = pd.DataFrame(columns=['text_id', 'tokens', 'relfix', 'lrp'])
model_importance = ['attention_0', 'attention_5', 'attention_11', 'flow_11']

ii = -1
for irow, row in df.iterrows():
    ii += 1
    print(ii)
    df_attention.loc[ii] = [None for _ in range(len(df_attention.columns))]
    for col in model_importance:
        if col not in df_attention:
            df_attention[col] = None
        if 'attention' in col:
            att_columns = [att_col for att_col in df.columns if f'x_{col.split("_")[-1]}' in att_col]
            assert (len(att_columns) == 12)
            att = np.mean(row[att_columns].tolist(), axis=0)
        else:
            att = row['x_' + col]
        if len(att) != len(row['tokens']):
            import pdb;

            pdb.set_trace()

        att = [a for t, a in zip(row['tokens'], att) if t not in special_tokens]
        tokens = [t for t in row['tokens'] if t not in special_tokens]
        # relative_attention = scipy.special.softmax(att)
        relative_attention = att
        tokens, merged_attention = merge_subwords(tokens, relative_attention)
        tokens, merged_attention = merge_hyphens(tokens, merged_attention)
        tokens, merged_attention = merge_symbols(tokens, merged_attention)
        tokens, merged_attention = merge_symbols(tokens, merged_attention)
        df_attention.at[ii, col] = merged_attention

    index = []
    for jrow, row_fix in df_human_avg_tsr.iterrows():
        if tokens[0].lower() == row_fix['tokens'][0].lower():
            index.append(row_fix['text_id'])
    if len(index) == 1:
        df_attention.loc[ii, 'text_id'] = index[0]
        df_attention.at[ii, 'tokens'] = tokens
    elif len(index) > 1:
        it = 0
        while len(index) > 1:
            it += 1
            old_index = index
            index = []
            for idx in old_index:
                if tokens[it].lower() == df_human_avg_tsr.query('text_id==@idx')['tokens'].tolist()[0][it].lower():
                    index.append(idx)
            if len(index) == 1:
                df_attention.loc[ii, 'text_id'] = index[0]
                df_attention.at[ii, 'tokens'] = tokens
    else:
        print(tokens[0], '\n')
        for hh, hrow in df_human_avg_tsr.iterrows():
            print(hrow['tokens'][0])
        import pdb;

        pdb.set_trace()

    idx_fix = df_human_avg_tsr.query('text_id==@index[0]').index[0]
    assert len(df_attention.loc[ii, 'flow_11']) == len(df_human_avg_tsr.loc[idx_fix, 'tokens'])

    df_attention.at[ii, 'relfix'] = [float(s) / np.nansum(df_human_avg_tsr.loc[idx_fix, 'x']) for s in
                                     df_human_avg_tsr.loc[idx_fix, 'x']]

    assert len(df_lrp.loc[df_attention.loc[ii, 'text_id'], '_x_raw']) == len(
        df_lrp.loc[df_attention.loc[ii, 'text_id'], '_tokens_raw'])
    att = abs(df_lrp.loc[df_attention.loc[ii, 'text_id'], '_x_raw'])
    att = [a for t, a in zip(df_lrp.loc[df_attention.loc[ii, 'text_id'], '_tokens_raw'], att) if
           t not in special_tokens]
    tokens = [t for t in df_lrp.loc[df_attention.loc[ii, 'text_id'], '_tokens_raw'] if t not in special_tokens]
    # relative_attention = scipy.special.softmax(att)
    relative_attention = att
    tokens, merged_attention = merge_subwords(tokens, relative_attention)
    tokens, merged_attention = merge_hyphens(tokens, merged_attention)
    tokens, merged_attention = merge_symbols(tokens, merged_attention)
    tokens, merged_attention = merge_symbols(tokens, merged_attention)
    assert len(merged_attention) == len(df_attention.loc[ii, 'flow_11'])
    df_attention.at[ii, 'lrp'] = merged_attention

df_attention.to_pickle('sst_importance.pkl')
