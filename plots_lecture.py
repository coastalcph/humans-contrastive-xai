import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

relevance = [0.05, 0.1, -0.1, -0.05, -0.2, 0.01, 0.1, 0.01, 0.1, 0.2, -0.01, 0.34, 0.25, 0.01]
relevance = relevance/np.sum(relevance)
print(np.sum(relevance))
text = 'The acting could be better , but overall this movie is worth watching .'
# text = ['This', 'is', 'a', 'really', 'good', 'movie.']


idx_sorted = np.argsort(relevance)[::-1]

for istep, rel in enumerate(idx_sorted):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 0.5])
    tokens_masked = [text.split(" ")[idx] if idx in idx_sorted[:istep+1] else '[MASK]' for idx in range(len(text.split(" ")))]
    rel_masked = [relevance[idx] if idx in idx_sorted[:istep+1] else 0 for idx in
                  range(len(text.split(" ")))]
    print(tokens_masked)
    print(rel_masked)
    # tokens = [tok if itok in rel else '[MASK]' for itok, tok in enumerate(text.split(" "))]
    # print(tokens)

    sns.heatmap(np.array(rel_masked)[None, :],
                annot=np.array(tokens_masked)[None, :],
                fmt='', cmap='vlag',
                vmin=-np.max(relevance),
                vmax=np.max(relevance),
                annot_kws={"size": 6},
                cbar=False
                )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(f'expl{istep}.png', dpi=300, bbox_inches='tight')

istep+=1
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10, 0.5])
sns.heatmap(np.array(rel_masked)[None, :],
            annot=np.array(np.around(rel_masked, decimals=2))[None, :],
            fmt='', cmap='vlag',
            vmin=-np.max(relevance),
            vmax=np.max(relevance),
            annot_kws={"size": 6},
            cbar=False
            )
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f'expl{istep}.png', dpi=300, bbox_inches='tight')