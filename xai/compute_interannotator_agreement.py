import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import numpy as np
from xai.xai_utils.annotations_utils import read_annotations

CONTRASTIVE_COMMON_SET = 9
CONTRASTIVE_FILENAME = "contrastive_screening"
STANDARD_COMMON_SET = 12
SETTING = "standard"
METRIC = cohen_kappa_score

# Compute inter-annotator agreement
annotations = read_annotations(SETTING, label_name='all')

# Compute inter-annotator agreement
annotators = {}
for key, annotator_entry in annotations.items():
    if len(annotator_entry.keys()) == STANDARD_COMMON_SET:
        for (annotator_id, annotation) in annotator_entry.items():
            annotator_id = annotator_id.split("-")[-1].strip("[]")[-4:]
            if annotator_id not in annotators:
                annotators[annotator_id] = [score for tok, score in annotation]
            else:
                annotators[annotator_id].extend([score for tok, score in annotation])

# Compute inter-annotator agreement
agreements = np.zeros([len(annotators), len(annotators)])
for i, (annotator_1_id, scores_1) in enumerate(annotators.items()):
    for j, (annotator_2_id, scores_2) in enumerate(annotators.items()):
        agreements[i][j] = METRIC(scores_1, scores_2)


# Plot inter-annotator agreement
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=[8, 8])
agreements[agreements == 0] = agreements.T[agreements == 0]
plt.imshow(agreements)
plt.colorbar()
axs.set_xticks(np.arange(0, len(annotators)), list(annotators.keys()), rotation=90)
axs.set_yticks(np.arange(0, len(annotators)), list(annotators.keys()))

axs.xaxis.tick_top()
axs.xaxis.set_label_position('top')
plt.title('Inter-annotator agreement (Cohen\'s kappa)')
plt.show()

# Plot inter-annotator agreement
for idx, annotator_id in enumerate(annotators.keys()):
    print(f'Annotator {annotator_id}: {(np.sum(agreements[idx]) - 1) * 100 / (len(annotators) - 1):.1f}')
