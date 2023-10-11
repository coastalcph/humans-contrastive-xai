import pickle
import re
from os.path import join
import click
from prodigy_annotations_utils import read_annotations, EXCLUDED_ANNOTATORS
from sklearn.metrics import cohen_kappa_score, f1_score
from nltk.corpus import stopwords
import numpy as np
ENGLISH_STOPWORDS = set(stopwords.words('english'))


def fleiss_kappa(rater1_results, rater2_results):
    """
    Calculate Fleiss' kappa for two raters.

    Arguments:
    rater1_results -- List of results for rater 1 (e.g., [0, 1, 2, 1, 0])
    rater2_results -- List of results for rater 2 (e.g., [0, 1, 1, 1, 2])

    Returns:
    Fleiss' kappa value.
    """

    # Count the number of categories
    categories = set(rater1_results + rater2_results)
    num_categories = len(categories)

    # Count the number of subjects (items)
    num_subjects = len(rater1_results)

    # Calculate the observed agreement
    observed_agreement = sum((rater1_results[i] == rater2_results[i]) for i in range(num_subjects)) / num_subjects

    # Calculate the chance agreement
    p_j = np.zeros(num_categories)  # Probability of each category
    for category in categories:
        p_j[category] = (sum((result == category) for result in rater1_results) +
                         sum((result == category) for result in rater2_results)) / (num_subjects * 2)

    chance_agreement = np.sum(p_j ** 2)

    # Calculate Fleiss' kappa
    kappa = (observed_agreement - chance_agreement) / (1 - chance_agreement)

    return kappa

#contrastive
# Number of shared samples: 113
# Labels			Psych	Surge	Nurse	Dent	Phys	All
# ----------------------------------------------------------------------------------------------------
# Token F1:		55.90	66.57	52.58	68.40	31.17	53.96
# Cohen’s Kappa:	51.61	63.71	49.07	64.83	26.14	50.08
# Fleiss’ Kappa:	50.15	62.70	47.81	64.31	24.33	48.81
# ----------------------------------------------------------------------------------------------------


@click.command()
@click.option('--annotations_filename', default='contrastive_biosbias_rationales')
def main(annotations_filename):

    f1_scores = {}
    cohen_kappas = {}
    fleiss_kappas = {}
    for medical_occupation in ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician', 'all']:
        ANNOTATIONS, _ = read_annotations(annotations_filename=annotations_filename, label_name=medical_occupation,
                                                   exclude_annotators=EXCLUDED_ANNOTATORS)
        ANNOTATIONS = {re.sub('[^a-z]', '', key.lower()): value for key, value in ANNOTATIONS.items()}
        ANNOTATIONS = {annotation_id: [ANNOTATIONS[annotation_id][annotator_id] for annotator_id in ANNOTATIONS[annotation_id]][:3]
                                for annotation_id in ANNOTATIONS if len(ANNOTATIONS[annotation_id]) >= 3}
        shared_samples = len(ANNOTATIONS)
        temp_f1_scores = []
        temp_cohen_kappas = []
        temp_fleiss_kappa = []
        for annotation_id in ANNOTATIONS:
            rationales = []
            sample_f1_scores = []
            sample_cohen_kappas = []
            sample_fleiss_kappa = []
            for annotation in ANNOTATIONS[annotation_id]:
                rationale = [score for word, score in annotation if re.search('[a-z]', word.lower()) and word.lower() not in ENGLISH_STOPWORDS]
                rationales.append(rationale)
            for pairs in [(0, 1), (0, 2), (1, 2)]:
                sample_f1_scores.append((f1_score(rationales[pairs[0]], rationales[pairs[1]]) + f1_score(rationales[pairs[1]], rationales[pairs[0]])) / 2)
                sample_cohen_kappas.append(cohen_kappa_score(rationales[pairs[0]], rationales[pairs[1]]))
                sample_fleiss_kappa.append(fleiss_kappa(rationales[pairs[0]], rationales[pairs[1]]))
            temp_f1_scores.append(np.mean(sample_f1_scores))
            temp_cohen_kappas.append(np.mean(sample_cohen_kappas))
            temp_fleiss_kappa.append(np.mean(sample_fleiss_kappa))

        f1_scores[medical_occupation] = f'{np.mean(temp_f1_scores):.2f}'
        cohen_kappas[medical_occupation] = f'{np.mean(temp_cohen_kappas):.2f}'
        fleiss_kappas[medical_occupation] = f'{np.mean(temp_fleiss_kappa):.2f}'

    # Compare rationales
    print(f'Number of shared samples: {shared_samples}')
    print(f'Labels\t\t\t' + '\t'.join(['Psych', 'Surge', 'Nurse', 'Dent', 'Phys', 'All']))
    print('-' * 100)
    print(f'Token F1:\t\t' + '\t'.join([f1_scores[medical_occupation] for medical_occupation in ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician', 'all']]))
    print(f'Cohen’s Kappa:\t' + '\t'.join([cohen_kappas[medical_occupation] for medical_occupation in ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician', 'all']]))
    print(f'Fleiss’ Kappa:\t' + '\t'.join([fleiss_kappas[medical_occupation] for medical_occupation in ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician', 'all']]))
    print('-' * 100)
    print()

    pickle.dump(cohen_kappas, open(join('../results', f'{annotations_filename}_scores.pkl'), 'wb'))


if __name__ == "__main__":
    main()
