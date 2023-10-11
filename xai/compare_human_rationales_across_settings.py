import pickle
import re
from os.path import join
import click
from xai.xai_utils.prodigy_annotations_utils import aggregate_annotations, EXCLUDED_ANNOTATORS
from utils import set_up_dir
from sklearn.metrics import cohen_kappa_score, f1_score
import numpy as np
from nltk.corpus import stopwords
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

@click.command()
@click.option('--results_dir', default='./results')
def main(results_dir):
    results_dir = join(results_dir, 'human-human')
    set_up_dir(results_dir)

    for aggregation_method in ['majority', 'full_overlap', 'any']:
        f1_scores = {}
        cohen_kappas = {}
        fleiss_kappas = {}
        for medical_occupation in ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician', 'all']:
            STANDARD_ANNOTATIONS, _ = aggregate_annotations(annotations_filename='standard_biosbias_rationales', aggregation_method=aggregation_method, label_name=medical_occupation,
                                                            exclude_annotators=EXCLUDED_ANNOTATORS)
            STANDARD_ANNOTATIONS = {re.sub('[^a-z]', '', key.lower()): value for key, value in STANDARD_ANNOTATIONS.items()}

            CONTRASTIVE_ANNOTATIONS, _ = aggregate_annotations(annotations_filename='contrastive_biosbias_rationales', aggregation_method=aggregation_method, label_name=medical_occupation,
                                                               exclude_annotators=EXCLUDED_ANNOTATORS)
            CONTRASTIVE_ANNOTATIONS = {re.sub('[^a-z]', '', key.lower()): value for key, value in CONTRASTIVE_ANNOTATIONS.items()}

            shared_samples = 0
            ref_rationales = []
            target_rationales = []
            for annotation_id in STANDARD_ANNOTATIONS:
                if annotation_id in CONTRASTIVE_ANNOTATIONS and (len(STANDARD_ANNOTATIONS[annotation_id]) == len(CONTRASTIVE_ANNOTATIONS[annotation_id])):
                    shared_samples += 1
                    ref_rationale = [score for word, score in STANDARD_ANNOTATIONS[annotation_id] if re.search('[a-z]', word.lower()) and word.lower() not in ENGLISH_STOPWORDS]
                    target_rationale = [score for word, score in CONTRASTIVE_ANNOTATIONS[annotation_id] if re.search('[a-z]', word.lower()) and word.lower() not in ENGLISH_STOPWORDS]
                    ref_rationales.extend(ref_rationale)
                    target_rationales.extend(target_rationale)

            f1_scores[medical_occupation] = f'{((f1_score(ref_rationales, target_rationales) + f1_score(target_rationales, ref_rationales)) / 2 ):.2f}'
            cohen_kappas[medical_occupation] = f'{cohen_kappa_score(ref_rationales, target_rationales):.2f}'
            fleiss_kappas[medical_occupation] = f'{fleiss_kappa(ref_rationales, target_rationales):.2f}'

        # Compare rationales
        print(f'Aggregation method: {aggregation_method.title()}')
        # print(f'Number of shared samples: {shared_samples}')
        print(f'Labels\t\t\t' + '\t'.join(['Psych', 'Surge', 'Nurse', 'Dent', 'Phys', 'All']))
        print('-' * 100)
        print(f'Token F1:\t\t' + '\t'.join([f1_scores[medical_occupation] for medical_occupation in ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician', 'all']]))
        print(f'Cohen’s Kappa:\t' + '\t'.join([cohen_kappas[medical_occupation] for medical_occupation in ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician', 'all']]))
        print(f'Fleiss’ Kappa:\t' + '\t'.join([fleiss_kappas[medical_occupation] for medical_occupation in ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician', 'all']]))
        print('-' * 100)
        print()

        labels = ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician', 'all']

        scores_out = (labels, f1_scores, cohen_kappas, fleiss_kappas)
        pickle.dump(scores_out, open(join(results_dir, f'{aggregation_method}_scores.pkl'), "wb"))


if __name__ == "__main__":
    main()
