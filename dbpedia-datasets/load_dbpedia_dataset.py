from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset('coastalcph/dbpedia-datasets', 'animals', use_auth_token='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')
# print(dataset['train'].features['label'].names)
#
# text_lengths = [len(text.split()) for text in dataset['train']['text']]
#
# plt.hist(text_lengths, bins=20)
# plt.show()

