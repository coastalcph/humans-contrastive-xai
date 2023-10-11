import collections
import json
import random

from datasets import load_dataset

random.seed(42)
dataset = load_dataset('DeveloperOats/DBPedia_Classes')
# filtered_dataset = dataset.filter(lambda example: example['l2'] == 'Animal')
filtered_dataset = dataset.filter(lambda example: example['l2'] == 'Person' and example['l3'] in ['Architect', 'Journalist', 'Judge', 'MilitaryPerson', 'Model', 'Monarch', 'Noble', 'OfficeHolder', 'Philosopher', 'Religious'])
filtered_dataset_train = filtered_dataset['train'].select(random.sample(range(len(filtered_dataset['train'])), 8000))
label_counts = collections.Counter([example['l3'] for example in filtered_dataset_train])
print(label_counts.most_common())
with open('../train.jsonl', 'w') as file:
    for example in filtered_dataset_train:
        file.write(json.dumps({'text': example['text'], 'label': example['l3']}) + '\n')

filtered_dataset_validation = filtered_dataset['validation'].select(random.sample(range(len(filtered_dataset['validation'])), 1000))
label_counts = collections.Counter([example['l3'] for example in filtered_dataset_validation])
print(label_counts.most_common())
with open('../validation.jsonl', 'w') as file:
    for example in filtered_dataset_validation:
        file.write(json.dumps({'text': example['text'], 'label': example['l3']}) + '\n')

filtered_dataset_test = filtered_dataset['test'].select(random.sample(range(len(filtered_dataset['test'])), 1000))
label_counts = collections.Counter([example['l3'] for example in filtered_dataset_test])
print(label_counts.most_common())
with open('../test.jsonl', 'w') as file:
    for example in filtered_dataset_test:
        file.write(json.dumps({'text': example['text'], 'label': example['l3']}) + '\n')