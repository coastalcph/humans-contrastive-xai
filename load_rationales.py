import json

indices = []
total_count = 0
unique_count = 0
annotations = {}
with open('./data/annotations/batch#2/bios_batch_2.jsonl') as file:
    for line in file:
        data = json.loads(line)
        print(data['_annotator_id'])
        total_count += 1
        indices.append(data['_input_hash'])
        if data['_input_hash'] not in annotations:
            unique_count += 1
            annotations[data['_input_hash']] = [(data['_annotator_id'], data['spans'] if 'spans' in data else None, data['text'])]
        else:
            annotations[data['_input_hash']].append((data['_annotator_id'], data['spans'] if 'spans' in data else None, data['text']))

print(total_count)
print(unique_count)

import pdb;pdb.set_trace()