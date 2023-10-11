import json

indices = []
total_count = 0
unique_count = 0
annotations = {}
annotators = {}
with open(f'../data/rationales-biosbias/standard_biosbias_rationales.jsonl') as file:
    for line in file:
        data = json.loads(line)
        total_count += 1
        indices.append(data['_input_hash'])
        if data['_annotator_id'] in ['bios_batch_1_v2-12345', 'bios_batch_4-[prolific-id]', 'bios_batch_4-607717eaae6e81fa5a889d7f',
                                     'bios_batch_3-[612cecf6ebfcc62494f287eb]', 'bios_batch_4_repeated-5a8367b9190420000155ec2a']:
            continue
        if data['text'].split('Bio: ')[-1] not in annotations and 'spans' in data:
            unique_count += 1
            annotations[data['text'].split('Bio: ')[-1]] = [(data['_annotator_id'], data['_timestamp'], data['spans'] if 'spans' in data else None, data['text'])]
            if data['_annotator_id'] not in annotators:
                annotators[data['_annotator_id']] = 1
            else:
                annotators[data['_annotator_id']] += 1
        elif 'spans' in data:
            annotations[data['text'].split('Bio: ')[-1]].append((data['_annotator_id'], data['_timestamp'], data['spans'] if 'spans' in data else None, data['text']))
            if data['_annotator_id'] not in annotators:
                annotators[data['_annotator_id']] = 1
            else:
                annotators[data['_annotator_id']] += 1

print(total_count)
print(unique_count)

annotators_list = [annotation[0].split('-')[-1].strip('[]') for annotation in annotations[list(annotations.keys())[0]]]

print(annotators_list)
print(len(annotators_list))

for annotation in annotations[list(annotations.keys())[0]]:
    print(annotation)


# seconds = 1684398909 - 1684398064
# print(seconds)
# seconds = 1684399684 - 1684398696
# print(seconds)
#
# # Convert timestamp to datetime
# import datetime
# timestamp = 1684399684
# dt_object = datetime.datetime.fromtimestamp(timestamp)
# print(dt_object)