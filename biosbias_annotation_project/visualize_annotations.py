import copy
import json

total_count = 0
ANNOTATOR_ID = 'bios_batch_4_repeated-5a8367b9190420000155ec2a'
with open('../data/rationales-biosbias/standard_biosbias_rationales.jsonl') as file:
    for line in file:
        data = json.loads(line)
        if ANNOTATOR_ID in data['_annotator_id']:
            if 'spans' in data:
                tokens = [token['text'] for token in data['tokens']]
                spans = [(span['token_start'], span['token_end'], 1) for span in data['spans']]
                current_idx = 0
                final_spans = copy.deepcopy(spans)
                for span_idx, span in enumerate(spans):
                    final_spans.append((current_idx, span[0] - 1, 0))
                    current_idx = span[1] + 1
                final_spans.append((current_idx, len(tokens)-1, 0))
                ordered_spans = sorted(final_spans, key=lambda x: x[0])
                printed_spans = [f'[{" ".join(tokens[span_start:span_end+1])}]' if labeled else f'{" ".join(tokens[span_start:span_end+1] if span_start != span_end else [tokens[span_start]])}' for span_start, span_end, labeled in ordered_spans]
                print(' '.join(printed_spans))
                print('-' * 150)
                total_count += 1

print('Number of annotated samples: ', total_count)
