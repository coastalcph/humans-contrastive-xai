import json
import os
import argparse

from data import DATA_DIR

MODELS = {'roberta': ['coastalcph/roberta-small','roberta-base', 'roberta-large'],
          'gpt2': ['distilgpt2','gpt2', 'gpt2-medium'],
          't5': ['google/t5-v1_1-small', 'google/t5-v1_1-base', 'google/t5-v1_1-large']
          }
LRS = {
    'roberta': ['1e-5', '3e-5', '5e-5'],
    'gpt2': ['1e-5', '3e-5', '5e-5'],
    'sst2': ['1e-5', '3e-5', '5e-5'],
    'dynasent': ['1e-5', '3e-5', '5e-5'],
    't5': ['1e-3', '1e-4', '5e-5']
}
def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--dataset', default='biosbias')
    parser.add_argument('--subset', default='eval')
    parser.add_argument('--model_name', default='roberta')
    config = parser.parse_args()
    macroF1s, microF1s, accuracies = [], [], []
    for model_name in MODELS[config.model_name]:
        print('-' * 120)
        print(f'Model: {model_name}')
        for lr in LRS[config.model_name]:
            try:
                with open(os.path.join(DATA_DIR, 'finetuned_models', f'{model_name}-{config.dataset}-lr-{lr}', 'all_results.json')) as file:
                    data = json.load(file)

                macroF1s.append(data[f"{config.subset}_macro-f1"])
                microF1s.append(data[f"{config.subset}_micro-f1"])
                accuracies.append(data[f"{config.subset}_accuracy"])

                print(f'LR {lr:>3}:\t macro-F1: {data[f"{config.subset}_macro-f1"]* 100:.1f}\t\t'
                      f'micro-F1: {data[f"{config.subset}_macro-f1"]* 100:.1f}'
                      f'\t\taccuracy: {data[f"{config.subset}_accuracy"]* 100:.1f}')
            except FileNotFoundError:
                print(f'LR {lr:>3}:\t N/A')
        print('-' * 120)


if __name__ == '__main__':
    main()