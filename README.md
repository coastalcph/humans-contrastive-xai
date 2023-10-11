# An Investigation of Contrastive Explanations: Comparing Model post-hoc Explanations with Human Gaze / Rationales


## How to finetune a model?

```shell
MODEL_PATH='roberta-base'
DATASET_NAME='sst2' # 'dynabench/dynasent'
DATASET_CONFIG='sst2' # 'dynabench.dynasent.r1.all'
BATCH_SIZE=32
MAX_SEQ_LENGTH=64
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false


# DELETE CACHED DATASET
rm -rf ../.cache/huggingface/datasets/${DATASET_NAME}

# TRAIN STANDARD CLASSIFIER
python train_models/finetune_model.py \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config ${DATASET_CONFIG} \
    --output_dir data/finetuned_models/${MODEL_PATH}-${DATASET_NAME} \
    --do_train \
    --do_eval \
    --do_pred \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --greater_is_better true \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --seed 42 \
    --num_train_epochs 10 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --fp16 \
    --fp16_full_eval \
    --lr_scheduler_type cosine

```

## Available Fine-tuned Models

| Model Name                                                            | Dataset                                            |
|-----------------------------------------------------------------------|----------------------------------------------------|
 | `coastalcph/roberta-{small, base, large}-sst2`                        | `sst2`                                             |
| `coastalcph/roberta-base-dynasent`                                    | `dynabench/dynasent`                               |
| `coastalcph/{gpt2, roberta, t5}-{small, base, large}-dbpedia-animals` | `coastalcph/dbpedia-datasets`                      |
| `coastalcph/{gpt2, roberta, t5}-{small, base, large}-dbpedia-animals` | `coastalcph/xai_fairness_benchmark` |



## Explanations
### How to explain a model's predictions?

```shell
MODEL='roberta-base'
DATASET='sst2' # 'dynabench/dynasent'
MODE = true #foil, contrastive
SPLIT = test #validation
XAI_METHOD = lrp #gi, lrp_norm

python ./xai/extract_lrp_relevance.py \  
    --modelname coastalcph/${MODEL}-${DATASET} \
    --dataset_name ${DATASET} \
    --case ${XAI_METHOD} \
    --mode ${MODE} \
    --dataset_split ${SPLIT}
    
```

### How to run comparison across settings and humans/models?

```shell
python ./xai/xai_comparison.py \
    --source_path ./results/${MODEL}-${DATASET} \
    --xai_method ${XAI_METHOD} \
    --modelname ${MODEL}
    
python ./xai/compute_human_model_alignment.py \
    --modelname coastalcph/${MODEL}-${DATASET} \
    --dataset_name ${DATASET} \
    --model_type ${MODELTYPE} \
    --importance_aggregator ${AGGREGATOR} \
    --annotations_filename standard_biosbias_rationales \
    --results_dir ./results/${MODEL}-${DATASET}
  
python ./xai/compute_human_model_alignment.py \
    --modelname coastalcph/${MODEL}-${DATASET} \
    --dataset_name ${DATASET} \
    --model_type ${MODELTYPE} \
    --importance_aggregator ${AGGREGATOR} \
    --annotations_filename contrastive_biosbias_rationales \
    --results_dir ./results/${MODEL}-${DATASET}
  
python ./xai/compare_human_rationales_across_settings.py \
    --results_dir ./results/${MODEL}-${DATASET}
```