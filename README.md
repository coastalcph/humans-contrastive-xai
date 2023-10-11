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

| Model Name                            | Dataset                |
|---------------------------------------|------------------------|
 | `coastalcph/roberta-base-sst2`        | `sst2`                 |
| `coastalcph/roberta-base-dynasent`    | `dynabench/dynasent`   |


## How to explain a model's predictions?

TODO