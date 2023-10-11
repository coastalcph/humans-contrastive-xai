#!/usr/notebooks/env python
# coding=utf-8
""" Fine-tuning HF Classifiers."""

import logging
import os
import random
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
import glob
import shutil

import transformers
from transformers import (
    Trainer,
    Seq2SeqTrainer,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    AutoModelForSequenceClassification,
    default_data_collator,
    T5ForConditionalGeneration,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from data_helpers import filter_out_sst2, filter_out_dynasent, fix_dynasent

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.20.0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)

FILTER_OUT_FUNCTS = {'sst2': filter_out_sst2, 'dynabench/dynasent': filter_out_dynasent}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="sst2",
        metadata={
            "help": "The name of the dataset in HF Hub."
        },
    )
    dataset_config: Optional[str] = field(
        default="sst2",
        metadata={
            "help": "The name of the dataset configuration in HF Hub, if applicable."
        },
    )
    max_seq_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    filter_out_dataset: Optional[int] = field(
        default=True,
        metadata={
            "help": "Whether to filter out dataset."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
    server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default=None,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup distant debugging if needed
    if data_args.server_ip and data_args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(data_args.server_ip, data_args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Labels
    if data_args.dataset_name in ['sst2', 'dynabench/dynasent']:
        label_list = [0, 1]
        labels_names = ['negative', 'positive']
        num_labels = len(label_list)
        text_column_name = 'sentence'
    elif data_args.dataset_name == 'coastalcph/dbpedia-datasets':
        label_list = [0, 1, 2, 3, 4, 5, 6, 7]
        if 't5' in model_args.model_name_or_path:
            labels_names = ['<extra_id_0>', '<extra_id_1>', '<extra_id_2>', '<extra_id_3>', '<extra_id_4>', '<extra_id_5>', '<extra_id_6>', '<extra_id_7>']
        else:
            labels_names = ['Amphibian', 'Arachnid', 'Bird', 'Crustacean', 'Fish', 'Insect', 'Mollusca', 'Reptile']
        num_labels = len(label_list)
        text_column_name = 'text'
    elif data_args.dataset_name == 'coastalcph/xai_fairness_benchmark':
        label_list = [0, 1, 2, 3, 4]
        labels_names = ['psychologist', 'surgeon', 'nurse', 'dentist', 'physician']
        num_labels = len(label_list)
        text_column_name = 'text'

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading eurlex dataset from the hub.
    if training_args.do_train:
        train_dataset = load_dataset(data_args.dataset_name, data_args.dataset_config, split="train", use_auth_token=model_args.use_auth_token)

        # Filter out
        if data_args.filter_out_dataset:
            if data_args.dataset_name == 'dynabench/dynasent':
                train_dataset = fix_dynasent(train_dataset)
            train_dataset = FILTER_OUT_FUNCTS[data_args.dataset_name](train_dataset)

    if training_args.do_eval:
        eval_dataset = load_dataset(data_args.dataset_name, data_args.dataset_config, split="validation", use_auth_token=model_args.use_auth_token)

        # Filter out
        if data_args.filter_out_dataset:
            if data_args.dataset_name == 'dynabench/dynasent':
                eval_dataset = fix_dynasent(eval_dataset)
            eval_dataset = FILTER_OUT_FUNCTS[data_args.dataset_name](eval_dataset)

    if training_args.do_predict:
        predict_dataset = load_dataset(data_args.dataset_name, data_args.dataset_config, split="test", use_auth_token=model_args.use_auth_token)

        if data_args.dataset_name == 'dynabench/dynasent':
            predict_dataset = fix_dynasent(predict_dataset)

    # Label descriptors mode
    label_desc2id = {label_desc: idx for idx, label_desc in enumerate(labels_names)}
    label_id2desc = {idx: label_desc.lower() for idx, label_desc in enumerate(labels_names)}

    print(f'LabelDesc2Id: {label_desc2id}')

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        label2id={l: i for i, l in enumerate(labels_names)},
        id2label={i: l for i, l in enumerate(labels_names)},
        use_auth_token=model_args.use_auth_token,
        finetuning_task=data_args.dataset_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    if 'gpt2' in model_args.model_name_or_path:
        config.pad_token_id = config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        use_auth_token=model_args.use_auth_token,
        revision=model_args.model_revision,
    )

    if 'gpt2' in model_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token

    if 't5' in model_args.model_name_or_path:
        model_class = T5ForConditionalGeneration
        training_args.generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path)
        # Define se2seq training configuration parameters
        training_args.generation_config.max_length = 2
        training_args.generation_max_length = 2
        training_args.generation_config.min_length = 2
        training_args.generation_min_length = 2
        training_args.generation_config.num_beams = 1
        training_args.generation_num_beams = 1
        training_args.predict_with_generate = True
        config.max_length = 2
        config.min_length = 2
    else:
        model_class = AutoModelForSequenceClassification

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        use_auth_token=model_args.use_auth_token,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )

    # Preprocessing the datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    def preprocess_function(examples):
        # Tokenize the texts
        batch = tokenizer(
            examples[text_column_name],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
        )

        # Tokenize labels for T5
        if 't5' in model_args.model_name_or_path:
            label_batch = tokenizer(
                [label_id2desc[label] for label in examples["label"]],
                padding=False,
                max_length=2,
                truncation=True,
            )
            batch['labels'] = label_batch['input_ids']

        return batch

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['label'] if 't5' in model_args.model_name_or_path else None,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['label'] if 't5' in model_args.model_name_or_path else None,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=['label'] if 't5' in model_args.model_name_or_path else None,
                load_from_cache_file=False,
                desc="Running tokenizer on prediction dataset",
            )

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        if 't5' in model_args.model_name_or_path:
            preds = [tokenizer.decode(pred).replace('<pad>', '').replace('</s>', '').replace('<s>', '').strip() for pred in p.predictions]
            preds = [label_desc2id[pred] if pred in label_desc2id else 0 for pred in preds]
            labels = [tokenizer.decode(label).replace('<pad>', '').replace('</s>', '').replace('<s>', '').strip() for label in p.label_ids]
            p.label_ids = [label_desc2id[label] if label in label_desc2id else 0 for label in labels]
        else:
            logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
        accuracy = accuracy_score(y_true=p.label_ids, y_pred=preds)
        return {'macro-f1': macro_f1, 'micro-f1': micro_f1, 'accuracy': accuracy}

    # Initialize our Trainer

    trainer_class = Seq2SeqTrainer if 't5' in model_args.model_name_or_path else Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=2) \
        if 't5' in model_args.model_name_or_path else default_data_collator

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        if 't5' in model_args.model_name_or_path:
            preds = [tokenizer.decode(pred).replace('<pad>', '').replace('</s>', '').replace('<s>', '').strip() for pred in predictions]
            hard_predictions = [label_desc2id[pred] if pred in label_desc2id else 0 for pred in preds]
            labels = [tokenizer.decode(label).replace('<pad>', '').replace('</s>', '').replace('<s>', '').strip() for label in labels]
            labels = [label_desc2id[label] if label in label_desc2id else 0 for label in labels]

        else:
            hard_predictions = np.argmax(predictions, axis=-1)

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        report_predict_file = os.path.join(training_args.output_dir, "test_classification_report.txt")
        output_predict_file = os.path.join(training_args.output_dir, "test_predictions.csv")
        if trainer.is_world_process_zero():
            cls_report = classification_report(y_true=labels, y_pred=hard_predictions,
                                               target_names=list(config.label2id.keys()),
                                               zero_division=0)
            with open(report_predict_file, "w") as writer:
                writer.write(cls_report)
            with open(output_predict_file, "w") as writer:
                try:
                    for index, pred_list in enumerate(predictions[0]):
                        pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                        writer.write(f"{index}\t{pred_line}\n")
                except:
                    try:
                        for index, pred_list in enumerate(predictions):
                            pred_line = '\t'.join([f'{pred:.5f}' for pred in pred_list])
                            writer.write(f"{index}\t{pred_line}\n")
                    except:
                        pass

            logger.info(cls_report)

    # Save Model to Hub
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # Clean up checkpoints
    checkpoints = [filepath for filepath in glob.glob(f'{training_args.output_dir}/*/') if '/checkpoint' in filepath]
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)


if __name__ == "__main__":
    main()
