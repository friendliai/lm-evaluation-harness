# Periflow evaluation harness

## Overview
- We only support subset of lm-evaluation-harness. Please check [periflow-supported-task-table](periflow_supported_task_table.md).

## Install

To install the `lm-eval` refactor branch from the github repository, run:

```bash
git clone https://github.com/friendliai/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

To install additional multilingual tokenization and text segmentation packages, you must install the package with the `multilingual` extra:

```bash
pip install -e ".[multilingual]"
```

To install the package with all extras, run
```bash
pip install -e ".[all]"
```


## Evaluation Command
### 1. Sequential Request Evaluation
```bash
python main.py \
--model periflow \
--model_args model_name_or_path {model_name_of_path from huggingface hub},req_url={engine request url} \
--tasks {evaluation tas} \
--num_fewshot {number of fewshot samples} # optional. without this option, num_fewshot=0
--no_cache # optional. without this option, if evaluation result is existing, then skip the evaluation process, return cached results.
--average_acc_tasks # only for mmlu tasks. In lm-evaluation-harness, mmlu dataset contains a lots of seperated datasets. Using this option, the average acc of all seperated datsets is added in result table.
```

### 2. Async Request Evaluation
```bash
python main.py \
--model periflow_async \
--model_args model_name_or_path {model_name_of_path from huggingface hub},req_url={engine request url} \
--tasks {evaluation tas} \
--num_fewshot {number of fewshot samples} # optional. without this option, num_fewshot=0
--no_cache # optional. without this option, if evaluation result is existing, then skip the evaluation process, return cached results.
--average_acc_tasks # only for mmlu tasks. In lm-evaluation-harness, mmlu dataset contains a lots of seperated datasets. Using this option, the average acc of all seperated datsets is added in result table.
```

## Evaluation Result 
- Selected Tasks are same as [HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

### 1. Sequential Request Evaluation Result
|Model|ARC `acc_norm`|Hellaswag `acc_norm`|MMLU `average acc of all`|TruthfulQA `mc2`|
|---|---|---|---|---|
|Llama-2-7b-chat-hf|52.65|78.52|48.10|45.31|

### 2. Async Request Evaluation Result
|Model|ARC `acc_norm`|Hellaswag `acc_norm`|MMLU `average acc of all`|TruthfulQA `mc2`|
|---|---|---|---|---|
|Llama-2-7b-chat-hf|52.82|78.51|48.20|45.32|

