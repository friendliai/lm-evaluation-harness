# ORCA evaluation harness

## Supported Task
We only support subset of lm-evaluation-harness. Please check [orca-supported-task-table](orca_supported_task_table.md).

## Evaluation Command

### 1. Sequential Request Evaluation
```bash
python main.py \
--model periflow \
--model_args model_name_or_path {model_name_of_path from huggingface hub},req_url={engine request url} \
--tasks {evaluation tas} \
--num_fewshot {number of fewshot samples} # optional. without this option, num_fewshot=0
--no-cache # optional. without this option, if evaluation result is existing, then skip the evaluation process, return cached results.
--average_acc_tasks # only for mmlu tasks. In lm-evaluation-harness, mmlu dataset contains a lots of seperated datasets. Using this option, the average acc of all seperated datsets is added in result table.
```


### 2. Async Request Evaluation
```bash
python main.py \
--model periflow_async \
--model_args model_name_or_path {model_name_of_path from huggingface hub},req_url={engine request url} \
--tasks {evaluation tas} \
--num_fewshot {number of fewshot samples} # optional. without this option, num_fewshot=0
--no-cache # optional. without this option, if evaluation result is existing, then skip the evaluation process, return cached results.
--average_acc_tasks # only for mmlu tasks. In lm-evaluation-harness, mmlu dataset contains a lots of seperated datasets. Using this option, the average acc of all seperated datsets is added in result table.
```

## Evaluation Result 
### 1. Sequential Request Evaluation Result
|Model|Average|ARC `acc_norm`|Hellaswag `acc_norm`|MMLU `average acc of all`|TruthfulQA `mc2`|
|---|---|---|---|---|---|
|Llama-2-7b-chat-hf|56.14|52.65|78.52|48.10|45.31|
