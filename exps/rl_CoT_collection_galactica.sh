#!/bin/bash
export TOKENIZERS_PARALLELISM=True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get the commit hash
commit_hash=$(git rev-parse --short HEAD)

# Get the commit name
commit_name=$(git log -1 --pretty=format:%s)

# save time in DDMMHHMM format
time=$(date +"%d%m%H%M")

# add time to exp_name
exp_name="CoT_collection_galactica_125m_${time}"
model_dir="thinking_models/_models_outputs_rl_small/${exp_name}"

train_file="jeggers/CoT-Collection"
model_name_or_path="facebook/galactica-125m"
tokenizer_name_or_path="facebook/galactica-125m"
ref_model_name_or_path="facebook/galactica-125m"

gamma="0.9999"
lam="0.999"
vf_coef="1"
kl_coef="0.005"
reward_correct="5"
reward_starts_correct="2.5"
reward_contains_answer_trigger=".5"
reward_max_gen_length="-0.3"
learning_rate="3e-7"
clip_grad_norm="1"
start_penalty_after="0"
penalty_warmup_steps="1"

ppo_epochs="2"
mini_batch_size="40"
eval_batch_size="50"
batch_size="75"
num_processes='8'

keep_num_ckpt='0'
n_epochs="700"
num_workers="0"
weight_decay="0"
warmup_step="0"
adv_whitening='local'
evaluating_epoch_freq="1"
logging_epoch_freq="1"
saving_epoch_freq="1"
evaluating_step_freq="-100"
logging_step_freq="1"
saving_step_freq="-100"
seed="42"
max_input_length="200"
max_gen_length="200"
wandb_log="True"
wandb_project="thinking_small"
wandb_run_name="${exp_name}"
temperature="1.0"
add_special_token_ids="True"

main_process_port='8889'

mkdir -p "${model_dir}"
accelerate launch \
        --config_file ./default_config_deepspeed.yaml \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
    train_thinking.py \
        --model_name_or_path "${model_name_or_path}" \
        --tokenizer_name_or_path "${tokenizer_name_or_path}" \
        --ref_model_name_or_path "${ref_model_name_or_path}" \
        --train_file "${train_file}" \
        --test_file "${test_file}" \
        --model_dir "${model_dir}" \
        --batch_size "${batch_size}" \
        --mini_batch_size "${mini_batch_size}" \
        --eval_batch_size "${eval_batch_size}" \
        --ppo_epochs "${ppo_epochs}" \
        --n_epochs "${n_epochs}" \
        --num_workers "${num_workers}" \
        --learning_rate "${learning_rate}" \
        --weight_decay "${weight_decay}" \
        --warmup_step "${warmup_step}" \
        --clip_grad_norm "${clip_grad_norm}" \
        --vf_coef "${vf_coef}" \
        --kl_coef "${kl_coef}" \
        --gamma "${gamma}" \
        --lam "${lam}" \
        --evaluating_epoch_freq "${evaluating_epoch_freq}" \
        --logging_epoch_freq "${logging_epoch_freq}" \
        --saving_epoch_freq "${saving_epoch_freq}" \
        --evaluating_step_freq "${evaluating_step_freq}" \
        --logging_step_freq "${logging_step_freq}" \
        --saving_step_freq "${saving_step_freq}" \
        --seed "${seed}" \
        --max_input_length "${max_input_length}" \
        --max_gen_length "${max_gen_length}" \
        --wandb_log "${wandb_log}" \
        --wandb_project "${wandb_project}" \
        --wandb_run_name "${wandb_run_name}" \
        --adv_whitening "${adv_whitening}" \
        --keep_num_ckpt "${keep_num_ckpt}" \
        --temperature "${temperature}" \
        --start_penalty_after "${start_penalty_after}" \
        --penalty_warmup_steps "${penalty_warmup_steps}" \
        --add_special_token_ids "${add_special_token_ids}" \
        --git_commit_hash "${commit_hash}" \
        --git_commit_name "${commit_name}" \
        --reward_correct "${reward_correct}" \
        --reward_starts_correct "${reward_starts_correct}" \
        --reward_contains_answer_trigger "${reward_contains_answer_trigger}" \
        --reward_max_gen_length "${reward_max_gen_length}" \
        1> >(tee "${model_dir}"/"${exp_name}".log) \
        2> >(tee "${model_dir}"/"${exp_name}".err >&2)
