#!/bin/bash
export TOKENIZERS_PARALLELISM=True
exp_name="CoT-Collection_nl_galactica_125m_reft_01"
model_dir="thinking_models/_models_outputs_rl_small/CoT-Collection_nl_galactica_125m_reft_01"
# train_file="data/gsm8k_python_sdp.json"
# test_file="data/gsm8k_test_set.json"
train_file="jeggers/CoT-Collection"
engine='nl' # 'python' or 'nl'

model_name_or_path="facebook/galactica-125m"
tokenizer_name_or_path="facebook/galactica-125m"
ref_model_name_or_path="facebook/galactica-125m"
value_model_name_or_path="facebook/galactica-125m"

max_per_task=100
max_test_per_task=100
keep_num_ckpt='0'
batch_size="50"
mini_batch_size="50"
eval_batch_size="50"
ppo_epochs="2"
n_epochs="700"
num_workers="0"
learning_rate="3e-6"
weight_decay="0"
warmup_step="0"
clip_grad_norm="1"
vf_coef="5"
kl_coef="0.01"
gamma="1.0"
lam="0.95"
adv_whitening='global'
evaluating_epoch_freq="1"
logging_epoch_freq="1"
saving_epoch_freq="-100"
evaluating_step_freq="-100"
logging_step_freq="1"
saving_step_freq="-100"
seed="42"
max_input_length="280"
max_gen_length="70"
wandb_log="True"
wandb_project="thinking_small"
wandb_run_name="${exp_name}"
policy_update_steps='10'

num_processes='1'
main_process_port='8889'
no_policy_loss_steps=30

mkdir -p "${model_dir}"
accelerate launch \
            --config_file ./default_config.yaml \
            --num_processes=${num_processes} \
            --main_process_port=${main_process_port} \
    train_thinking.py \
            --model_name_or_path "${model_name_or_path}" \
            --tokenizer_name_or_path "${tokenizer_name_or_path}" \
            --ref_model_name_or_path "${ref_model_name_or_path}" \
            --value_model_name_or_path "${value_model_name_or_path}" \
            --train_file "${train_file}" \
            --test_file "${test_file}" \
            --model_dir "${model_dir}" \
            --batch_size "${batch_size}" \
            --mini_batch_size "${mini_batch_size}" \
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
            --engine "${engine}" \
            --adv_whitening "${adv_whitening}" \
            --keep_num_ckpt "${keep_num_ckpt}" \
            --max_per_task "${max_per_task}" \
            --no_policy_loss_steps "${no_policy_loss_steps}" \
            --policy_update_steps "${policy_update_steps}" \
            1> >(tee "${model_dir}"/"${exp_name}".log) \
            2> >(tee "${model_dir}"/"${exp_name}".err >&2)
