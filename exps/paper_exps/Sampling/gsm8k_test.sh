# gsm8k
## SDP 
### Codellama
prefix='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_codellama' \
ckpt_name='global_step_2926_epoch_19' \
input_path='data/gsm8k_test_set.json' \
engine='python' \
batch_size='2' \
max_length='1024' \
num_return_sequences='100' \
do_sample='1' \
    bash exps/paper_exps/Sampling/_template.sh

### Galactica
prefix='ppo_paper_final_new/_models_outputs_sft/gsm8k_python_sdp_galactica' \
ckpt_name='global_step_6160_epoch_40' \
input_path='data/gsm8k_test_set.json' \
engine='python' \
batch_size='2' \
max_length='1024' \
num_return_sequences='100' \
do_sample='1' \
    bash exps/paper_exps/Sampling/_template.sh

## NL 
### Codellama
prefix='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_codellama' \
ckpt_name='global_step_4524_epoch_29' \
input_path='data/gsm8k_test_set.json' \
engine='nl' \
batch_size='2' \
max_length='1024' \
num_return_sequences='100' \
do_sample='1' \
    bash exps/paper_exps/Sampling/_template.sh

### Galactica
prefix='ppo_paper_final_new/_models_outputs_sft/gsm8k_nl_galactica' \
ckpt_name='global_step_5304_epoch_34' \
input_path='data/gsm8k_test_set.json' \
engine='nl' \
batch_size='2' \
max_length='1024' \
num_return_sequences='100' \
do_sample='1' \
    bash exps/paper_exps/Sampling/_template.sh
