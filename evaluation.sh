#!/bin/bash
export TOKENIZERS_PARALLELISM=True

model_name="google/gemma-2-2b-it"
dataset_name= "jeggers/CoT-Collection"
split_name="test_in_dist"
question_column_name="final_input"
answer_column_name="final_target"
max_gen_length=380
batch_size=128
pad_to_multiple_of=8
cot_trigger="BOT: "
answer_trigger="ANSWER: "
instruction=""

accelerate launch \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
    evaluation.py \
        --model_name=${model_name} \
        --dataset_name=${dataset_name} \
        --split_name=${split_name} \
        --question_column_name=${question_column_name} \
        --answer_column_name=${answer_column_name} \
        --max_gen_length=${max_gen_length} \
        --batch_size=${batch_size} \
        --pad_to_multiple_of=${pad_to_multiple_of} \
        --cot_trigger=${cot_trigger} \
        --answer_trigger=${answer_trigger} \	
        --instruction=${instruction} \

