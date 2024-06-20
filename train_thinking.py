from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import (
    load_dataset,
    DatasetDict,
)
from datetime import timedelta
from functools import partial
import json
import os
import random
from src.utils import (
    set_seed,
    discount_cumsum,
    do_gather,
    allgather,
    allgather_masked_whiten,
)
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import deepspeed
from transformers import (
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)
from trl import AutoModelForCausalLMWithValueHead
from trl.core import masked_mean, masked_var, masked_whiten, logprobs_from_logits
import numpy as np
import wandb
import shutil
from dotenv import load_dotenv
import random

load_dotenv()

tqdm = partial(tqdm, ncols=0, leave=False)

TIMEOUT = 10

REWARD_STARTSWITH_CORRECT = 0.5
REWARD_EXACT_MATCH = 1.0
REWARD_CONTAINS_ANSWER_TRIGGER = 0.1
REWARD_ALLOW_NEGATIVE = True
REWARD_MAX_GEN_LENGTH = -1

IN_COL = "final_input"
OUT_COL = "final_target"

penalty_trigger = "Penalty:"
problem_prefix = "Problem:"
# must not be present in the problem text
# cot_trigger = "Let's think step-by-step:"
cot_trigger = "My Solution:"
answer_trigger = "Answer:"

# instruction = f"""
# Solve the problem below.
# You may think step-by-step.
# To indicate your final answer, write '{answer_trigger}' followed by your answer.
# """

instruction = f"""
Here is the my problem and its solution.
Some problems I broke down into steps.
The final answer is always indicated by '{answer_trigger}'.
"""


def format_input_batch(input_batch, penalties=None, answer_trigger="", outputs=None):
    if penalties is None:
        penalties = [0.001] * len(input_batch)

    if outputs is None:
        outputs = [""] * len(input_batch)

    return [
        f"{penalty_trigger} {penalty:.5f}\n{instruction}\n\n{problem_prefix}\n{input}\n\n{cot_trigger}\n{answer_trigger}{output}"
        for input, output, penalty in zip(input_batch, outputs, penalties)
    ]


# takes a batch of input and completion strings
# returns a list of completion strings
def extract_completion_batch(input_and_completion_batch):
    splitted = [res.split(cot_trigger) for res in input_and_completion_batch]
    return [cot_trigger.join(split[1:])[1:] for split in splitted]


# assumes to get back only generated tokens
# TODO: verify if this is correct
# returns empty string if no answer trigger is found
def extract_answer_cot_batch(answer_cot_batch):
    splitted = [res.split(answer_trigger) for res in answer_cot_batch]
    return [splitted[-1] if len(splitted) >= 2 else "" for splitted in splitted]


def check_answer(extracted_ans, target_answer):
    if extracted_ans.strip() == target_answer:
        return REWARD_EXACT_MATCH
    if extracted_ans.strip().startswith(target_answer):
        return REWARD_STARTSWITH_CORRECT
    return 0


def compare_and_calculate_reward(cot, target_answer, token_count=0, cot_penalty=0):
    reward = 0
    if answer_trigger in cot:
        extracted_ans = extract_answer_cot_batch([cot])[0]
        reward = check_answer(extracted_ans, target_answer)
        if reward == 0:
            reward += REWARD_CONTAINS_ANSWER_TRIGGER

    reward -= cot_penalty * token_count

    if not REWARD_ALLOW_NEGATIVE:
        reward = max(0, reward)
    return reward


def sample_log_uniform(a, b, size=1):
    # Take logarithm of the bounds
    log_a = np.log(a)
    log_b = np.log(b)
    # Sample from uniform distribution in the range [log_a, log_b]
    u = np.random.uniform(log_a, log_b, size)
    # Transform the sample using the exponential function
    x = np.exp(u)
    return x


def prepare_deepspeed_ref_model(model):
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepspeed_plugin.deepspeed_config
    if model is not None:
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if (
                hidden_size is not None
                and config_kwargs["zero_optimization"]["stage"] == 3
            ):
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size
                        * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10
                        * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9
                        * hidden_size
                        * hidden_size,
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


def prepare_datasets_and_data_loaders(args, tokenizer):
    with accelerator.main_process_first():
        dataset = load_dataset("jeggers/CoT-Collection")

        randon_elems = random.sample(range(len(dataset["test"])), 500)
        dataset = DatasetDict(
            {
                "train": dataset["train"],
                "test_all": dataset["test"],
                "test_small": dataset["test"].select(randon_elems),
            }
        )

        accelerator.print("raw data:", dataset)

        # check if CoT trigger is present in the input
        def check_cot_trigger(batch):
            if any([cot_trigger in item for item in batch[IN_COL]]):
                raise ValueError("CoT trigger found in input")

            return batch

        check_cot_trigger(dataset, batched=True, batch_size=10000)

        # filter for tokens left
        def filter_fn(batch):
            formatted_input = format_input_batch(
                batch[IN_COL], answer_trigger=answer_trigger, outputs=batch[OUT_COL]
            )
            encoded = tokenizer(formatted_input)["input_ids"]
            max_len = args["max_input_length"]
            result = [len(toks) < max_len for toks in encoded]
            return result

        dataset = dataset.filter(filter_fn, batched=True, batch_size=10000)

        accelerator.print("filtered data:", dataset)
        accelerator.print("Using instruction:", instruction)
        accelerator.print("Using problem_prefix:", problem_prefix)
        accelerator.print("Using cot_trigger:", cot_trigger)
        accelerator.print("Using answer_trigger:", answer_trigger)

        def tokenize_fn(batch, tokenizer):
            assert tokenizer.eos_token_id is not None, (
                tokenizer.eos_token_id,
                tokenizer.eos_token,
            )
            penalties = sample_log_uniform(0.0005, 0.01, len(batch[IN_COL]))
            formatted_input = format_input_batch(batch[IN_COL], penalties=penalties)

            encoded_input = tokenizer(
                formatted_input, add_special_tokens=False, padding="longest"
            )

            encoded_input["formatted_input"] = formatted_input
            encoded_input["raw_input"] = batch[IN_COL]
            encoded_input["dataset_name"] = batch["dataset"]
            encoded_input["target"] = batch[OUT_COL]
            encoded_input["penalty"] = penalties
            return encoded_input

        tokenized_dataset = DatasetDict(
            {
                mode: dataset.map(
                    tokenize_fn,
                    fn_kwargs={"tokenizer": tokenizer},
                    batched=True,
                    batch_size=args["batch_size"],
                    remove_columns=dataset.column_names,
                    drop_last_batch=True,
                )
                for mode, dataset in dataset.items()
            }
        )
        accelerator.print("tokenized data:", tokenized_dataset)

        if accelerator.is_main_process and args["wandb_log"]:
            wandb.config.update(
                {
                    "instruction": instruction,
                    "problem_prefix": problem_prefix,
                    "cot_trigger": cot_trigger,
                    "answer_trigger": answer_trigger,
                    "raw_dataset": str(dataset),
                    "tokenized_dataset": str(tokenized_dataset),
                }
            )

    def collate_fn(batch):
        # note batch now list of dicts instead dict of lists!

        input_ids = torch.LongTensor([item["input_ids"] for item in batch])
        attention_mask = torch.BoolTensor([item["attention_mask"] for item in batch])

        result = {
            "formatted_input": [item["formatted_input"] for item in batch],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": [item["target"] for item in batch],
            "dataset_name": [item["dataset_name"] for item in batch],
            "penalty": [item["penalty"] for item in batch],
        }

        return result

    train_dataloader, test_all_dataloader, test_small_dataloader = [
        DataLoader(
            tokenized_dataset[mode],
            batch_size=args["batch_size"],
            num_workers=args["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn,
        )
        for mode in dataset.keys()
    ]

    return train_dataloader, test_all_dataloader, test_small_dataloader


def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    os.makedirs(save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        save_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    tokenizer.save_pretrained(save_path)
    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        most_recent_ckpts_paths.append(save_path)
        if (
            args["keep_num_ckpt"] is not None
            and len(most_recent_ckpts_paths) > args["keep_num_ckpt"]
        ):
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            shutil.rmtree(ckpt_to_be_removed)


def rollout(args, model, ref_model, tokenizer, batch, iter=None):
    model.eval()
    # batch must contain:
    # batch["input_ids"], batch["attention_mask"], batch["target"], batch["penalty"]
    with torch.no_grad():
        completed_tensors = accelerator.unwrap_model(model).generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            temperature=args["temperature"],
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args["max_gen_length"],
        )

    # pad at the end
    completed_tensors = pad_across_processes(
        completed_tensors, dim=1, pad_index=tokenizer.pad_token_id
    )

    # mask no padding (input output only)
    no_padding_mask = completed_tensors != tokenizer.pad_token_id
    # mask for model input (prompt only, no padding)
    input_mask = pad_across_processes(batch["attention_mask"], dim=1, pad_index=0)
    # pad with 0s to size of completed tensors
    padding = (0, completed_tensors.shape[1] - input_mask.shape[1])
    input_mask = torch.nn.functional.pad(input_mask, padding, value=0)
    # mask for model output (output only, no padding)
    output_mask = no_padding_mask & ~input_mask
    # mask for effective cot tokens (output only, no padding, no answer trigger, no answer)
    # start with copy of output mask
    effective_cot_mask = output_mask.clone()

    input_mask_input = tokenizer.batch_decode(
        torch.where(
            input_mask.bool(), completed_tensors, torch.tensor(tokenizer.pad_token_id)
        ),
        skip_special_tokens=False,
    )

    output_mask_input = tokenizer.batch_decode(
        torch.where(
            output_mask.bool(), completed_tensors, torch.tensor(tokenizer.pad_token_id)
        ),
        skip_special_tokens=False,
    )

    # Evaluate score
    completed_texts = tokenizer.batch_decode(
        completed_tensors, skip_special_tokens=True
    )

    # completed_texts_with_special = tokenizer.batch_decode(
    #     completed_tensors, skip_special_tokens=False
    # )

    generated_texts = extract_completion_batch(completed_texts)
    answers = extract_answer_cot_batch(generated_texts)

    correctness = []
    for i, (answer, target) in enumerate(zip(answers, batch["target"])):
        effective_cot_length = torch.sum(output_mask[i]).item()

        # tokenize final answer and trigger to substract from cot length
        # subtract answer length from cot only if answer trigger is present at least two times
        answer_length = -1
        if answer_trigger in generated_texts[i]:
            answer_length = len(tokenizer(answer_trigger + answer)["input_ids"])
            assert (
                effective_cot_length >= answer_length
            ), f"input Mask'{input_mask_input[i]}' Output Mask '{output_mask_input[i]}' '{completed_texts[i]}' '{generated_texts[i]}' {effective_cot_length} '{answer}' {answer_length}"

            output_mask_indices = output_mask[i].nonzero().squeeze()
            effective_cot_mask[i, output_mask_indices[-answer_length:]] = 0
        else:
            pass
            # accelerator.print(f"item {i}: answer trigger not found...")

        reward = compare_and_calculate_reward(generated_texts[i], target)
        correctness.append(reward)

        if (
            accelerator.is_main_process
            and args["wandb_log"]
            and iter
            and iter % 100 == 0
        ):
            accelerator.print(f"---batch item {i}---")
            accelerator.print(
                f"prompt length: {input_mask[i].sum().item()}\n"
                f"completion length: {output_mask[i].sum().item()}\n"
                f"no padding length: {no_padding_mask[i].sum().item()}\n"
                f"completed text: {completed_texts[i]}\n"
                f"target: {target}\n"
                f"cot length: {torch.sum(effective_cot_mask[i]).item()}\n"
                f"extracted answer: {answer}\n"
                f"answer len: {answer_length}\n"
                f"reward: {reward}\n"
            )

    score_rew = torch.zeros(completed_tensors.shape, device=completed_tensors.device)
    # always reward the last token (eos) or any token in case of early stopping
    last_completed_token = [torch.nonzero(x).max().item() for x in output_mask]
    score_rew[:, last_completed_token] = torch.tensor(
        correctness, device=completed_tensors.device, dtype=torch.float32
    )

    max_gen_length_penalty_rew = torch.zeros(
        completed_tensors.shape, device=completed_tensors.device
    )
    reached_max_gen_length = sum(output_mask, dim=1) >= args["max_gen_length"]
    max_gen_length_penalty_rew[:, last_completed_token] = (
        reached_max_gen_length.float() * REWARD_MAX_GEN_LENGTH
    )

    penalties = torch.tensor(batch["penalty"], device=completed_tensors.device)
    cot_penalty_rew = torch.zeros(
        completed_tensors.shape, device=completed_tensors.device
    )  # (bs, seqlen)
    cot_penalty_rew = -penalties[:, None] * effective_cot_mask

    # add cot_penaly_reward only after X steps (linearly increasing in Y steps)
    start_penalty_after = args["start_penalty_after"]
    penalty_warmup_steps = args["penalty_warmup_steps"]
    cot_penalty_rew *= np.clip(
        (iter - start_penalty_after) / penalty_warmup_steps, min=0.0, max=1.0
    )

    with torch.no_grad():
        # Get old logprob and val
        lm_logits, _, val = model(
            input_ids=completed_tensors, attention_mask=no_padding_mask
        )

        # TODO: understand why cut last token for each sequence???
        # what happens otherwise?
        old_logprob = logprobs_from_logits(
            lm_logits, labels=completed_tensors
        )  # (bs, seqlen-1)

        # print(
        #     f"old logprob shape: {old_logprob.shape}, lm_logits shape: {lm_logits.shape}, val shape: {val.shape}"
        # )

        # Get the ref model logprob
        ref_logprob = None
        if ref_model is not None:
            ref_lm_logits, _, _ = ref_model(
                input_ids=completed_tensors, attention_mask=no_padding_mask
            )
            ref_logprob = logprobs_from_logits(
                ref_lm_logits, labels=completed_tensors
            )  # (bs, seqlen-1)

    kl_rew = torch.zeros(
        completed_tensors.shape, device=completed_tensors.device
    )  # (bs, seqlen)
    props = None
    if ref_logprob is not None:
        # Original implementation
        # kl = old_logprob - ref_logprob  # (bs, seqlen-1)
        # # square the kl divergence elementwise
        # kl = kl**2
        # kl = kl.float() * output_mask

        # other implementation
        props = F.softmax(lm_logits, dim=-1)
        ref_props = F.softmax(ref_lm_logits, dim=-1)
        kl = props * (torch.log(props) - torch.log(ref_props))
        kl = kl.sum(dim=-1)
        kl = kl * output_mask

        # same in both implementations
        kl_rew = -kl  # NOTE the minus sign
        kl_coef = args["kl_coef"]
        kl_rew *= kl_coef

    rew = score_rew + cot_penalty_rew + kl_rew + max_gen_length_penalty_rew

    # TODO: fix lambda gamma error. They are used with swapped values
    gamma = args["gamma"]
    lam = args["lam"]

    # Process val ret adv logprob
    val = val.float() * output_mask
    adv = torch.zeros(rew.shape, device=completed_tensors.device)  # (bs, seqlen)
    # original implementation
    for i in range(len(rew)):
        cur_rew, cur_val = rew[i], val[i]
        cur_delta = -cur_val[:-1] + cur_rew[:-1] + gamma * cur_val[1:]
        cur_adv = discount_cumsum(cur_delta.cpu().numpy(), discount=gamma * lam)
        cur_adv = torch.tensor(cur_adv.copy(), device=completed_tensors.device)
        cur_adv *= output_mask[i][:-1]
        adv[i][:-1] = cur_adv

    # other implementation
    # last_gaelam = 0
    # for t in reversed(range(rew.size(1))):
    #     if t == rew.size(1) - 1:
    #         nextnonterminal = 1.0 - output_mask[:, t]
    #         delta = rew[:, t] + gamma * val[:, t] * nextnonterminal - val[:, t]
    #         adv[:, t] = last_gaelam = delta
    #     else:
    #         nextnonterminal = 1.0 - output_mask[:, t]
    #         delta = rew[:, t] + gamma * val[:, t] * nextnonterminal - val[:, t]
    #         adv[:, t] = last_gaelam = delta + gamma * lam * nextnonterminal * last_gaelam

    # original implementation:
    ret = adv + val  # (bs, seqlen)

    # other implementation: returns = rewards[:, :-1] + gamma * next_values
    # must be in shape (bs, seqlen)
    # ret = torch.zeros(rew.shape, device=completed_tensors.device)
    # ret[:, :-1] = rew[:, :-1] + gamma * val[:, 1:]

    old_logprob = old_logprob * output_mask
    cot_lengths = torch.sum(effective_cot_mask, dim=1)

    model.train()
    return (
        completed_tensors,
        no_padding_mask,
        output_mask,
        rew,
        score_rew,
        kl_rew,
        ret,
        correctness,
        val,
        old_logprob,
        adv,
        generated_texts,
        cot_lengths,
        props,
    )


def train_one_epoch(
    args,
    model,
    ref_model,
    train_dataloader,
    optimizer,
    scheduler,
    tokenizer,
    global_step,
    global_iter_num,
    test_dataloader,
    prefix,
    epoch,
    best_eval_log_dict,
    summary_log_dict,
    most_recent_ckpts_paths,
):
    model_dir = args["model_dir"]
    clip_grad_norm = args.get("clip_grad_norm", None)
    vf_coef = args["vf_coef"]
    mini_batch_size_per_gpu = args["mini_batch_size"]
    ppo_epochs = args["ppo_epochs"]
    evaluating_step_freq = args.get("evaluating_step_freq", None)
    logging_step_freq = args.get("logging_step_freq", None)
    saving_step_freq = args.get("saving_step_freq", None)
    epoch_result_dict = defaultdict(list)
    for batch in tqdm(
        train_dataloader,
        total=len(train_dataloader),
        disable=not accelerator.is_main_process,
        desc="Train Loop",
    ):
        result_dict = defaultdict(list)
        # Do rollout first
        (
            model_input_ids,
            model_attention_mask,
            mask,
            rew,
            score_rew,
            kl_rew,
            ret,
            correctness,
            val,
            old_logprob,
            adv,
            generated_texts,
            cot_lengths,
            old_props,
        ) = rollout(args, model, ref_model, tokenizer, batch, iter=global_iter_num)
        torch.distributed.barrier()
        # preprocess
        if args["adv_whitening"] == "global":
            adv = allgather_masked_whiten(adv, mask)  # (mini_bs, seqlen)
        elif args["adv_whitening"] == "local":
            adv = masked_whiten(adv, mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_size_per_gpu = len(batch["input_ids"])
        train_stats = {}
        if accelerator.is_main_process and args["wandb_log"]:
            reward_sums = torch.sum(rew, dim=1)
            extracted_ans = extract_answer_cot_batch(generated_texts)
            data = {
                "dataset": batch["dataset_name"],
                "input": batch["formatted_input"],
                "generation": generated_texts,
                "extracted answer": extracted_ans,
                "correct answer": batch["target"],
                "answer score": correctness,
                "total reward": reward_sums,
            }
            # # sort by length of thinking, descending
            # data = sorted(data, key=lambda x: len(x[2]), reverse=True)
            table = wandb.Table(
                data=list(zip(*data.values())), columns=list(data.keys())
            )
            wandb.log({"thinking": table}, step=global_iter_num)

            # create dataframe with columns dataset, cot length and score
            # group by dataset and calculate mean cot length
            # log mean cot length per dataset
            # cot_lengths = allgather(cot_lengths)
            dataset_cot_lengths = defaultdict(list)
            for dataset, cot_len, score in zip(
                batch["dataset_name"], cot_lengths, correctness
            ):
                dataset_cot_lengths[dataset].append(
                    {"cot_length": cot_len.cpu(), "score": score}
                )
            dataset_metrics = {
                dataset: {
                    "mean_cot_length": np.mean([x["cot_length"] for x in metrics]),
                    "mean_score": np.mean([x["score"] for x in metrics]),
                }
                for dataset, metrics in dataset_cot_lengths.items()
            }
            wandb.log({"dataset_metrics": dataset_metrics}, step=global_iter_num)
            # log penalty and cot_length
            wandb.log(
                {
                    "penalty/cot-length": [
                        {"penalty": penalty, "cot_length": cot_len}
                        for penalty, cot_len in zip(batch["penalty"], cot_lengths)
                    ]
                },
                step=global_iter_num,
            )

        for _ in range(ppo_epochs):
            perms = torch.randperm(batch_size_per_gpu)
            for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
                b_inds = perms[mini_idx : mini_idx + mini_batch_size_per_gpu]
                # Subset to batch

                cur_val = val[b_inds].contiguous()  # mini_bs x seqlen
                cur_old_logprob = old_logprob[b_inds].contiguous()  # mini_bs x seqlen

                cur_mask = mask[b_inds].contiguous()  # mini_bs x seqlen
                cur_rew = rew[b_inds].contiguous()  # mini_bs x seqlen
                cur_score_rew = score_rew[b_inds].contiguous()  # mini_bs x seqlen
                cur_kl_rew = (
                    None if kl_rew is None else kl_rew[b_inds].contiguous()
                )  # mini_bs x seqlen
                cur_ret = ret[b_inds].contiguous()  # mini_bs x seqlen
                cur_adv = adv[b_inds].contiguous()  # mini_bs x seqlen
                cur_model_input_ids = model_input_ids[
                    b_inds
                ].contiguous()  # mini_bs x seqlen
                cur_model_attention_mask = model_attention_mask[
                    b_inds
                ].contiguous()  # mini_bs x seqlen

                resp_len_per_sample = torch.clamp(
                    torch.sum(cur_mask, dim=1), min=1.0
                )  # (mini_bs,)
                cur_query_mask = torch.logical_xor(
                    cur_mask, cur_model_attention_mask
                )  # (mini_bs, seqlen)
                query_len_per_sample = torch.clamp(
                    torch.sum(cur_query_mask, dim=1), min=1.0
                )  # (mini_bs,)
                # thinking starts after end of query and unil answer_trigger
                # thinking_len_per_sample = torch.clamp(

                # Preprocess advantage and get metrics
                cur_mask = cur_mask.type(cur_adv.dtype).contiguous()
                mean_adv, var_adv = masked_mean(cur_adv, cur_mask), masked_var(
                    cur_adv, cur_mask
                )

                # TODO: WHY A SECOND FORWARD PASS?
                # Forward current model
                # TODO: why model.eval()???? Shouldn't it be model.train()?

                model.eval()
                lm_logits, _, vpreds = model(
                    input_ids=cur_model_input_ids,
                    attention_mask=cur_model_attention_mask,
                )

                logprob = logprobs_from_logits(
                    lm_logits, cur_model_input_ids
                )  # (mini_bs, seqlen-1)

                # Compute losses

                # original implementation

                # loss = 0
                # # policy gradient loss
                # ratio = torch.exp(logprob - cur_old_logprob)
                # pg_losses = -cur_adv * ratio
                # pg_losses2 = -cur_adv * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                # pg_loss = (
                #     (torch.max(pg_losses, pg_losses2) * cur_mask).sum(dim=-1)
                #     / resp_len_per_sample
                # ).mean()
                # # pg_loss = (torch.max(pg_losses, pg_losses2) * cur_mask]).sum() / cur_mask.sum()
                # # pg_loss = (-logprob * cur_ret[:,:-1]).sum() / cur_mask.sum()

                # # value loss
                # vpredclipped = torch.max(
                #     torch.min(vpreds, cur_val + 0.2), cur_val - 0.2
                # )
                # vf_losses1 = (vpreds - cur_ret) ** 2
                # vf_losses2 = (vpredclipped - cur_ret) ** 2
                # vf_loss = (
                #     0.5
                #     * (
                #         (torch.max(vf_losses1, vf_losses2) * cur_mask).sum(dim=-1)
                #         / resp_len_per_sample
                #     ).mean()
                # )
                # # vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum() / cur_mask.sum())

                # # total loss
                # loss += pg_loss + vf_coef * vf_loss

                # other implementation
                cur_old_props = old_props[b_inds].contiguous()
                cur_props = F.softmax(lm_logits, dim=-1).contiguous()
                cur_adv = cur_adv.bfloat16()
                cur_ret = cur_ret.bfloat16()
                # policy gradient loss
                ratio = torch.exp(torch.log(cur_props) - torch.log(cur_old_props))
                ratio = ratio.mean(dim=-1)
                CLIP_PARAM = 0.2
                surr1 = ratio * cur_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * cur_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                # value loss
                vf_loss = F.mse_loss(vpreds, cur_ret)
                loss = pg_loss + vf_coef * vf_loss

                torch.distributed.barrier()

                # token related metrics
                mean_query_len = torch.mean(allgather(torch.mean(query_len_per_sample)))
                std_query_len = torch.mean(allgather(torch.std(query_len_per_sample)))
                mean_resp_len = torch.mean(allgather(torch.mean(resp_len_per_sample)))
                std_resp_len = torch.mean(allgather(torch.std(resp_len_per_sample)))

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # value related metrics
                # vf_expl_var_num = torch.var(torch.masked_select(cur_ret - vpreds, cur_mask.bool()))
                # vf_expl_var_dem = torch.var(torch.masked_select(cur_ret, cur_mask.bool()))
                vf_expl_var_num = masked_var(cur_ret - vpreds, cur_mask)
                vf_expl_var_dem = masked_var(cur_ret, cur_mask)
                vf_expl_var = 1.0 - vf_expl_var_num / (vf_expl_var_dem + 1e-8)
                vf_expl_var = max(
                    -1.0, vf_expl_var.item()
                )  # the truncated value suffices
                mean_vpred = masked_mean(vpreds, cur_mask)
                mean_return = masked_mean(cur_ret, cur_mask)
                mean_reward = masked_mean(cur_rew, cur_mask)
                mean_score_reward = masked_mean(cur_score_rew, cur_mask)
                mean_kl_reward = (
                    0.0 if cur_kl_rew is None else masked_mean(cur_kl_rew, cur_mask)
                )
                mean_kcxkl_reward = args["kl_coef"] * mean_kl_reward

                # policy related metrics
                mean_ratio = masked_mean(ratio, cur_mask)
                # mean_adv = masked_mean(cur_adv, cur_mask)
                mean_logprob = masked_mean(logprob, cur_mask)
                # sequence-level kl
                mean_seq_kl = -1.0
                if cur_kl_rew is not None:
                    cur_kl = -cur_kl_rew
                    seq_kl = torch.sum(cur_kl * cur_mask, dim=1)  # (mini_bs,)
                    mean_seq_kl = torch.mean(seq_kl)

                # Update
                epoch_result_dict["loss"].append(loss.item())

                if accelerator.distributed_type == "DEEPSPEED":
                    accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                    total_grad_norm = 0.0
                    for n, p in model.named_parameters():
                        cur_grad = deepspeed.utils.safe_get_full_grad(p).view(-1)
                        cur_grad_norm_sqrt = torch.norm(cur_grad, 2)
                        if cur_grad_norm_sqrt < 1e-8:
                            accelerator.print(
                                f"{n} grad_norm_sqrt: {cur_grad_norm_sqrt}"
                            )
                        total_grad_norm += cur_grad_norm_sqrt**2
                    total_grad_norm = total_grad_norm**0.5
                    # Deepspeed's `engine.step` performs the following operations:
                    # - gradient accumulation check
                    # - gradient clipping
                    # - optimizer step
                    # - zero grad
                    # - checking overflow
                    # - lr_scheduler step (only if engine.lr_scheduler is not None)
                    accelerator.deepspeed_engine_wrapped.engine.step()
                else:
                    accelerator.backward(loss)
                    total_grad_norm = -1.0
                    if clip_grad_norm is not None:
                        total_grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), clip_grad_norm
                        )

                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

                # Update running stats
                n_correct, total = do_gather([sum(correctness), len(correctness)])
                train_stats["acc"] = n_correct / total
                train_stats["ncor"] = n_correct
                train_stats["total"] = total
                train_stats["pg_loss"] = pg_loss.item()
                train_stats["vf_loss"] = vf_loss.item()
                train_stats["vf_expl_var"] = vf_expl_var

                for k, v in train_stats.items():
                    result_dict[k].append(v)

                total_param_norm = 0.0
                if accelerator.distributed_type == "DEEPSPEED":
                    for n, p in model.named_parameters():
                        cur_param = deepspeed.utils.safe_get_full_fp32_param(p).view(-1)
                        total_param_norm += torch.norm(cur_param, 2) ** 2
                    total_param_norm = total_param_norm**0.5
                else:
                    total_param_norm = torch.norm(
                        torch.cat([p.view(-1) for p in model.parameters()]),
                        p=2,  # L2 norm
                    )
                # logging
                if accelerator.is_main_process and args["wandb_log"]:
                    wandb.log(
                        {
                            "nn/total_grad_norm": total_grad_norm,
                            "nn/total_param_norm": total_param_norm,
                            "nn/lr": scheduler.get_last_lr()[0],
                            "acc/acc": train_stats["acc"],
                            "acc/ncor": train_stats["ncor"],
                            "acc/total": train_stats["total"],
                            "loss/loss:": loss,
                            "loss/pg_loss": pg_loss,
                            "loss/vf_loss": vf_loss,
                            "tokens/mean_query_len": mean_query_len,
                            "tokens/std_query_len": std_query_len,
                            "tokens/mean_resp_len": mean_resp_len,
                            "tokens/std_resp_len": std_resp_len,
                            "policy/mean_ratio": mean_ratio,
                            "policy/mean_adv": mean_adv,
                            "policy/var_adv": var_adv,
                            "policy/mean_logprob": mean_logprob,
                            "policy/mean_seq_kl": mean_seq_kl,
                            "value/vf_expl_var": vf_expl_var,
                            "value/mean_vpred": mean_vpred,
                            "value/mean_return": mean_return,
                            "value/mean_reward": mean_reward,
                            "value/mean_score_reward": mean_score_reward,
                            "value/mean_kl_reward": mean_kl_reward,
                            "value/mean_kcxkl_reward": mean_kcxkl_reward,
                        },
                        step=global_iter_num,
                    )
                # Update iter num
                # torch.distributed.barrier()
                global_iter_num += 1

        scheduler.step()
        global_step += 1
        # Step update metric
        epoch_result_dict["loss"].append(loss.item())
        for k, v in train_stats.items():
            epoch_result_dict[k].append(v)

        # Step evaluating
        eval_log_dict = {}
        is_best = False
        if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
            evaluate_result_dict = {
                f"Eval.Gen.{k}": v
                for k, v in evaluate_generation(
                    args, model, test_dataloader, tokenizer
                ).items()
            }
            eval_log_dict.update(evaluate_result_dict)
            if eval_log_dict["Eval.Gen.value_accuracy"] > best_eval_log_dict.get(
                "Eval.Gen.value_accuracy_best", -1
            ):
                is_best = True
                best_eval_log_dict["Eval.Gen.value_accuracy_best"] = eval_log_dict[
                    "Eval.Gen.value_accuracy"
                ]
                if "Eval.Gen.value_accuracy" not in summary_log_dict:
                    summary_log_dict["Eval.Gen.value_accuracy"] = []
                summary_log_dict["Eval.Gen.value_accuracy"].append(
                    eval_log_dict["Eval.Gen.value_accuracy"]
                )

        # Step logging
        train_log_dict = {}
        if logging_step_freq is not None and global_step % logging_step_freq == 0:
            train_log_dict = {
                f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                for k, v in epoch_result_dict.items()
            }

        if eval_log_dict or train_log_dict:
            log_dict = {
                "lr": scheduler.get_last_lr()[0],
                **train_log_dict,
                **eval_log_dict,
                **best_eval_log_dict,
            }
            if accelerator.is_main_process and args["wandb_log"]:
                wandb.log(log_dict, step=global_iter_num)
                log_dict = {
                    "wandb": args["wandb_project"] + "|" + args["wandb_run_name"],
                    **log_dict,
                }
            log_dict = {
                k: f"{v:.5g}" if isinstance(v, float) else v
                for k, v in log_dict.items()
            }
            accelerator.print(
                f"{prefix}[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}"
            )

        # Step saving
        if saving_step_freq is not None and global_step % saving_step_freq == 0:
            if is_best:
                save_path = os.path.join(model_dir, f"best")
                do_checkpoint(args, model, tokenizer, save_path)
            if args["keep_num_ckpt"] > 0:
                save_path = os.path.join(model_dir, f"global_step_{str(global_step)}")
                do_checkpoint(
                    args, model, tokenizer, save_path, most_recent_ckpts_paths
                )

        # Keep only max_record items
        for k, v in epoch_result_dict.items():
            if len(v) > 1:
                epoch_result_dict[k] = v[-1:]

    # Metric summary:
    epoch_result_dict = {
        k: (sum(v) / len(v) if isinstance(v, list) else v)
        for k, v in epoch_result_dict.items()
    }
    return epoch_result_dict, global_step, global_iter_num


def evaluate_generation(args, model, dataloader, tokenizer):
    model.eval()
    predictions = []
    targets = []
    for batch in tqdm(
        dataloader,
        total=len(dataloader),
        disable=not accelerator.is_main_process,
        desc="Evaluation Gen Loop",
    ):
        output_ = accelerator.unwrap_model(model).generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=args["max_gen_length"],
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=1,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_.sequences
        generated_ids = pad_across_processes(
            generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True
        )

        gathered = accelerator.gather_for_metrics([generated_ids, batch["target"]])
        generated_ids, target = [], []

        if accelerator.is_main_process:
            generated_ids = [item for sublist in gathered[::2] for item in sublist]
            target = [item for sublist in gathered[1::2] for item in sublist]

        preds = [
            tokenizer.decode(
                g.cpu().numpy().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            for g in generated_ids
        ]
        predictions.extend(preds)
        targets.extend(target)

    if accelerator.is_main_process and accelerator.is_local_main_process:
        pred_cots = extract_completion_batch(predictions)
        pred_answers = extract_answer_cot_batch(pred_cots)
        results = []
        corr_value = 0
        for pred_cot, answer, target in zip(pred_cots, pred_answers, targets):
            cur_res = {
                "prediction_cot": pred_cot,
                "prediction_value": answer,
                "target": target,
            }
            results.append(cur_res)
            corr_value += compare_and_calculate_reward(answer, target)

        res_path = args["model_dir"].rstrip("/") + "/" + "_res.json"
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)

        value_accuracy = corr_value / len(results) * 100
        accelerator.print(f"[Eval Info] value_accuracy: {value_accuracy:.5g}%")
        value_accuracy = torch.FloatTensor([value_accuracy]).to(accelerator.device)
    else:
        value_accuracy = torch.FloatTensor([-1.0]).to(accelerator.device)
    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]

    # Metric summary:
    model.train()
    return {"value_accuracy": value_accuracy}


def main(args):
    set_seed(args["seed"] + accelerator.process_index)

    if accelerator.is_main_process and args["wandb_log"]:
        wandb.init(project=args["wandb_project"], name=args["wandb_run_name"])
        wandb.config.update(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args["tokenizer_name_or_path"], use_fast=True, padding_side="left"
    )
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2

    train_dataloader, _, test_small_dataloader = prepare_datasets_and_data_loaders(
        args, tokenizer
    )

    MODEL_CLASS = AutoModelForCausalLMWithValueHead
    model = MODEL_CLASS.from_pretrained(
        args["model_name_or_path"], trust_remote_code=True
    )

    # initialize ref model (if any)
    ref_model = None
    if args["ref_model_name_or_path"]:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args["ref_model_name_or_path"], trust_remote_code=True
        )

    # optimizer
    n_epochs = args["n_epochs"]
    num_training_steps = len(train_dataloader) // accelerator.num_processes * n_epochs
    warmup_step = (
        args["warmup_step"]
        if args["warmup_step"] is not None and args["warmup_step"] >= 0
        else int(0.1 * num_training_steps)
    )
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": args["weight_decay"],
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in ["bias", "LayerNorm.weight"])
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args["learning_rate"], eps=1e-8
    )
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step
    )
    model, optimizer, train_dataloader, test_small_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_small_dataloader
    )
    if ref_model is not None:
        if accelerator.distributed_type == "DEEPSPEED":
            ref_model = prepare_deepspeed_ref_model(ref_model)
        else:
            ref_model = accelerator.prepare(ref_model)

    global_step = 0
    global_iter_num = 0
    evaluating_epoch_freq = args["evaluating_epoch_freq"]
    logging_epoch_freq = args["logging_epoch_freq"]
    saving_epoch_freq = args["saving_epoch_freq"]
    model_dir = args["model_dir"]
    best_eval_log_dict = {}
    summary_log_dict = {}
    os.makedirs(model_dir, exist_ok=True)
    most_recent_ckpts_paths = []
    with tqdm(range(1, n_epochs + 1), total=n_epochs, disable=False) as t:
        for epoch in t:
            kwargs = {
                "args": args,
                "model": model,
                "ref_model": ref_model,
                "train_dataloader": train_dataloader,
                "test_dataloader": test_small_dataloader,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "global_step": global_step,
                "global_iter_num": global_iter_num,
                "tokenizer": tokenizer,
                "prefix": "",
                "epoch": epoch,
                "best_eval_log_dict": best_eval_log_dict,
                "summary_log_dict": summary_log_dict,
                "most_recent_ckpts_paths": most_recent_ckpts_paths,
            }
            train_epoch_result_dict, global_step, global_iter_num = train_one_epoch(
                **kwargs
            )

            eval_log_dict = {}
            is_best = False
            if evaluating_epoch_freq is not None and epoch % evaluating_epoch_freq == 0:
                evaluate_result_dict = {
                    f"Eval.Gen.{k}": v
                    for k, v in evaluate_generation(
                        args,
                        model,
                        test_small_dataloader,
                        tokenizer,
                    ).items()
                }
                eval_log_dict.update(evaluate_result_dict)
                if eval_log_dict["Eval.Gen.value_accuracy"] > best_eval_log_dict.get(
                    "Eval.Gen.value_accuracy_best", -1
                ):
                    is_best = True
                    best_eval_log_dict["Eval.Gen.value_accuracy_best"] = eval_log_dict[
                        "Eval.Gen.value_accuracy"
                    ]
                    if "Eval.Gen.value_accuracy" not in summary_log_dict:
                        summary_log_dict["Eval.Gen.value_accuracy"] = []
                    summary_log_dict["Eval.Gen.value_accuracy"].append(
                        eval_log_dict["Eval.Gen.value_accuracy"]
                    )

            train_log_dict = {}
            if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                train_log_dict = {
                    f"T.{k}": sum(v) / len(v) if isinstance(v, list) else v
                    for k, v in train_epoch_result_dict.items()
                }

            if eval_log_dict or train_log_dict:
                log_dict = {
                    "lr": scheduler.get_last_lr()[0],
                    **train_log_dict,
                    **eval_log_dict,
                    **best_eval_log_dict,
                }
                if accelerator.is_main_process and args["wandb_log"]:
                    wandb.log(log_dict, step=global_iter_num)
                    log_dict = {
                        "wandb": args["wandb_project"]
                        + "|"
                        + args["wandb_run_name"]
                        + "|"
                        + wandb.run.id,
                        **log_dict,
                    }

                log_dict = {
                    k: f"{v:.5g}" if isinstance(v, float) else v
                    for k, v in log_dict.items()
                }
                accelerator.print(
                    f"[Epoch={epoch}/{args['n_epochs']}, Step={global_step}] {log_dict}"
                )

            if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                if is_best:
                    save_path = os.path.join(model_dir, f"best")
                    do_checkpoint(args, model, tokenizer, save_path)
                #
                if args["keep_num_ckpt"] > 0:
                    # save the checkpoint only if keep num ckpt > 0
                    save_path = os.path.join(
                        args["model_dir"],
                        f"global_step_{str(global_step)}_epoch_{epoch}",
                    )
                    do_checkpoint(
                        args, model, tokenizer, save_path, most_recent_ckpts_paths
                    )

    return


if __name__ == "__main__":
    from transformers import HfArgumentParser

    NONE_INT = -100
    NONE_STR = "None"

    @dataclass
    class Arguments:
        model_name_or_path: str
        tokenizer_name_or_path: str
        model_dir: str
        train_file: str
        test_file: str
        batch_size: int = field(default=8)
        mini_batch_size: int = field(default=8)
        eval_batch_size: int = field(default=8)
        ppo_epochs: int = field(default=1)
        n_epochs: int = field(default=40)
        num_workers: int = field(default=8)
        learning_rate: float = field(default=2e-5)
        weight_decay: float = field(default=1e-6)
        warmup_step: int = field(default=0)
        clip_grad_norm: float = field(default=1)
        vf_coef: float = field(default=1.0)
        kl_coef: float = field(default=0.1)
        gamma: float = field(default=0.98)
        lam: float = field(default=0.95)
        ref_model_name_or_path: str = field(default="")
        evaluating_epoch_freq: int = field(default=1)
        logging_epoch_freq: int = field(default=1)
        saving_epoch_freq: int = field(default=1000)
        evaluating_step_freq: int = field(default=NONE_INT)
        logging_step_freq: int = field(default=NONE_INT)
        logging_seq_str_step_freq: int = field(default=NONE_INT)
        logging_values_step_freq: int = field(default=NONE_INT)
        saving_step_freq: int = field(default=NONE_INT)
        seed: int = field(default=42)
        max_input_length: int = field(default=700)
        max_gen_length: int = field(default=700)
        keep_num_ckpt: int = field(default=5)
        # wandb stuff
        wandb_log: bool = field(default=False)
        wandb_project: str = field(default="tmp_anvfupsadfn")
        wandb_run_name: str = field(default="default_run_name")
        ###
        adv_whitening: str = field(default="global")
        ### new!
        temperature: float = field(default=1.0)
        start_penalty_after: int = field(default=0)
        penalty_warmup_steps: int = field(default=100)

    parser = HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = asdict(args)
    for k, v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]
    )
    accelerator.print(args)
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    main(args)
