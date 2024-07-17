from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset
from datetime import timedelta
from functools import partial
import json
import os
from src.utils import (
    set_seed,
    discount_cumsum,
    do_gather,
    allgather,
    allgather_masked_whiten,
)
from tqdm import tqdm
import torch
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
from matplotlib import cm
import zlib
import base64

load_dotenv()

cmap = cm.get_cmap("RdYlGn")

tqdm = partial(tqdm, ncols=0, leave=False)

IN_COL = "final_input"
OUT_COL = "final_target"
TASK_COL = "dataset"
VISUALIZATION_LINK = "https://jorineg.github.io/js-plots?data="

penalty_trigger = "Penalty:"
problem_prefix = "Problem:"
answer_trigger = "ANSWER: "
cot_trigger = f"BOT: "
instruction = ""

answer_trigger_token_count = -1


def format_input_batch(
    input_batch, enable_penalty=True, penalties=None, answer_trigger="", outputs=None
):
    if penalties is None:
        penalties = [0.001] * len(input_batch)

    if outputs is None:
        outputs = [""] * len(input_batch)

    penalty_announce = [""] * len(input_batch)
    if enable_penalty:
        penalty_announce = [f"{penalty_trigger} {penalty:.5f}" for penalty in penalties]

    return [
        f"{penalty}{input}\n{cot_trigger}{answer_trigger}{output}"
        for input, output, penalty in zip(input_batch, outputs, penalty_announce)
    ]


# takes a batch of input and completion strings
# returns a list of completion strings
def extract_completion_batch(input_and_completion_batch):
    cot_trigger_count_in_instructions = instruction.count(cot_trigger)
    splitted = [res.split(cot_trigger) for res in input_and_completion_batch]
    return [
        cot_trigger.join(split[cot_trigger_count_in_instructions + 1 :])[1:]
        for split in splitted
    ]


# assumes to get back only generated tokens
# returns empty string if no answer trigger is found
def extract_answer_cot_batch(answer_cot_batch):
    splitted_batch = [res.split(answer_trigger) for res in answer_cot_batch]
    return [
        answer_trigger.join(splitted[1:]) if len(splitted) >= 2 else ""
        for splitted in splitted_batch
    ]


def check_answer(extracted_ans, target_answer):
    if extracted_ans.strip() == target_answer:
        return args["reward_correct"]
    if extracted_ans.strip().startswith(target_answer):
        return args["reward_starts_correct"]
    return 0


def compare_and_calculate_reward(cot, target_answer):
    reward = 0
    if answer_trigger in cot:
        extracted_ans = extract_answer_cot_batch([cot])[0]
        reward = check_answer(extracted_ans, target_answer)
        if reward == 0:
            reward = args["reward_contains_answer_trigger"]
    return reward


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
                # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
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
        answer_trigger_token_count = len(tokenizer(answer_trigger)["input_ids"])
        raw_dataset = load_dataset("jeggers/CoT-Collection")

        # filter out first 3000 samples
        raw_dataset["train"] = raw_dataset["train"].select(
            list(range(3000, len(raw_dataset["train"])))
        )

        accelerator.print("Raw data:", raw_dataset)

        accelerator.print(f"Using instruction: '{instruction}'")
        accelerator.print(f"Using cot_trigger: '{cot_trigger}'")
        accelerator.print(f"Using answer_trigger: '{answer_trigger}'")

        def tokenize_fn(batch, tokenizer):
            assert tokenizer.eos_token_id is not None, (
                tokenizer.eos_token_id,
                tokenizer.eos_token,
            )

            formatted_batch = format_input_batch(batch[IN_COL], enable_penalty=False)
            tokenized_batch = tokenizer(formatted_batch, add_special_tokens=False)

            tokenized_batch["question"] = batch[IN_COL]
            tokenized_batch["prefix_text"] = formatted_batch
            tokenized_batch["task"] = batch[TASK_COL]
            tokenized_batch["targets"] = batch[OUT_COL]

            return tokenized_batch

        tokenized_dataset = raw_dataset.map(
            tokenize_fn,
            fn_kwargs={"tokenizer": tokenizer},
            batched=True,
            remove_columns=raw_dataset["train"].column_names,
            batch_size=-1,
        )

        # filter for <= max_input_length tokens
        def filter_fn(batch):
            return [
                len(prefix) <= args["max_input_length"] for prefix in batch["input_ids"]
            ]

        tokenized_dataset = tokenized_dataset.filter(
            filter_fn, batched=True, batch_size=-1
        )

        accelerator.print("Processed data:", tokenized_dataset)

        if accelerator.is_main_process and args["wandb_log"]:
            wandb.config.update(
                {
                    "instruction": instruction,
                    "cot_trigger": cot_trigger,
                    "answer_trigger": answer_trigger,
                    "raw_dataset": str(raw_dataset),
                    "tokenized_dataset": str(tokenized_dataset),
                }
            )

    def collate_fn(batch, tokenizer):
        max_prefix_length = max([len(item["input_ids"]) for item in batch])
        prefix_left_padded = []
        prefix_attention_mask_left_padded = []

        for item in batch:
            prefix_left_padded.append(
                [tokenizer.pad_token_id] * (max_prefix_length - len(item["input_ids"]))
                + item["input_ids"]
            )
            prefix_attention_mask_left_padded.append(
                [0] * (max_prefix_length - len(item["attention_mask"]))
                + item["attention_mask"]
            )

        kwargs = {
            "prefix_text": [item["prefix_text"] for item in batch],
            "input_ids": torch.LongTensor(prefix_left_padded),
            "targets": [item["targets"] for item in batch],
            "task": [item["task"] for item in batch],
            "attention_mask": torch.BoolTensor(prefix_attention_mask_left_padded),
        }

        return kwargs

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        pin_memory=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    test_all_dataloader = DataLoader(
        tokenized_dataset["test"],
        shuffle=False,
        batch_size=args["eval_batch_size"],
        num_workers=args["num_workers"],
        pin_memory=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    return train_dataloader, test_all_dataloader


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


def rollout(
    args,
    model,
    ref_model,
    tokenizer,
    batch,
    iter=None,
):
    model.eval()
    with torch.no_grad():
        gen_output = accelerator.unwrap_model(model).generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args["max_gen_length"],
        )
        completed_tensors = gen_output
        completed_tensors = pad_across_processes(
            completed_tensors, dim=1, pad_index=tokenizer.pad_token_id, pad_first=False
        )

    # Evaluate score
    completed_texts = tokenizer.batch_decode(
        completed_tensors.cpu().numpy().tolist(), skip_special_tokens=True
    )
    # completed_texts_special = tokenizer.batch_decode(
    #     completed_tensors.cpu().numpy().tolist(), skip_special_tokens=False
    # )
    programs = extract_completion_batch(completed_texts)

    correctness = []
    token_texts = []
    for i, cot in enumerate(programs):
        target_value = batch["targets"][i]
        reward = compare_and_calculate_reward(cot, target_value)
        token_text = tokenizer.batch_decode(
            completed_tensors[i], skip_special_tokens=False
        )
        correctness.append(reward)
        token_texts.append(token_text)

    # # mask for model input (prompt only, no padding)
    # completion_start_index = batch["input_ids"].shape[1]
    # input_mask = torch.zeros_like(completed_tensors, device=completed_tensors.device)
    # input_mask[:, :completion_start_index] = 1
    # # mask no padding (input output only)
    # no_padding_mask = completed_tensors != tokenizer.pad_token_id
    # input_mask = input_mask & no_padding_mask
    # # mask for model output (output only, no padding)
    # output_mask = no_padding_mask & ~input_mask

    # last_completed_token = torch.tensor(
    #     [torch.nonzero(x).max().item() for x in output_mask],
    #     device=completed_tensors.device,
    # )

    # answer_trigger_end_tokens = []
    # answer_end_token_positions = []
    # for i, target in enumerate(batch["targets"]):
    #     # subtract answer length from cot only if answer trigger is present at least two times
    #     answer_trigger_start_token = -1
    #     answer_end_token = last_completed_token[i]
    #     reward = compare_and_calculate_reward(programs[i], target)
    #     if answer_trigger in programs[i]:
    #         # decode generated text token wise
    #         generated_token_texts = tokenizer.batch_decode(
    #             completed_tensors[i], skip_special_tokens=True
    #         )
    #         text = ""
    #         for idx, token in enumerate(generated_token_texts[completion_start_index:]):
    #             text += token
    #             if answer_trigger in text:
    #                 answer_trigger_start_token = (
    #                     completion_start_index + idx - answer_trigger_token_count
    #                 )
    #                 break
    #         assert (
    #             answer_trigger_start_token != -1
    #         ), f"answer trigger not found in '{generated_token_texts}' '{text}', '{programs[i]}'"
    #         if reward > 0:
    #             text = ""
    #             answer_pos = answer_trigger_start_token + answer_trigger_token_count
    #             for idx, token in enumerate(generated_token_texts[answer_pos:]):
    #                 text += token
    #                 if target in text:
    #                     answer_end_token = answer_pos + idx
    #                     break
    #         # effective_cot_mask[i, answer_trigger_start_token:answer_end_token] = 0
    #     answer_end_token_positions.append(answer_end_token)
    #     answer_trigger_end_tokens.append(
    #         answer_trigger_start_token + answer_trigger_token_count
    #         if answer_trigger_start_token != -1
    #         else 0
    #     )
    model_input_ids = completed_tensors
    model_attention_mask = completed_tensors != tokenizer.pad_token_id
    with torch.no_grad():
        # Get old logprob and val
        lm_logits, _, val = model(
            input_ids=model_input_ids, attention_mask=model_attention_mask
        )
        old_logprob = logprobs_from_logits(
            lm_logits[:, :-1, :], labels=model_input_ids[:, 1:]
        )  # (bs, seqlen-1)

        # Get the ref model logprob
        ref_logprob = None
        if ref_model is not None:
            ref_lm_logits, _, _ = ref_model(
                input_ids=model_input_ids, attention_mask=model_attention_mask
            )
            ref_logprob = logprobs_from_logits(
                ref_lm_logits[:, :-1, :], labels=model_input_ids[:, 1:]
            )  # (bs, seqlen-1)

    # Masking the last prompt token up untils the token before eos_token_id
    prompt_len = batch["input_ids"].size(1)
    mask = torch.zeros_like(model_input_ids, dtype=torch.bool)  # (bs, seqlen)
    mask[:, batch["input_ids"].size(1) - 1 : -1] = 1
    score_rew = np.zeros(mask.shape)  # (bs, seqlen)
    # like scatter_ but in numpy
    # np.put_along_axis(
    #     score_rew, last_completed_token.unsqueeze(1).cpu().numpy(), np.array(correctness).reshape(-1, 1), axis=1
    # )

    # score_rew[:, -2] = np.array(correctness)
    # score_rew = torch.zeros_like(mask, device=mask.device, dtype=torch.float)

    # correctness_tensor = torch.tensor(correctness, device=mask.device, dtype=torch.float32)
    # score_rew = score_rew.scatter_(
    #     1,
    #     last_completed_token.unsqueeze(1),
    #     correctness_tensor.unsqueeze(1),
    # )

    # # back to numpy
    # score_rew = score_rew.cpu().numpy()

    nonzero = (model_input_ids == tokenizer.eos_token_id).nonzero()
    for bidx, tidx in nonzero:
        mask[bidx][tidx + 1 :] = 0
        score_rew[bidx][tidx:] = 0
        score_rew[bidx][tidx] = correctness[bidx]

    # Make the kl reward and the full reward
    kl_rew = None
    rew = score_rew
    if ref_logprob is not None:
        kl = old_logprob - ref_logprob  # (bs, seqlen-1)
        kl = (kl.float() * mask[:, :-1]).cpu().numpy()
        kl_rew = np.zeros(mask.shape)  # (bs, seqlen)
        kl_rew[:, :-1] = -kl  # NOTE the minus sign

        # other implementation uses distribution accross all tokens and not only generated tokens
        # props = torch.nn.functional.softmax(lm_logits, dim=-1)
        # ref_props = torch.nn.functional.softmax(ref_lm_logits, dim=-1)
        # kl = torch.sum(props * (torch.log(props) - torch.log(ref_props)), dim=-1)
        # kl_rew = (-kl * mask).cpu().numpy()

        kl_coef = args["kl_coef"]
        # if iter < 80:
        #     kl_coef = 0.1
        rew = score_rew + kl_coef * kl_rew

    # Process val ret adv logprob
    val = (val.float() * mask).cpu().numpy()
    gamma = args["gamma"]
    lam = args["lam"]
    # ret = np.zeros_like(rew)
    adv = np.zeros_like(rew)
    for i in range(len(rew)):
        cur_rew, cur_val = rew[i], val[i]
        cur_delta = -cur_val[:-1] + cur_rew[:-1] + gamma * cur_val[1:]
        cur_adv = discount_cumsum(cur_delta, discount=gamma * lam)
        cur_adv[: prompt_len - 1] = 0
        adv[i][:-1] = cur_adv

    # lambda_return = GAE + values
    ret = adv + val  # (bs, seqlen)

    rew = torch.tensor(rew, device=mask.device, dtype=old_logprob.dtype) * mask
    score_rew = (
        torch.tensor(score_rew, device=mask.device, dtype=old_logprob.dtype) * mask
    )
    if kl_rew is not None:
        kl_rew = (
            torch.tensor(kl_rew, device=mask.device, dtype=old_logprob.dtype) * mask
        )
    ret = torch.tensor(ret, device=mask.device, dtype=old_logprob.dtype) * mask
    val = torch.tensor(val, device=mask.device, dtype=old_logprob.dtype) * mask
    adv = torch.tensor(adv, device=mask.device, dtype=old_logprob.dtype) * mask
    old_logprob = old_logprob * mask[:, :-1]

    model.train()
    return (
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
        ref_logprob,
        adv,
        programs,
        token_texts,
    )


def compress_text(text):
    compressed = zlib.compress(text.encode("utf-8"))
    b64 = base64.b64encode(compressed)
    return b64.decode("ascii").replace("+", "-").replace("/", "_").replace("=", "")


def round(tensor, places=5):
    return torch.round(tensor * 10**places) / (10**places)


# metrics_dict: generated_texts, tasks, input_texts, targets,
def log_table_metrics(
    metrics_dict,
    tokenizer,
    vpreds,
    rew,
    ret,
    score_rew,
    kl_rew,
    global_iter_num,
    token_texts,
):
    generated_texts = metrics_dict["generation"]
    reward_sums = torch.sum(rew, dim=1)
    extracted_ans = extract_answer_cot_batch(generated_texts)
    cot_lengths = [
        len(tokenizer(g)["input_ids"]) - len(tokenizer(e)["input_ids"])
        for g, e in zip(generated_texts, extracted_ans)
    ]
    links = []
    for i in range(len(token_texts)):
        # tokenized_text = tokenizer.tokenize(generated_texts_special[i])
        # replace G and C with spaces/newlines
        # tokenized_text = [
        #     t.replace("Ġ", " ").replace("Ċ", "\n") for t in tokenized_text
        # ]
        tokenized_text = token_texts[i]
        visualization_metrics = [
            {
                "score reward": round(score_rew[i]).tolist(),
                "kl reward": round(kl_rew[i]).tolist(),
                "reward": round(rew[i]).tolist(),
            },
            {
                "actual value": round(ret[i]).tolist(),
                "predicted value": round(vpreds[i]).tolist(),
            },
        ]
        abs_max = max(torch.max(vpreds[i]), torch.abs(torch.min(vpreds[i])))
        normalized_preds = vpreds[i] / abs_max
        # now we have values between -1 and 1
        # for diverging color map, divide by 2 and add 0.5
        normalized_preds = (normalized_preds / 2 + 0.5).tolist()

        text_colors = [
            f"rgb{tuple(int(255 * x) for x in cmap(r)[:3])}" for r in normalized_preds
        ]
        visualization_obj = {
            "textColors": text_colors,
            "tokens": tokenized_text,
            "metrics": visualization_metrics,
        }
        assert len(tokenized_text) == len(text_colors) and len(text_colors) == len(
            visualization_metrics[0]["score reward"]
        ), (
            len(tokenized_text),
            len(text_colors),
            len(visualization_metrics[0]["score reward"]),
        )
        json_str = json.dumps(visualization_obj)
        json_str = (
            json_str.replace(", ", ",").replace("0.0,", "0,").replace("-0,", "0,")
        )
        base64_str = compress_text(json_str)
        links.append(VISUALIZATION_LINK + base64_str)

    data = {
        **metrics_dict,
        "extracted answer": extracted_ans,
        "cot length": cot_lengths,
        "score reward": torch.sum(score_rew, dim=1).tolist(),
        "kl reward": torch.sum(kl_rew, dim=1).tolist(),
        # "penalty reward": torch.sum(cot_penalty_rew, dim=1).tolist(),
        # "max len reward": torch.sum(max_gen_length_penalty_rew, dim=1).tolist(),
        # "answer present reward": torch.sum(
        #     answer_trigger_present_rew, dim=1
        # ).tolist(),
        "total reward": reward_sums,
        "link": links,
    }
    # # sort by length of thinking, descending
    # data = sorted(data, key=lambda x: len(x[2]), reverse=True)
    table = wandb.Table(data=list(zip(*data.values())), columns=list(data.keys()))
    wandb.log({"thinking": table}, step=global_iter_num)


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
    evaluating_step_freq = args.get("evaluating_step_freq", None)
    logging_step_freq = args.get("logging_step_freq", None)
    saving_step_freq = args.get("saving_step_freq", None)
    model.train()
    epoch_result_dict = defaultdict(list)
    with tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        disable=not accelerator.is_main_process,
        desc="Train Loop",
    ) as t:
        for idx, batch in t:
            result_dict = defaultdict(list)
            # Do rollout first
            model.eval()
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
                _,
                adv,
                generated_texts,
                token_texts,
            ) = rollout(
                args,
                model,
                ref_model,
                tokenizer,
                batch,
                iter=global_iter_num,
            )

            if accelerator.is_main_process and args["wandb_log"]:
                metrics_dict = {
                    "dataset": batch["task"],
                    "input": batch["prefix_text"],
                    "correct answer": batch["targets"],
                    "generation": generated_texts,
                }
                log_table_metrics(
                    metrics_dict,
                    tokenizer,
                    val,
                    rew,
                    ret,
                    score_rew,
                    kl_rew,
                    global_iter_num,
                    token_texts,
                )

            model.train()
            # preprocess
            if args["adv_whitening"] == "global":
                adv = allgather_masked_whiten(adv, mask)  # (mini_bs, seqlen)
            elif args["adv_whitening"] == "local":
                adv = masked_whiten(adv, mask)

            batch_size_per_gpu = len(batch["input_ids"])
            mini_batch_size_per_gpu = args["mini_batch_size"]
            ppo_epochs = args["ppo_epochs"]
            train_stats = {}

            for ppo_epoch in range(ppo_epochs):
                perms = torch.randperm(batch_size_per_gpu)
                for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
                    b_inds = perms[mini_idx : mini_idx + mini_batch_size_per_gpu]
                    # Subset to batch
                    cur_val = val[b_inds].contiguous()  # mini_bs x seqlen
                    cur_old_logprob = old_logprob[
                        b_inds
                    ].contiguous()  # mini_bs x seqlen
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

                    # Forward current model
                    model.eval()
                    lm_logits, _, vpreds = model(
                        input_ids=cur_model_input_ids,
                        attention_mask=cur_model_attention_mask,
                    )
                    logprob = logprobs_from_logits(
                        lm_logits[:, :-1, :], cur_model_input_ids[:, 1:]
                    )  # (mini_bs, seqlen-1)

                    # Compute losses
                    loss = 0

                    # policy gradient loss
                    ratio = torch.exp(logprob - cur_old_logprob)
                    pg_losses = -cur_adv[:, :-1] * ratio
                    pg_losses2 = -cur_adv[:, :-1] * torch.clamp(
                        ratio, 1.0 - 0.2, 1.0 + 0.2
                    )
                    pg_loss = (
                        (torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum(
                            dim=-1
                        )
                        / resp_len_per_sample
                    ).mean()

                    # value loss
                    vpredclipped = torch.max(
                        torch.min(vpreds, cur_val + 0.2), cur_val - 0.2
                    )
                    vf_losses1 = (vpreds - cur_ret) ** 2
                    vf_losses2 = (vpredclipped - cur_ret) ** 2
                    vf_loss = (
                        0.5
                        * (
                            (torch.max(vf_losses1, vf_losses2) * cur_mask).sum(dim=-1)
                            / resp_len_per_sample
                        ).mean()
                    )
                    # vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum() / cur_mask.sum())

                    # total loss
                    loss += pg_loss + vf_coef * vf_loss

                    # token related metrics
                    mean_query_len = torch.mean(
                        allgather(torch.mean(query_len_per_sample))
                    )
                    std_query_len = torch.mean(
                        allgather(torch.std(query_len_per_sample))
                    )
                    mean_resp_len = torch.mean(
                        allgather(torch.mean(resp_len_per_sample))
                    )
                    std_resp_len = torch.mean(allgather(torch.std(resp_len_per_sample)))

                    # value related metrics
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
                    mean_ratio = masked_mean(ratio, cur_mask[:, :-1])
                    # mean_adv = masked_mean(cur_adv[:, :-1], cur_mask[:, :-1])
                    mean_logprob = masked_mean(logprob, cur_mask[:, :-1])
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
                            cur_param = deepspeed.utils.safe_get_full_fp32_param(
                                p
                            ).view(-1)
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
            if (
                evaluating_step_freq is not None
                and global_step % evaluating_step_freq == 0
            ):
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
                    save_path = os.path.join(
                        model_dir, f"global_step_{str(global_step)}"
                    )
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
    for idx, batch in tqdm(
        enumerate(dataloader),
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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_.sequences
        generated_ids = pad_across_processes(
            generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True
        )

        generated_ids = accelerator.gather(generated_ids)

        preds = [
            tokenizer.decode(
                g.cpu().numpy().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            for g in generated_ids
        ]
        predictions.extend(preds)
        target = batch["targets"]
        targets.extend(target)

    # predictions = predictions[: len(dataset)]
    # targets = targets[: len(dataset)]

    if accelerator.is_main_process and accelerator.is_local_main_process:
        pred_cots = extract_completion_batch(predictions)
        pred_answers = extract_answer_cot_batch(pred_cots)
        results = []
        corr_value = 0
        for pred_cot, target, pred_answer in zip(pred_cots, targets, pred_answers):
            cur_res = {
                "prediction_cot": pred_cot,
                "prediction_value": pred_answer,
                "target": target,
            }
            results.append(cur_res)
            corr_value += compare_and_calculate_reward(pred_cot, target)

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
        args["tokenizer_name_or_path"], use_fast=True
    )
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 2

    train_dataloader, test_all_dataloader = prepare_datasets_and_data_loaders(
        args, tokenizer
    )

    MODEL_CLASS = AutoModelForCausalLMWithValueHead
    model = MODEL_CLASS.from_pretrained(args["model_name_or_path"])
    # initialize ref model (if any)
    ref_model = None
    if args["ref_model_name_or_path"]:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args["ref_model_name_or_path"]
        )
        # from copy import deepcopy
        # ref_model = deepcopy(model)

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
    model, optimizer, train_dataloader, test_all_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_all_dataloader
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
                "test_dataloader": test_all_dataloader,
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
                        test_all_dataloader,
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
        engine: str = field(default="python")
        adv_whitening: str = field(default="global")
        ### new!
        max_per_task: int = field(default=1000)
        max_test_per_task: int = field(default=100)
        no_policy_loss_steps: int = field(default=0)
        reward_correct: float = field(default=1.0)
        reward_starts_correct: float = field(default=0.5)
        reward_contains_answer_trigger: float = field(default=0.01)

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
