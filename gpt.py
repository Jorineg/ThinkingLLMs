rollout(args, model, ref_model, tokenizer, batch, iter=global_iter_num)
print("did rollout")
torch.distributed.barrier()
print("after barrier")
# preprocess
if args["adv_whitening"] == "global":
    adv = allgather_masked_whiten(adv, mask)  # (mini_bs, seqlen)
elif args["adv_whitening"] == "local":
    adv = masked_whiten(adv, mask)

print("after gather")

if torch.cuda.is_available():
    torch.cuda.synchronize()

if accelerator.is_main_process and args["wandb_log"]:
    # unimportant
    accelerator.print("logged thinking table")
    # ...
    accelerator.print("created dataset cot lengths")
    # ...
    accelerator.print("calculated dataset metrics")
    wandb.log({"dataset_metrics": dataset_metrics}, step=global_iter_num)
    accelerator.print("logged dataset metrics")
    # ...
    accelerator.print("logged penalty and cot length")

for _ in range(ppo_epochs):
    print("doing ppo epoch")
    perms = torch.randperm(batch_size_per_gpu)
    for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
        b_inds = perms[mini_idx : mini_idx + mini_batch_size_per_gpu]
        # Subset to batch
        cur_val = val[b_inds].contiguous()  # mini_bs x seqlen
        cur_old_logprob = old_logprob[b_inds].contiguous()  # mini_bs x seqlen
        # ...

        print("did make contiguous")

        # ...
        model.eval()
        lm_logits, _, vpreds = model(
            input_ids=cur_model_input_ids,
            attention_mask=cur_model_attention_mask,
        )

        print("did second forward pass")

        # Compute losses
        loss = 0
        # policy gradient loss
        ratio = torch.exp(logprob - cur_old_logprob)
        pg_losses = -cur_adv[:, :-1] * ratio
        pg_losses2 = -cur_adv[:, :-1] * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
        pg_loss = (
            (torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum(dim=-1)
            / resp_len_per_sample
        ).mean()
        # pg_loss = (torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum() / cur_mask[:, :-1].sum()
        # pg_loss = (-logprob * cur_ret[:,:-1]).sum() / cur_mask[:, :-1].sum()

        # ...
        # total loss
        loss += pg_loss + vf_coef * vf_loss

        torch.distributed.barrier()

        print("computed losses")
        # token related metrics
        mean_query_len = torch.mean(allgather(torch.mean(query_len_per_sample)))
        std_query_len = torch.mean(allgather(torch.std(query_len_per_sample)))
        mean_resp_len = torch.mean(allgather(torch.mean(resp_len_per_sample)))
        std_resp_len = torch.mean(allgather(torch.std(resp_len_per_sample)))

        print("gathered metrics")