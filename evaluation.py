print("importing libraries...")
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from datasets import load_dataset
import torch
import accelerate
import tqdm

# params
@dataclass
class Arguments:
    model_name: str
    dataset_name: str
    split_name: str
    question_column_name: str
    answer_column_name: str
    max_gen_length: int
    batch_size: int
    pad_to_multiple_of: int
    cot_trigger: str
    answer_trigger: str
    instruction: str

parser = HfArgumentParser(Arguments)
args = asdict(parser.parse_args_into_dataclasses()[0])

# format input
format_input = lambda x: f"{args["instruction"]}{x[args["question_column_name"]]}\n{args["cot_trigger"]}"


# tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
tokenizer.padding_side = "left"
# explicitly set pad/eos token????????
######################################

print("load dataset...")
df = load_dataset(args["dataset_name"], split=args["split_name"]).to_pandas()
df["formatted_input"] = df.apply(format_input, axis=1)
df["group"] = df.index // args["batch_size"]


def tokenize_input(batch):
    res = tokenizer(
        batch["formatted_input"].tolist(),
        return_tensors="pt",
        pad_to_multiple_of=args["pad_to_multiple_of"],
        padding=True,
        truncation=False,
    )
    res["targets"] = batch[args["answer_column_name"]].tolist()
    return res


print("tokenize dataset...")
batches = df.groupby("group").apply(tokenize_input)


# model
print("load model...")
model = AutoModelForCausalLM.from_pretrained(
    args["model_name"], torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
)
model.eval()

print("move model to multi GPUs...")
distributed_state = accelerate.PartialState()
model.to(distributed_state.device)


# inference
print("evaluating...")
results = []
with distributed_state.split_between_processes(batches.tolist()) as batches:
    for batch in tqdm(batches, disable=not accelerate.is_main_process()):
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=args["max_gen_length"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=0.0,
                num_return_sequences=1,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(list(zip(decoded, batch["targets"])))


# functions
# takes a batch strings that contain input and completion
# returns a list of completion strings
def extract_completion_batch(input_and_completion_batch):
    cot_trigger_count_in_instructions = args["instruction"].count(args["cot_trigger"])
    splitted = [res.split(args["cot_trigger"]) for res in input_and_completion_batch]
    return [
        args["cot_trigger"].join(split[cot_trigger_count_in_instructions + 1 :])
        for split in splitted
    ]


# takes a batch strings that contain only the completion
# returns a list of answer strings (part after the first found answer trigger)
# returns empty string if no answer trigger is found
def extract_answer_cot_batch(answer_cot_batch):
    splitted_batch = [res.split(args["answer_trigger"]) for res in answer_cot_batch]
    return [
        args["answer_trigger"].join(splitted[1:]) if len(splitted) >= 2 else ""
        for splitted in splitted_batch
    ]

def check_equal(a, b):
    return a.lower().strip() == b.lower().strip()

# evaluation
if distributed_state.is_main_process:
    gathered_results = accelerate.utils.gather_object(results)
    gathered_results = list(zip(*gathered_results))

    # cut if apply_padding = True
    # gathered_results = gathered_results[:len(df)]

    completions = extract_completion_batch(gathered_results[0])
    answers = extract_answer_cot_batch(completions)
    targets = gathered_results[1]
    correct = [check_equal(a, b) for a, b in zip(answers, targets)]
    accuracy = sum(correct) / len(correct)
    print(f"Accuracy: {accuracy:.2f}")

