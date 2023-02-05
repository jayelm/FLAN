import tqdm
from flan.v2 import mixtures  # noqa
from flan.v2.constants_niv2 import NATINST_DEFAULT_TEST_TASKS
from transformers import AutoTokenizer

import seqio

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})

# train_dataset = seqio.get_mixture_or_task("niv2-train_zsopt").get_dataset(
train_dataset = seqio.get_mixture_or_task(
    "train_tfds_natural_instructions_template_0_zero_shot"
).get_dataset(
    sequence_length={"inputs": 1024, "targets": 128},
    split="train",
    shuffle=True,
    num_epochs=1,
    use_cached=False,
    seed=42,
)


# eval_dataset = seqio.get_mixture_or_task("niv2-eval_zsopt").get_dataset(
#     sequence_length={"inputs": 1024, "targets": 128},
#     split="train",
#     shuffle=False,
#     num_epochs=1,
#     # shard_info=seqio.ShardInfo(index=0, num_shards=10),
#     use_cached=False,
#     seed=42,
# )


# Verify that train dataset includes only train tasks.
seen_task_names = set()
for i, ex in enumerate(tqdm.tqdm(train_dataset.as_numpy_iterator(), desc="Eval")):
    if ex["task_name"] not in seen_task_names:
        print(ex["task_name"])
        seen_task_names.add(ex["task_name"])
seen_task_names = {t.decode("utf-8") for t in seen_task_names}

NATINST_DEFAULT_TEST_TASKS = set(NATINST_DEFAULT_TEST_TASKS)
print("Seen tasks not in test tasks:", seen_task_names - NATINST_DEFAULT_TEST_TASKS)
print("Test tasks not in seen tasks:", NATINST_DEFAULT_TEST_TASKS - seen_task_names)
# Assert intersection of seen and test tasks is empty.
assert not (seen_task_names & NATINST_DEFAULT_TEST_TASKS)
