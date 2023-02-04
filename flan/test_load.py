import tqdm
from flan.v2 import mixtures  # noqa
from transformers import AutoTokenizer

import seqio

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})

train_dataset = seqio.get_mixture_or_task("niv2-train_zsopt").get_dataset(
    sequence_length={"inputs": 1024, "targets": 128},
    split="train",
    shuffle=True,
    num_epochs=1,
    shard_info=seqio.ShardInfo(index=0, num_shards=10),
    use_cached=False,
    seed=42,
)


eval_dataset = seqio.get_mixture_or_task("niv2-eval_zsnoopt").get_dataset(
    sequence_length={"inputs": 1024, "targets": 128},
    split="train",
    shuffle=False,
    num_epochs=1,
    shard_info=seqio.ShardInfo(index=0, num_shards=10),
    use_cached=False,
    seed=42,
)


for i, ex in enumerate(tqdm.tqdm(eval_dataset.as_numpy_iterator(), desc="Eval")):
    print(i, tokenizer.decode(ex["inputs"]))
