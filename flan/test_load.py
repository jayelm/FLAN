import tqdm
from flan import mixtures as mixtures_v1
from flan.v2 import mixtures  # noqa
from flan.v2.constants_niv2 import NATINST_DEFAULT_TEST_TASKS
from transformers import AutoTokenizer

import seqio

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})

train_dataset = seqio.get_mixture_or_task(
    # "palmflan_flan_zs_noopt_10gist_randompos",
    "palmflan_niv2-eval_zs_noopt_5gist_fixedpos",
    # "wsc_template_0to10_no_opt_zero_shot_2gist_fixedpos"
).get_dataset(
    sequence_length={"inputs": 1024, "targets": 128},
    split="train",
    shuffle=True,
    num_epochs=1,
    use_cached=False,
    seed=42,
)


for _, ex in zip(range(5), train_dataset.as_numpy_iterator()):
    print(tokenizer.decode(ex['inputs']), tokenizer.decode(ex['targets']))
