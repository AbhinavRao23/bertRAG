from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from datasets import load_dataset 
from transformers import AutoTokenizer
import tqdm
import einops

from bert import BertClassifier

key = jax.random.PRNGKey(0)
bert_config = {"vocab_size": 30522,
               "hidden_size": 128,
               "num_hidden_layers": 2,
               "num_attention_heads": 2,
               "hidden_act": "gelu",
               "intermediate_size": 512,
               "hidden_dropout_prob": 0.1,
               "attention_probs_dropout_prob": 0.1,
               "max_position_embeddings": 512,
               "type_vocab_size": 2,
               "initializer_range": 0.02,}

tokenizer = AutoTokenizer.from_pretrained(
    "google/bert_uncased_L-2_H-128_A-2", model_max_length=128)
def tokenize(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True)
ds = load_dataset("sst2")
ds = ds.map(tokenize, batched=True)
ds.set_format(type="jax", columns=["input_ids", "token_type_ids", "label"])

def make_eval_step(model, inputs):
    return jax.vmap(partial(model, enable_dropout=False))(inputs)

key, model_key, train_key = jax.random.split(key, 3)
classifier = BertClassifier(config=bert_config, num_classes=2, key=model_key)
model = eqx.tree_deserialise_leaves('./model/sst2_bert.eqx', classifier)
batch_size = 32

outputs = []
for batch in tqdm.tqdm(
    ds["validation"].iter(batch_size=batch_size),
    unit="steps",
    total=np.ceil(ds["validation"].num_rows / batch_size),
    desc="Validation"):
    token_ids, token_type_ids = batch["input_ids"], batch["token_type_ids"]
    label = batch["label"]
    inputs = {"token_ids": token_ids, "segment_ids": token_type_ids}

    # Compare predicted class with label.
    # output = p_make_eval_step(model, inputs)
    output = make_eval_step(model, inputs)
    output = map(float, np.argmax(output.reshape(-1, 2), axis=-1) == label)
    outputs.extend(output)

print('*'*100)
print(f"Accuracy: {100 * np.sum(outputs) / len(outputs):.2f}%")
print('*'*100)