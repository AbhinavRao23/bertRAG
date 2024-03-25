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

key, model_key, train_key = jax.random.split(key, 3)
classifier = BertClassifier(config=bert_config, num_classes=2, key=model_key)

tokenizer = AutoTokenizer.from_pretrained(
    "google/bert_uncased_L-2_H-128_A-2", model_max_length=128)
def tokenize(example):
    return tokenizer(example["sentence"], padding="max_length", truncation=True)
ds = load_dataset("sst2")
ds = ds.map(tokenize, batched=True)
ds.set_format(type="jax", columns=["input_ids", "token_type_ids", "label"])


@eqx.filter_value_and_grad
def compute_loss(classifier, inputs, key):
    batch_size = inputs["token_ids"].shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    logits = jax.vmap(classifier, in_axes=(0, None, 0))(inputs, True, batched_keys)
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=inputs["label"]
        )
    )

def make_step(model, inputs, opt_state, key, tx):
    key, new_key = jax.random.split(key)
    loss, grads = compute_loss(model, inputs, key)
    updates, opt_state = tx.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state, new_key

def make_eval_step(model, inputs):
    return jax.vmap(partial(model, enable_dropout=False))(inputs)

epochs = 1
batch_size = 64
learning_rate = 1e-5
tx = optax.adam(learning_rate=learning_rate)
tx = optax.chain(optax.clip_by_global_norm(1.0), tx)
opt_state = tx.init(eqx.filter(classifier, eqx.is_inexact_array))
model = classifier

for epoch in range(epochs):
    with tqdm.tqdm(
        ds["train"].iter(batch_size=batch_size, drop_last_batch=True),
        total=ds["train"].num_rows // batch_size,
        unit="steps",
        desc=f"Epoch {epoch+1}/{epochs}",) as tqdm_epoch:
        
        for batch in tqdm_epoch:
            token_ids, token_type_ids = batch["input_ids"], batch["token_type_ids"]
            label = batch["label"]
            inputs = {
                "token_ids": token_ids,
                "segment_ids": token_type_ids,
                "label": label,
            }
            loss, model, opt_state, train_key = make_step(
                model, inputs, opt_state, train_key, tx)
            tqdm_epoch.set_postfix(loss=np.sum(loss).item())

eqx.tree_serialise_leaves('./model/sst2_bert.eqx', model)