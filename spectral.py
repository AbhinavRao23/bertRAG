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
from utils import ravel_model

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
model = eqx.tree_deserialise_leaves('./model/sst2_bert.eqx', classifier)

flat, meta, static = ravel_model(model)
print('Total Number of parameters in BERT:', flat.shape[0])

print(meta.)