from typing import Dict, List, Mapping, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int 

SEQUENCE_LENGTH = " sequence_length"
NUMBER_HEADS = " num_heads"
HIDDEN_SIZE = " hidden_size"
NUMBER_CLASSES = " num_classes"

class EmbedderBlock(eqx.Module):
    token_embedder: eqx.nn.Embedding
    segment_embedder: eqx.nn.Embedding
    position_embedder: eqx.nn.Embedding
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 type_vocab_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey):

        token_key, segment_key, position_key = jax.random.split(key, 3)
        self.token_embedder = eqx.nn.Embedding(
            num_embeddings=vocab_size, embedding_size=embedding_size, 
            key=token_key)
        self.segment_embedder = eqx.nn.Embedding(
            num_embeddings=type_vocab_size, embedding_size=embedding_size,
            key=segment_key)
        self.position_embedder = eqx.nn.Embedding(
            num_embeddings=max_length, embedding_size=embedding_size, 
            key=position_key)
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self,
                 token_ids: Int[Array, SEQUENCE_LENGTH],
                 position_ids: Int[Array, SEQUENCE_LENGTH],
                 segment_ids: Int[Array, SEQUENCE_LENGTH],
                 enable_dropout: bool = False,
                 key: Optional[jax.random.PRNGKey] = None,
                 ) -> Float[Array, SEQUENCE_LENGTH + HIDDEN_SIZE]:

        tokens = jax.vmap(self.token_embedder)(token_ids)
        segments = jax.vmap(self.segment_embedder)(segment_ids)
        positions = jax.vmap(self.position_embedder)(position_ids)
        embedded_inputs = tokens + segments + positions
        embedded_inputs = jax.vmap(self.layernorm)(embedded_inputs)
        embedded_inputs = self.dropout(embedded_inputs, 
                                       inference=not enable_dropout, key=key)
        return embedded_inputs


class FeedForwardBlock(eqx.Module):
    mlp: eqx.nn.MLP
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 dropout_rate: float,
                 key: jax.random.PRNGKey):
        
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size, out_size=hidden_size, width_size=intermediate_size, 
            depth=1, activation=jax.nn.gelu, key = key)
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self,
                 inputs: Float[Array, HIDDEN_SIZE],
                 enable_dropout: bool = True,
                 key: Optional[jax.random.PRNGKey] = None
                 ) -> Float[Array, HIDDEN_SIZE]:
        
        output = self.mlp(inputs)
        output = self.dropout(output, inference=not enable_dropout, key=key)
        output += inputs
        output = self.layernorm(output)
        return output

class AttentionBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.Embedding
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 dropout_rate: float,
                 attention_dropout_rate: float,
                 key: jax.random.PRNGKey):
        
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key,)
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self,
                 inputs: Float[Array, SEQUENCE_LENGTH + HIDDEN_SIZE],
                 mask: Optional[Int[Array, SEQUENCE_LENGTH]],
                 enable_dropout: bool = False,
                 key: jax.random.PRNGKey = None,
                 ) -> Float[Array, SEQUENCE_LENGTH + HIDDEN_SIZE]:
        
        #todo JAXCOND (???)
        if mask is not None:
            mask = self.make_self_attention_mask(mask)
        
        #todo JAXCOND (???)
        # attention_key, dropout_key = (
        #     (None, None) if key is None else jax.random.split(key))
        attention_key, dropout_key = jax.lax.cond((key is None), 
                                                  lambda x: jnp.array((x,x)), 
                                                  lambda x: jax.random.split(x), 
                                                  key)

        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,)

        result = attention_output
        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)
        result = result + inputs
        result = jax.vmap(self.layernorm)(result)
        return result

    def make_self_attention_mask(
        self, mask: Int[Array, SEQUENCE_LENGTH]
    ) -> Float[Array, NUMBER_HEADS + SEQUENCE_LENGTH + SEQUENCE_LENGTH]:
        mask = jnp.multiply(
            jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2)
        )
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)


class TransformerLayer(eqx.Module):
    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_heads: int,
                 dropout_rate: float,
                 attention_dropout_rate: float,
                 key: jax.random.PRNGKey,):

        attention_key, ff_key = jax.random.split(key)
        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )

    def __call__(self,
                 inputs: Float[Array, SEQUENCE_LENGTH + HIDDEN_SIZE],
                 mask: Optional[Int[Array, SEQUENCE_LENGTH]] = None,
                 *,
                 enable_dropout: bool = False,
                 key: Optional[jax.random.PRNGKey] = None,
                 ) -> Float[Array, SEQUENCE_LENGTH + HIDDEN_SIZE]:
        
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(
            inputs, mask, enable_dropout=enable_dropout, key=attn_key)
        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys)
        
        return output


class Encoder(eqx.Module):

    embedder_block: EmbedderBlock
    layers: List[TransformerLayer]
    pooler: eqx.nn.Linear

    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 type_vocab_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 intermediate_size: int,
                 num_layers: int,
                 num_heads: int,
                 dropout_rate: float,
                 attention_dropout_rate: float,
                 key: jax.random.PRNGKey,):

        embedder_key, layer_key, pooler_key = jax.random.split(key, num=3)
        self.embedder_block = EmbedderBlock(vocab_size=vocab_size,
                                            max_length=max_length,
                                            type_vocab_size=type_vocab_size,
                                            embedding_size=embedding_size,
                                            hidden_size=hidden_size,
                                            dropout_rate=dropout_rate,
                                            key=embedder_key,)

        layer_keys = jax.random.split(layer_key, num=num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    key=layer_key,
                )
            )

        self.pooler = eqx.nn.Linear(
            in_features=hidden_size, out_features=hidden_size, key=pooler_key
        )

    def __call__(
            self,
            token_ids: Int[Array, SEQUENCE_LENGTH],
            position_ids: Int[Array, SEQUENCE_LENGTH],
            segment_ids: Int[Array, SEQUENCE_LENGTH],
            *,
            enable_dropout: bool = False,
            key: Optional[jax.random.PRNGKey] = None) -> Dict[str, Array]:
        
        emb_key, l_key = (None, None) if key is None else jax.random.split(key)
        embeddings = self.embedder_block(token_ids=token_ids,
                                         position_ids=position_ids,
                                         segment_ids=segment_ids,
                                         enable_dropout=enable_dropout,
                                         key=emb_key,)
        mask = jnp.asarray(token_ids != 0, dtype=jnp.int32)
        x = embeddings
        layer_outputs = []
        ##todo JAX SCAN (???) 
        for layer in self.layers:
            cl_key, l_key = (None, None) if l_key is None else jax.random.split(l_key)
            x = layer(x, mask, enable_dropout=enable_dropout, key=cl_key)
            layer_outputs.append(x)

        first_token_last_layer = x[..., 0, :]
        pooled = self.pooler(first_token_last_layer)
        pooled = jnp.tanh(pooled)

        return {"embeddings": embeddings, "layers": layer_outputs, "pooled": pooled}


class BertClassifier(eqx.Module):
    """BERT classifier."""

    encoder: Encoder
    classifier_head: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, 
                 config: Mapping, 
                 num_classes: int, 
                 key: jax.random.PRNGKey):
        
        encoder_key, head_key = jax.random.split(key)
        self.encoder = Encoder(
            vocab_size=config["vocab_size"],
            max_length=config["max_position_embeddings"],
            type_vocab_size=config["type_vocab_size"],
            embedding_size=config["hidden_size"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            num_layers=config["num_hidden_layers"],
            num_heads=config["num_attention_heads"],
            dropout_rate=config["hidden_dropout_prob"],
            attention_dropout_rate=config["attention_probs_dropout_prob"],
            key=encoder_key)
        
        self.classifier_head = eqx.nn.Linear(
            in_features=config["hidden_size"], out_features=num_classes, 
            key=head_key)
        self.dropout = eqx.nn.Dropout(config["hidden_dropout_prob"])

    def __call__(self,
                 inputs: Dict[str, Int[Array, SEQUENCE_LENGTH]],
                 enable_dropout: bool = True,
                 key: jax.random.PRNGKey = None,
                 ) -> Float[Array, NUMBER_CLASSES]:
        
        seq_len = inputs["token_ids"].shape[-1]
        position_ids = jnp.arange(seq_len)

        #JAXCOND
        # e_key, d_key = (None, None) if key is None else jax.random.split(key)
        e_key, d_key = jax.lax.cond((key is None), 
                                    lambda x: jnp.array((x,x)), 
                                    lambda x: jax.random.split(x), key)

        pooled_output = self.encoder(token_ids=inputs["token_ids"],
                                     segment_ids=inputs["segment_ids"],
                                     position_ids=position_ids,
                                     enable_dropout=enable_dropout,
                                     key=e_key,)["pooled"]
        pooled_output = self.dropout(pooled_output, 
                                     inference=not enable_dropout, key=d_key)
        return self.classifier_head(pooled_output)