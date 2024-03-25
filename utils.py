import jax
import jax.numpy as jnp
from jax import flatten_util
import equinox as eqx


def ravel_pytree(pytree):
    ''' Ravels pytree w/ array leaves into a flat array.
    Args:
        pytree: The pytree to be unraveled (with well-behaving leaves).
    Returns:
        flat: A 1D jax array with all the leaves of the pytree raveled.
        meta: Tuple with meta-data to rebuild pytree from flat.
    '''
    flat, meta = flatten_util.ravel_pytree(pytree)
    return flat, meta


def unravel_pytree(flat, meta):
    ''' Builds pytree w/ array leaves from a flat array.
    Args:
        flat: A 1D jax array.
        meta: Tuple with meta-data to rebuild corresponding pytree 
        from flat.
    Returns:
        pytree: The pytree built by unraveling 'flat' using the 
        tree structure from 'meta'.
    '''
    tree_def = meta.args[0]
    splits, shapes = meta.args[1].args
    leaves = [param.reshape(shape) for shape,param in \
              zip(shapes, jnp.split(flat, splits))]
    return jax.tree_util.tree_unflatten(tree_def, leaves)


def ravel_model(model):
    ''' Convenience function that ravels eqx model pytree into a
      flat array.
    Args:
        model: An equinox model (also a pytree).
    Returns:
        flat: A 1D jax array with all parameters of model raveled open.
        meta: Tuple with meta data to rebuild pytree from flat.
        static: Static information of all non-parametric leaf to 
        rebuild model from parametric pytree.
    '''
    params, static = eqx.partition(model, eqx.is_array)
    flat, meta = ravel_pytree(params)
    return flat, meta, static


def unravel_model(flat, meta, static):
    ''' Convenience function that unravels eqx model pytree from an array.
    Args:
        flat: A 1D jax array with all parameters of an equinox model 
        raveled open.
        meta: Tuple with meta data to rebuild parametric pytree from flat.
        static: Static information of all non-parametric leaf to rebuild 
        model from parametric pytree.
    Returns:
        model: A model built using flat as parameters.
    '''
    params = unravel_pytree(flat, meta)
    return eqx.combine(params, static)