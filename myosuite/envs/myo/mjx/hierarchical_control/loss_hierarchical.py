from typing import Any, Tuple

import jax
import optax
from jax import numpy as jnp
from brax.training.types import Params, Metrics
from train_hierarchical import LLSupervisedData
from brax.training.networks import FeedForwardNetwork
from mujoco import mjx


def hierarchical_ll_loss(
    params: Params,
    normalizer_params: Any,
    data: LLSupervisedData,
    network: FeedForwardNetwork
):
  logits = network.apply(normalizer_params, params, data.ll_obs)
  return hierarchical_ll_loss_head(logits, data)


@jax.custom_vjp
def hierarchical_ll_loss_head(
    logits,
    data: LLSupervisedData,
) -> Tuple[jnp.ndarray, Metrics]:
  """Computes PPO loss.

  Args:
    data: LLSupervisedData that with leading dimension [Batch, Time].
  Returns:
    A tuple (loss, metrics)
  """

  # flatten batch and time dimensions
  hl_torque_error = data.torque_designated - data.hl_desired_torque
  hl_torque_error_flat = jax.tree_util.tree_map(
    lambda x: jnp.reshape(x, (-1, x.shape[-1])),
    hl_torque_error
  )

  hl_torque_loss = 0.5*(hl_torque_error_flat*hl_torque_error_flat).sum(axis=1).mean()
  return hl_torque_loss, {
      'torque_loss': hl_torque_loss,
      'torque_error': hl_torque_error
  }


def hierarchical_ll_loss_fwd(logits, data: LLSupervisedData):
  loss, aux = hierarchical_ll_loss_head(logits, data)

  return (loss, aux), (data.jacobian, aux['torque_error'])


def hierarchical_ll_loss_bwd(res, g):
  # g is df/dL where f is the function being diffed. If L is diffed, then f=L and g should be [1].
  running_grads = jax.vmap(jax.vmap((lambda j, e: e@j), in_axes=0, out_axes=0), in_axes=1, out_axes=1)(res[0], res[1])
  return running_grads*g[0], None


hierarchical_ll_loss_head.defvjp(hierarchical_ll_loss_fwd, hierarchical_ll_loss_bwd)