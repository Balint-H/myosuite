from typing import Any, Tuple

import jax
import optax
from jax import numpy as jnp
from brax.training.types import Params, Metrics
from train_hierarchical import LLSupervisedData
from brax.training.networks import FeedForwardNetwork
from mujoco import mjx


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

  hl_torque_loss = 0.5*(hl_torque_error_flat*hl_torque_error_flat).sum(axis=0).mean()
  return hl_torque_loss, {
      'torque_loss': hl_torque_loss,
      'torque_error': hl_torque_error
  }

def hierarchical_ll_loss_fwd(logits, data: LLSupervisedData):
  # Returns primal output and residuals to be used in backward pass by f_bwd.
  loss, aux = hierarchical_ll_loss_head(logits, data)

  return (loss, aux), (data.jacobian, aux['torque_error'])

def hierarchical_ll_loss_bwd(res, g):
  running_grads = jax.vmap((lambda j, e: e@j.T), in_axes=[0, 1], out_axes=[0, 1])(res[0], res[1])
  return (running_grads*g[0], None)


hierarchical_ll_loss_head.defvjp(hierarchical_ll_loss_fwd, hierarchical_ll_loss_bwd)