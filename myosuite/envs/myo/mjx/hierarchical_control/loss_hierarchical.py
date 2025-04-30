from typing import Any, Tuple

import jax
from jax import numpy as jnp
from brax.training.types import Params, Metrics
from train_hierarchical import LLSupervisedData
from brax.training.networks import FeedForwardNetwork
from mujoco import mjx


@jax.custom_jvp
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
  data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, x.shape[-1])), data)
  hl_torque_error = data.desired_torque-data.torque_designated
  ll_error, _, _, _ = jax.vmap(jnp.linalg.lstsq)(data.jacobian, hl_torque_error, r_cond=None)
  ll_loss = jnp.mean(ll_error * ll_error) * 0.5 * 0.5

  return ll_loss, {
      'll_loss': ll_loss,
  }


@hierarchical_ll_loss.defjvp
def hierarchical_ll_loss_jvp(primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  ans = f(x, y)
  ans_dot = -1. * x_dot
  return ans, ans_dot