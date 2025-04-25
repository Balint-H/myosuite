from typing import Any, Tuple

import jax
from jax import numpy as jnp
from brax.training.types import Params, Metrics
from train_hierarchical import LLSupervisedData
from brax.training.networks import FeedForwardNetwork
from mujoco import mjx

def hierarchical_ll_loss(
    params: Params,
    normalizer_params: Any,
    data: LLSupervisedData,
    rng: jnp.ndarray,
    ll_network: FeedForwardNetwork,
) -> Tuple[jnp.ndarray, Metrics]:
  """Computes PPO loss.

  Args:
    params: Network parameters,
    normalizer_params: Parameters of the normalizer.
    data: Transition that with leading dimension [B, T]. extra fields required
      are ['state_extras']['truncation'] ['policy_extras']['raw_action']
      ['policy_extras']['log_prob']
    rng: Random key
    ppo_network: PPO networks.

  Returns:
    A tuple (loss, metrics)
  """

  # Put the time dimension first.
  data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
  policy_logits = policy_apply(
      normalizer_params, params.policy, data.observation
  )

  baseline = value_apply(normalizer_params, params.value, data.observation)
  terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
  bootstrap_value = value_apply(normalizer_params, params.value, terminal_obs)

  rewards = data.reward * reward_scaling
  truncation = data.extras['state_extras']['truncation']
  termination = (1 - data.discount) * (1 - truncation)

  target_action_log_probs = parametric_action_distribution.log_prob(
      policy_logits, data.extras['policy_extras']['raw_action']
  )
  behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

  vs, advantages = compute_gae(
      truncation=truncation,
      termination=termination,
      rewards=rewards,
      values=baseline,
      bootstrap_value=bootstrap_value,
      lambda_=gae_lambda,
      discount=discounting,
  )
  if normalize_advantage:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

  surrogate_loss1 = rho_s * advantages
  surrogate_loss2 = (
      jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
  )

  policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

  # Value function loss
  v_error = vs - baseline
  v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5


  total_loss = policy_loss + v_loss + entropy_loss
  return total_loss, {
      'total_loss': total_loss,
      'policy_loss': policy_loss,
      'v_loss': v_loss,
      'entropy_loss': entropy_loss,
  }