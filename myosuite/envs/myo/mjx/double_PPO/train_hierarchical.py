# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
import acting_hierarchical
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


# region Helper methods
def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return leaf.astype(leaf.dtype)

  return jax.tree_util.tree_map(f, tree)


def _validate_madrona_args(
    madrona_backend: bool,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    eval_env: Optional[envs.Env] = None,
):
  """Validates arguments for Madrona-MJX."""
  if madrona_backend:
    if eval_env:
      raise ValueError("Madrona-MJX doesn't support multiple env instances")
    if num_eval_envs != num_envs:
      raise ValueError('Madrona-MJX requires a fixed batch size')
    if action_repeat != 1:
      raise ValueError(
          "Implement action_repeat using PipelineEnv's _n_frames to avoid"
          ' unnecessary rendering!'
      )


def _maybe_wrap_env(
    env: Union[envs_v1.Env, envs.Env],
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    local_device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
  """Wraps the environment for training/eval if wrap_env is True."""
  if not wrap_env:
    return env
  if episode_length is None:
    raise ValueError('episode_length must be specified in ppo.train')
  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // local_device_count
    # all devices gets the same randomization rng
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng
    )
  if wrap_env_fn is not None:
    wrap_for_training = wrap_env_fn
  elif isinstance(env, envs.Env):
    wrap_for_training = envs.training.wrap
  else:
    wrap_for_training = envs_v1.wrappers.wrap_for_training
  env = wrap_for_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )  # pytype: disable=wrong-keyword-args
  return env


def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
  """Apply random translations to B x T x ... pixel observations.

  The same shift is applied across the unroll_length (T) dimension.

  Args:
    obs: a dictionary of observations
    key: a PRNGKey

  Returns:
    A dictionary of observations with translated pixels
  """

  @jax.vmap
  def rt_all_views(
      ub_obs: Mapping[str, jax.Array], key: PRNGKey
  ) -> Mapping[str, jax.Array]:
    # Expects dictionary of unbatched observations.
    def rt_view(
        img: jax.Array, padding: int, key: PRNGKey
    ) -> jax.Array:  # TxHxWxC
      # Randomly translates a set of pixel inputs.
      # Adapted from
      # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
      crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
      zero = jnp.zeros((1,), dtype=jnp.int32)
      crop_from = jnp.concatenate([zero, crop_from, zero])
      padded_img = jnp.pad(
          img,
          ((0, 0), (padding, padding), (padding, padding), (0, 0)),
          mode='edge',
      )
      return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

    out = {}
    for k_view, v_view in ub_obs.items():
      if k_view.startswith('pixels/'):
        key, key_shift = jax.random.split(key)
        out[k_view] = rt_view(v_view, 4, key_shift)
    return {**ub_obs, **out}

  bdim = next(iter(obs.items()), None)[1].shape[0]
  keys = jax.random.split(key, bdim)
  obs = rt_all_views(obs, keys)
  return obs


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
  """Removes pixel observations from the observation dict."""
  if not isinstance(obs, Mapping):
    return obs
  return {k: v for k, v in obs.items() if not k.startswith('pixels/')}
# endregion


# region Dataclasses
@flax.struct.dataclass
class PPOLearningParams:
  """Parameters related to the PPO learning algorithm configuration."""
  learning_rate: float = 1e-4
  entropy_cost: float = 1e-4
  discounting: float = 0.9
  reward_scaling: float = 1.0
  gae_lambda: float = 0.95
  clipping_epsilon: float = 0.3
  normalize_advantage: bool = True
  max_grad_norm: float | None = None
  num_updates_per_batch: int = 2

@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: jnp.ndarray
# endregion


def train(
    # region Input args
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,

    hl_network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,

    ll_network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,

    hl_ppo_learning_config: PPOLearningParams = PPOLearningParams(),
    ll_ppo_learning_config: PPOLearningParams = PPOLearningParams(),

    seed: int = 0,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    hl_policy_params_fn: Callable[..., None] = lambda *args: None,
    ll_policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    hl_save_checkpoint_path: Optional[str] = None,
    ll_save_checkpoint_path: Optional[str] = None,
    hl_restore_checkpoint_path: Optional[str] = None,
    ll_restore_checkpoint_path: Optional[str] = None,
    hl_restore_params: Optional[Any] = None,
    ll_restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    # endregion
):
  """PPO training.

  Args:
    environment: the environment to train
    num_timesteps: the total number of environment steps to use during training
    max_devices_per_host: maximum number of chips to use per host process
    wrap_env: If True, wrap the environment for training. Otherwise use the
      environment as is.
    madrona_backend: whether to use Madrona backend for training
    augment_pixels: whether to add image augmentation to pixel inputs
    num_envs: the number of parallel environments to use for rollouts
      NOTE: `num_envs` must be divisible by the total number of chips since each
        chip gets `num_envs // total_number_of_chips` environments to roll out
      NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
        data generated by `num_envs` parallel envs gets used for gradient
        updates over `num_minibatches` of data, where each minibatch has a
        leading dimension of `batch_size`
    episode_length: the length of an environment episode
    action_repeat: the number of timesteps to repeat an action
    wrap_env_fn: a custom function that wraps the environment for training. If
      not specified, the environment is wrapped with the default training
      wrapper.
    randomization_fn: a user-defined callback function that generates randomized
      environments
    learning_rate: learning rate for ppo loss
    entropy_cost: entropy reward for ppo loss, higher values increase entropy of
      the policy
    discounting: discounting rate
    unroll_length: the number of timesteps to unroll in each environment. The
      PPO loss is computed over `unroll_length` timesteps
    batch_size: the batch size for each minibatch SGD step
    num_minibatches: the number of times to run the SGD step, each with a
      different minibatch with leading dimension of `batch_size`
    num_updates_per_batch: the number of times to run the gradient update over
      all minibatches before doing a new environment rollout
    num_resets_per_eval: the number of environment resets to run between each
      eval. The environment resets occur on the host
    normalize_observations: whether to normalize observations
    reward_scaling: float scaling for reward
    clipping_epsilon: clipping epsilon for PPO loss
    gae_lambda: General advantage estimation lambda
    max_grad_norm: gradient clipping norm value. If None, no clipping is done
    normalize_advantage: whether to normalize advantage estimate
    network_factory: function that generates networks for policy and value
      functions
    seed: random seed
    num_evals: the number of evals to run during the entire training run.
      Increasing the number of evals increases total training time
    eval_env: an optional environment for eval only, defaults to `environment`
    num_eval_envs: the number of envs to use for evluation. Each env will run 1
      episode, and all envs run in parallel during eval.
    deterministic_eval: whether to run the eval with a deterministic policy
    log_training_metrics: whether to log training metrics and callback to
      progress_fn
    training_metrics_steps: the number of environment steps between logging
      training metrics
    progress_fn: a user-defined callback function for reporting/plotting metrics
    policy_params_fn: a user-defined callback function that can be used for
      saving custom policy checkpoints or creating policy rollouts and videos
    save_checkpoint_path: the path used to save checkpoints. If None, no
      checkpoints are saved.
    restore_checkpoint_path: the path used to restore previous model params
    restore_params: raw network parameters to restore the TrainingState from.
      These override `restore_checkpoint_path`. These paramaters can be obtained
      from the return values of ppo.train().
    restore_value_fn: whether to restore the value function from the checkpoint
      or use a random initialization

  Returns:
    Tuple of (make_policy function, network params, metrics)
  """
  # region Validate args
  assert batch_size * num_minibatches % num_envs == 0
  _validate_madrona_args(
      madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
  )
  # endregion

  # region Fetching and logging device info
  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(),
      process_count,
      process_id,
      local_device_count,
      local_devices_to_use,
  )
  device_count = local_devices_to_use * process_count
  assert num_envs % device_count == 0
  # endregion

  # region Calculate learning step sizes
  # The number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat
  )
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)
  # endregion

  # region PRNG setup
  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value = jax.random.split(global_key)
  del global_key
  # endregion

  # region Create wrapped, randomized env, get obs shape
  env = _maybe_wrap_env(
      environment,
      wrap_env,
      num_envs,
      episode_length,
      action_repeat,
      local_device_count,
      key_env,
      wrap_env_fn,
      randomization_fn,
  )
  reset_fn = jax.jit(jax.vmap(env.reset))
  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(
      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
  )
  env_state = reset_fn(key_envs)
  # Discard the batch axes over devices and envs.
  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
  # endregion

  # region HL and LL policy makers
  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize

  # region Set up HL policy
  hl_ppo_network = hl_network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  make_hl_policy = ppo_networks.make_inference_fn(hl_ppo_network)
  hl_optimizer = optax.adam(learning_rate=hl_ppo_learning_config.learning_rate)
  if hl_ppo_learning_config.max_grad_norm is not None:
    # TODO: Move gradient clipping to `training/gradients.py`.
    hl_optimizer = optax.chain(
        optax.clip_by_global_norm(hl_ppo_learning_config.max_grad_norm),
        optax.adam(learning_rate=hl_ppo_learning_config.learning_rate),
    )
  # endregion

  # region Set up LL policy
  ll_ppo_network = ll_network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  make_ll_policy = ppo_networks.make_inference_fn(ll_ppo_network)
  ll_optimizer = optax.adam(learning_rate=ll_ppo_learning_config.learning_rate)
  if ll_ppo_learning_config.max_grad_norm is not None:
    # TODO: Move gradient clipping to `training/gradients.py`.
    ll_optimizer = optax.chain(
        optax.clip_by_global_norm(ll_ppo_learning_config.max_grad_norm),
        optax.adam(learning_rate=ll_ppo_learning_config.learning_rate),
    )
  # endregion
  # endregion

  # region HL and LL policy gradient update functions
  hl_loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=hl_ppo_network,
      entropy_cost=hl_ppo_learning_config.entropy_cost,
      discounting=hl_ppo_learning_config.discounting,
      reward_scaling=hl_ppo_learning_config.reward_scaling,
      gae_lambda=hl_ppo_learning_config.gae_lambda,
      clipping_epsilon=hl_ppo_learning_config.clipping_epsilon,
      normalize_advantage=hl_ppo_learning_config.normalize_advantage,
  )

  ll_loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      ppo_network=ll_ppo_network,
      entropy_cost=ll_ppo_learning_config.entropy_cost,
      discounting=ll_ppo_learning_config.discounting,
      reward_scaling=ll_ppo_learning_config.reward_scaling,
      gae_lambda=ll_ppo_learning_config.gae_lambda,
      clipping_epsilon=ll_ppo_learning_config.clipping_epsilon,
      normalize_advantage=ll_ppo_learning_config.normalize_advantage,
  )

  hl_gradient_update_fn = gradients.gradient_update_fn(
      hl_loss_fn, hl_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )

  ll_gradient_update_fn = gradients.gradient_update_fn(
      ll_loss_fn, ll_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )
  # endregion

  # region Metrics and checkpoints
  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=training_metrics_steps
      or env_step_per_training_step,
      progress_fn=progress_fn,
  )

  hl_ckpt_config = checkpoint.network_config(
      observation_size=obs_shape,
      action_size=env.action_size,
      normalize_observations=normalize_observations,
      network_factory=hl_network_factory,
  )

  ll_ckpt_config = checkpoint.network_config(
      observation_size=obs_shape,
      action_size=env.action_size,
      normalize_observations=normalize_observations,
      network_factory=ll_network_factory,
  )
  # endregion

  def hierarchy_minibatch_step(
      carry,
      hl_data: types.Transition,
      ll_data: types.Transition,
      hl_normalizer_params: running_statistics.RunningStatisticsState,
      ll_normalizer_params: running_statistics.RunningStatisticsState,
  ):
    (hl_optimizer_state, ll_optimizer_state), (hl_params, ll_params), key = carry
    key, key_hl_loss, key_ll_loss = jax.random.split(key, 3)
    (_, metrics), hl_params, hl_optimizer_state = hl_gradient_update_fn(
        hl_params,
        hl_normalizer_params,
        hl_data,
        key_hl_loss,
        optimizer_state=hl_optimizer_state,
    )

    (_, _), ll_params, ll_optimizer_state = ll_gradient_update_fn(
        ll_params,
        ll_normalizer_params,
        ll_data,
        key_ll_loss,
        optimizer_state=ll_optimizer_state,
    )

    return ((hl_optimizer_state, ll_optimizer_state), (hl_params, ll_params), key), metrics

  def hierarchy_sgd_step(
      carry,
      unused_t,
      hl_data: types.Transition,
      ll_data: types.Transition,
      hl_normalizer_params: running_statistics.RunningStatisticsState,
      ll_normalizer_params: running_statistics.RunningStatisticsState,
  ):
    (hl_optimizer_state, ll_optimizer_state), (hl_params, ll_params), key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    if augment_pixels:
      key, key_hl_rt, key_ll_rt = jax.random.split(key,3)
      r_translate = functools.partial(_random_translate_pixels, key=key_hl_rt)
      hl_data = types.Transition(
          observation=r_translate(hl_data.observation),
          action=hl_data.action,
          reward=hl_data.reward,
          discount=hl_data.discount,
          next_observation=r_translate(hl_data.next_observation),
          extras=hl_data.extras,
      )
      r_translate = functools.partial(_random_translate_pixels, key=key_ll_rt)
      ll_data = types.Transition(
          observation=r_translate(ll_data.observation),
          action=ll_data.action,
          reward=ll_data.reward,
          discount=ll_data.discount,
          next_observation=r_translate(ll_data.next_observation),
          extras=ll_data.extras,
      )

    def convert_data(x: jnp.ndarray):  # TODO: Consider if we want the same shuffle for the two datas like this
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_hl_data = jax.tree_util.tree_map(convert_data, hl_data)
    shuffled_ll_data = jax.tree_util.tree_map(convert_data, ll_data)
    ((hl_optimizer_state, ll_optimizer_state), (hl_params, ll_params), _), metrics = jax.lax.scan(
        functools.partial(hierarchy_minibatch_step,
                          hl_normalizer_params=hl_normalizer_params,
                          ll_normalizer_params=ll_normalizer_params),
        ((hl_optimizer_state, ll_optimizer_state), (hl_params, ll_params), key_grad),
        (shuffled_hl_data, shuffled_ll_data),
        length=num_minibatches,
    )
    return ((hl_optimizer_state, ll_optimizer_state), (hl_params, ll_params), key), metrics

  def hierarchy_training_step(
      carry: Tuple[Tuple[TrainingState, TrainingState], envs.State, PRNGKey], unused_t
  ) -> Tuple[Tuple[Tuple[TrainingState, TrainingState], envs.State, PRNGKey], Metrics]:
    (hl_training_state, ll_training_state), state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    hl_policy = make_hl_policy((
        hl_training_state.normalizer_params,
        hl_training_state.params.policy,
        hl_training_state.params.value,
    ))

    ll_policy = make_ll_policy((
        ll_training_state.normalizer_params,
        ll_training_state.params.policy,
        ll_training_state.params.value,
    ))

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      next_state, hl_data, ll_data = acting_hierarchical.generate_unroll(
          env,
          current_state,
          hl_policy,
          ll_policy,
          current_key,
          unroll_length,
          extra_fields=('truncation', 'episode_metrics', 'episode_done'),
      )
      return (next_state, next_key), (hl_data, ll_data)

    (state, _), (hl_data, ll_data) = jax.lax.scan(
        f,
        (state, key_generate_unroll),
        (),
        length=batch_size * num_minibatches // num_envs,
    )
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    hl_data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), hl_data)
    hl_data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), hl_data
    )
    assert hl_data.discount.shape[1:] == (unroll_length,)

    ll_data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), ll_data)
    ll_data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), ll_data
    )
    assert ll_data.discount.shape[1:] == (unroll_length,)

    if log_training_metrics:  # log unroll metrics
      jax.debug.callback(
          metrics_aggregator.update_episode_metrics,
          hl_data.extras['state_extras']['episode_metrics'],
          hl_data.extras['state_extras']['episode_done'],
      )

    # Update normalization params and normalize observations.
    hl_normalizer_params = running_statistics.update(
        hl_training_state.normalizer_params,
        _remove_pixels(hl_data.observation),
        pmap_axis_name=_PMAP_AXIS_NAME,
    )
    ll_normalizer_params = running_statistics.update(
        ll_training_state.normalizer_params,
        _remove_pixels(ll_data.observation),
        pmap_axis_name=_PMAP_AXIS_NAME,
    )

    ((hl_optimizer_state, ll_optimizer_state), (hl_params, ll_params), _), metrics = jax.lax.scan(
        functools.partial(
            hierarchy_sgd_step, hl_data=hl_data, ll_data=ll_data,
            hl_normalizer_params=hl_normalizer_params, ll_normalizer_params=ll_normalizer_params
        ),
        ((hl_training_state.optimizer_state, ll_training_state.optimizer_state),
         (hl_training_state.params, ll_training_state.params), key_sgd),
        (),
        length=num_updates_per_batch,
    )

    new_hl_training_state = TrainingState(
        optimizer_state=hl_optimizer_state,
        params=hl_params,
        normalizer_params=hl_normalizer_params,
        env_steps=hl_training_state.env_steps + env_step_per_training_step,
    )

    new_ll_training_state = TrainingState(
        optimizer_state=ll_optimizer_state,
        params=ll_params,
        normalizer_params=ll_normalizer_params,
        env_steps=ll_training_state.env_steps + env_step_per_training_step,
    )
    return ((new_hl_training_state, new_ll_training_state), state, new_key), metrics

  def training_epoch(
      hl_training_state: TrainingState, ll_training_state: TrainingState, state: envs.State, key: PRNGKey
  ) -> Tuple[Tuple[TrainingState, TrainingState], envs.State, Metrics]:
    ((hl_training_state, ll_training_state), state, _), loss_metrics = jax.lax.scan(
        hierarchy_training_step,
        ((hl_training_state, ll_training_state), state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return (hl_training_state, ll_training_state), state, loss_metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def hierarchy_training_epoch_with_timing(
      hl_training_state: TrainingState,
      ll_training_state: TrainingState,
      env_state: envs.State, key: PRNGKey
  ) -> Tuple[Tuple[TrainingState, TrainingState], envs.State, Metrics]:
    nonlocal training_walltime
    t = time.time()
    hl_training_state, ll_training_state, env_state = _strip_weak_type((hl_training_state,
                                                                        ll_training_state,
                                                                        env_state))
    result = training_epoch(hl_training_state, ll_training_state, env_state, key)
    (hl_training_state, ll_training_state), env_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        num_training_steps_per_epoch
        * env_step_per_training_step
        * max(num_resets_per_eval, 1)
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return (hl_training_state, ll_training_state), env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

  # region Initialize model params and training state
  hl_init_params = ppo_losses.PPONetworkParams(
      policy=hl_ppo_network.policy_network.init(key_policy),
      value=hl_ppo_network.value_network.init(key_value),
  )

  obs_shape = jax.tree_util.tree_map(  # TODO: check if we need sub obs shapes?
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )
  hl_training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=hl_optimizer.init(hl_init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=hl_init_params,
      normalizer_params=running_statistics.init_state(
          _remove_pixels(obs_shape)
      ),
      env_steps=0,
  )

  ll_init_params = ppo_losses.PPONetworkParams(
      policy=ll_ppo_network.policy_network.init(key_policy),
      value=ll_ppo_network.value_network.init(key_value),
  )

  ll_training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=ll_optimizer.init(ll_init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=ll_init_params,
      normalizer_params=running_statistics.init_state(
          _remove_pixels(obs_shape)
      ),
      env_steps=0,
  )
  # endregion

  # region Load checkpoints
  if hl_restore_checkpoint_path is not None:
    hl_params = checkpoint.load(hl_restore_checkpoint_path)
    value_params = hl_params[2] if restore_value_fn else hl_init_params.value
    hl_training_state = hl_training_state.replace(
        normalizer_params=hl_params[0],
        params=hl_training_state.params.replace(
            policy=hl_params[1], value=value_params
        ),
    )

  if hl_restore_params is not None:
    logging.info('Restoring High-level TrainingState from `restore_params`.')
    value_params = hl_restore_params[2] if restore_value_fn else hl_init_params.value
    hl_training_state = hl_training_state.replace(
        normalizer_params=hl_restore_params[0],
        params=hl_training_state.params.replace(
            policy=hl_restore_params[1], value=value_params
        ),
    )

  if ll_restore_checkpoint_path is not None:
    ll_params = checkpoint.load(ll_restore_checkpoint_path)
    value_params = ll_params[2] if restore_value_fn else ll_init_params.value
    ll_training_state = ll_training_state.replace(
        normalizer_params=ll_params[0],
        params=ll_training_state.params.replace(
            policy=ll_params[1], value=value_params
        ),
    )

  if ll_restore_params is not None:
    logging.info('Restoring Low-level TrainingState from `restore_params`.')
    value_params = ll_restore_params[2] if restore_value_fn else ll_init_params.value
    ll_training_state = ll_training_state.replace(
        normalizer_params=ll_restore_params[0],
        params=ll_training_state.params.replace(
            policy=ll_restore_params[1], value=value_params
        ),
    )
  # endregion

  # region Early return if no training needed
  if num_timesteps == 0:
    return (
        make_hl_policy,
        (
            hl_training_state.normalizer_params,
            hl_training_state.params.policy,
            hl_training_state.params.value,
        ),
        make_ll_policy,
        (
            ll_training_state.normalizer_params,
            ll_training_state.params.policy,
            ll_training_state.params.value,
        ),
        {},
    )
  # endregion

  # region Move training states to GPU
  hl_training_state = jax.device_put_replicated(
      hl_training_state, jax.local_devices()[:local_devices_to_use]
  )

  ll_training_state = jax.device_put_replicated(
      ll_training_state, jax.local_devices()[:local_devices_to_use]
  )
  # endregion

  eval_env = _maybe_wrap_env(
      eval_env or environment,
      wrap_env,
      num_eval_envs,
      episode_length,
      action_repeat,
      local_device_count=1,  # eval on the host only
      key_env=eval_key,
      wrap_env_fn=wrap_env_fn,
      randomization_fn=randomization_fn,
  )
  evaluator = acting_hierarchical.Evaluator(
      eval_env,
      functools.partial(make_hl_policy, deterministic=deterministic_eval),
      functools.partial(make_ll_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )

  # region Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1:
    metrics = evaluator.run_evaluation(
        _unpmap((
            hl_training_state.normalizer_params,
            hl_training_state.params.policy,
            hl_training_state.params.value,
        )),
        _unpmap((
            ll_training_state.normalizer_params,
            ll_training_state.params.policy,
            ll_training_state.params.value,
        )),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)
  # endregion

  # region Run training
  training_metrics = {}
  training_walltime = 0
  current_step = 0
  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      ((hl_training_state, ll_training_state), env_state, training_metrics) = (
          hierarchy_training_epoch_with_timing(hl_training_state, ll_training_state, env_state, epoch_keys)
      )
      current_step = int(_unpmap(hl_training_state.env_steps))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
      )(key_envs, key_envs.shape[1])
      # TODO: move extra reset logic to the AutoResetWrapper.
      env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

    if process_id != 0:
      continue

    # region Process params after epoch
    # Process id == 0.
    hl_params = _unpmap((
        hl_training_state.normalizer_params,
        hl_training_state.params.policy,
        hl_training_state.params.value,
    ))

    ll_params = _unpmap((
        ll_training_state.normalizer_params,
        ll_training_state.params.policy,
        ll_training_state.params.value,
    ))

    hl_policy_params_fn(current_step, make_hl_policy, hl_params)
    ll_policy_params_fn(current_step, make_ll_policy, ll_params)
    # endregion

    if hl_save_checkpoint_path is not None:
      checkpoint.save(
          hl_save_checkpoint_path, current_step, hl_params, hl_ckpt_config
      )

    if ll_save_checkpoint_path is not None:
      checkpoint.save(
          ll_save_checkpoint_path, current_step, ll_params, ll_ckpt_config
      )

    if num_evals > 0:
      metrics = evaluator.run_evaluation(
          hl_params,
          ll_params,
          training_metrics,
      )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  assert total_steps >= num_timesteps

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(hl_training_state)
  pmap.assert_is_replicated(ll_training_state)
  hl_params = _unpmap((
      hl_training_state.normalizer_params,
      hl_training_state.params.policy,
      hl_training_state.params.value,
  ))
  ll_params = _unpmap((
      hl_training_state.normalizer_params,
      hl_training_state.params.policy,
      hl_training_state.params.value,
  ))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return ((make_hl_policy, make_ll_policy), (hl_params, ll_params), metrics)
