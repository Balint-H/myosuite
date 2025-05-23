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

"""Brax training acting functions."""

import time
from typing import Callable, Sequence, Tuple

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
import jax
import numpy as np

State = envs.State
Env = envs.Env


def actor_step(
    env: Env,
    env_state: State,
    hl_policy: Policy,
    ll_policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition, Transition]:
  """Collect data."""
  hl_key, ll_key, n_key = jax.random.split(key,3)
  hl_actions, hl_policy_extras = hl_policy(env_state.obs["hl_obs"], hl_key)
  env_state.obs['ll_obs']['hl_input'] = hl_actions
  actions, ll_policy_extras = ll_policy(env_state.obs['ll_obs'], ll_key)

  nstate = env.step(env_state, actions)
  nhl_actions, _ = hl_policy(nstate.obs["hl_obs"], hl_key)
  nstate.obs['ll_obs']['hl_input'] = nhl_actions

  state_extras = {x: nstate.info[x] for x in extra_fields}
  return (nstate,
          Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=env_state.obs["hl_obs"],
            action=hl_actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={'policy_extras': hl_policy_extras,
                    'state_extras': state_extras},
          ),
          Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
              observation=env_state.obs['ll_obs'],
              action=actions,
              reward=nstate.info['ll_reward'],
              discount=1 - nstate.done,
              next_observation=nstate.obs['ll_obs'],
              extras={'policy_extras': ll_policy_extras, 'state_extras': state_extras},
          ),
          )


def generate_unroll(
    env: Env,
    env_state: State,
    hl_policy: Policy,
    ll_policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, hl_transition, ll_transition = actor_step(
        env, state, hl_policy, ll_policy, current_key, extra_fields=extra_fields
    )
    return (nstate, next_key), (hl_transition, ll_transition)

  (final_state, _), (hl_data, ll_data) = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length
  )
  return final_state, hl_data, ll_data


# TODO: Consider moving this to its own file.
class Evaluator:
  """Class to run evaluations."""

  def __init__(
      self,
      eval_env: envs.Env,
      eval_hl_policy_fn: Callable[[PolicyParams], Policy],
      eval_ll_policy_fn: Callable[[PolicyParams], Policy],
      num_eval_envs: int,
      episode_length: int,
      action_repeat: int,
      key: PRNGKey,
  ):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.0

    eval_env = envs.training.EvalWrapper(eval_env)

    def generate_eval_unroll(
        hl_policy_params: PolicyParams, ll_policy_params: PolicyParams, key: PRNGKey
    ) -> State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_hl_policy_fn(hl_policy_params),
          eval_ll_policy_fn(ll_policy_params),
          key,
          unroll_length=episode_length // action_repeat,
      )[0]

    self._generate_eval_unroll = jax.jit(generate_eval_unroll)
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(
      self,
      hl_policy_params: PolicyParams,
      ll_policy_params: PolicyParams,
      training_metrics: Metrics,
      aggregate_episodes: bool = True,
  ) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state = self._generate_eval_unroll(hl_policy_params, ll_policy_params, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    for fn in [np.mean, np.std]:
      suffix = '_std' if fn == np.std else ''
      metrics.update({
          f'eval/episode_{name}{suffix}': (
              fn(value) if aggregate_episodes else value
          )
          for name, value in eval_metrics.episode_metrics.items()
      })
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics,
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray
