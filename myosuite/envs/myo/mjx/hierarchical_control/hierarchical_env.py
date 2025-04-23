from flax import struct
from datetime import datetime
from typing import Any, Dict, Optional, Union, Callable, Tuple, NamedTuple
from brax.envs.wrappers import training as brax_training
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
import abc
from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper, BraxDomainRandomizationVmapWrapper, BraxAutoResetWrapper


class HierarchicalEnv(mjx_env.MjxEnv, abc.ABC):
  @abc.abstractmethod
  def high_level_step(self, state: State, action: jax.Array) -> State:
    """Run high-level control, calculate input to low-level systems. Don't run dynamics yet."""

  @property
  @abc.abstractmethod
  def high_level_action_size(self) -> int:
    """Returns the size of the high-level action space (e.g., target angle)."""


class LLSupervisedData(NamedTuple):
  """Data collected for training the low-level supervised policy."""
  ll_observation: Dict[str, jp.ndarray]
  ctrl: jp.ndarray
  desired_torque: jp.ndarray
  actual_torque: jp.ndarray
  # Pre-computed Jacobian: d(torque)/d(act)
  jacobian: jp.ndarray


class HierarchicalBraxDomainRandomizationVmapWrapper(BraxDomainRandomizationVmapWrapper):
  """Brax wrapper for domain randomization of hierarchical environment."""

  def __init__(
      self,
      env: HierarchicalEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):

    super().__init__(env, randomization_fn)

  def _env_fn(self, mjx_model: mjx.Model) -> HierarchicalEnv:
    env = self.env
    env.unwrapped._mjx_model = mjx_model
    return env

  def high_level_step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    def high_level_step(mjx_model, s, a):
      env = self._env_fn(mjx_model=mjx_model)
      return env.high_level_step(s, a)

    res = jax.vmap(high_level_step, in_axes=[self._in_axes, 0, 0])(
        self._mjx_model_v, state, action
    )
    return res


class HierarchicalVmapWrapper(brax_training.VmapWrapper):
  """Vectorizes q hierarchical Brax env."""

  def __init__(self, env: HierarchicalEnv, batch_size: Optional[int] = None):
    self.env: HierarchicalEnv = env
    super().__init__(env, batch_size)

  def high_level_step(self, state: State, action: jax.Array) -> State:
    return jax.vmap(self.env.high_level_step)(state, action)


class HierarchicalEpisodeWrapper(brax_training.EpisodeWrapper):

  def __init__(self, env: HierarchicalEnv, episode_length: int, action_repeat: int):
    self.env: HierarchicalEnv = env
    super().__init__(env, episode_length, action_repeat)

  def high_level_step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return self.env.high_level_step(state, action)


class HierarchicalBraxAutoResetWrapper(BraxAutoResetWrapper):
  def high_level_step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return self.env.high_level_step(state, action)


def wrap_for_hierarchical_brax_training(
    env: mjx_env.MjxEnv,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    vision: whether the environment will be vision based
    num_vision_envs: number of environments the renderer should generate,
      should equal the number of batched envs
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  if randomization_fn is None:
    env = HierarchicalVmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    env = HierarchicalBraxDomainRandomizationVmapWrapper(env, randomization_fn)
  env = HierarchicalEpisodeWrapper(env, episode_length, action_repeat)
  env = HierarchicalBraxAutoResetWrapper(env)
  return env


