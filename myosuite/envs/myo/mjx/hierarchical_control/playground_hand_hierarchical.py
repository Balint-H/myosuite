from typing import Any, Dict, Optional, Union, Tuple
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from hierarchical_env import HierarchicalEnv, LLSupervisedData
from mujoco_playground._src import mjx_env


def default_config() -> config_dict.ConfigDict:
  env_config = config_dict.create(
    ctrl_dt=0.002,
    sim_dt=0.002,
    episode_length=300,
    action_repeat=1,
    action_scale=0.5,
    history_len=1,
    healthy_angle_range=(0, 2.1),
    noise_config=config_dict.create(
      reset_noise_scale=1e-1,
    ),
    reward_config=config_dict.create(
      angle_reward_weight=5,
      angle_reward_scale=2.5,
      ctrl_cost_weight=0.5,
      ctrl_cost_scale=0.01,
    ),
    pd_gains=config_dict.create(
      kp=4,
      kd=0.05
    )
  )

  rl_config = config_dict.create(
    num_timesteps=100_000_000,
    num_evals=16,
    episode_length=env_config.episode_length,
    hl_ppo_learning_config=config_dict.create(
      reward_scaling=1.0,
      clipping_epsilon=0.1,
      learning_rate=5e-5,
      entropy_cost=0.002,
      discounting=0.98,
      gae_lambda=0.95,
      max_grad_norm=1.0,
    ),
    normalize_observations=True,
    action_repeat=env_config.action_repeat,
    unroll_length=60,
    num_minibatches=64,
    num_updates_per_batch=2,
    num_resets_per_eval=1,
    num_envs=8192,
    batch_size=128,
    hl_network_factory=config_dict.create(
      policy_hidden_layer_sizes=(128, 64, 32),
      value_hidden_layer_sizes=(128, 64, 32),
      policy_obs_key="hl_obs",
      value_obs_key="hl_obs",
    ),
    ll_network_factory=config_dict.create(
      hidden_layer_sizes=(64, 32, 16),
      obs_key="ll_obs",
    ),
    ll_learning_config=config_dict.create(
      ll_opt_max_grad_norm=None,
      learning_rate=15e-5
    )
  )

  env_config["rl_config"] = rl_config
  return env_config


class HierarchicalPlaygroundElbow(HierarchicalEnv):
  """Hierarchical elbow environment with internal PD + HL modulation."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      is_msk=True,
      xml_path: Optional[str] = None,  # Allow passing xml path
      reference_trajectory: Optional[Tuple[jp.ndarray, jp.ndarray]] = None  # Allow passing trajectory
  ) -> None:
    super().__init__(config, config_overrides)
    xml_path = (rf"../../assets/elbow/myoelbow_1dof{6 if is_msk else 0}muscles_mjx.xml"
                if xml_path is None else xml_path)
    self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
    self._mj_model.opt.timestep = self.sim_dt

    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path

    self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    self._mj_model.opt.iterations = 6
    self._mj_model.opt.ls_iterations = 6
    self._mj_model.opt.disableflags = self._mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

    self._mjx_model = mjx.put_model(self._mj_model)

    # --- Reference Trajectory ---
    if reference_trajectory:
      self._qpos_ref, self._qvel_ref = reference_trajectory
    else:
      # Placeholder trajectory
      self._qpos_ref, self._qvel_ref = self._generate_placeholder_trajectory()

    self._ref_traj_len = self._qpos_ref.shape[0]
    self.kp = self._config.pd_gains.kp
    self.kd = self._config.pd_gains.kd

  def _generate_placeholder_trajectory(self) -> Tuple[jp.ndarray, jp.ndarray]:
    """Generates a simple sine wave reference trajectory."""
    cfg = config_dict.create(amplitude=1, frequency=0.01, offset=1.2)
    times = jp.arange(1/cfg.frequency/self._config.ctrl_dt) * self._config.ctrl_dt  # Time based on control dt
    qpos_ref = cfg.amplitude * jp.sin(2 * jp.pi * cfg.frequency * times) + cfg.offset
    qvel_ref = 0 * cfg.amplitude * 2 * jp.pi * cfg.frequency * jp.cos(2 * jp.pi * cfg.frequency * times)
    # Reshape assuming 1 DoF
    return qpos_ref[:, None], qvel_ref[:, None]

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng_noise = jax.random.split(rng)

    # Initial position/velocity noise
    low, hi = -self._config.noise_config.reset_noise_scale, self._config.noise_config.reset_noise_scale
    qpos_noise = jax.random.uniform(rng_noise, (self.mjx_model.nq,), minval=low, maxval=hi)

    rng_noise, rng_vel, rng_offset = jax.random.split(rng_noise,3)
    qvel_noise = jax.random.uniform(rng_vel, (self.mjx_model.nv,), minval=low, maxval=hi)

    qpos = self.mjx_model.qpos0 + qpos_noise
    qvel = jp.zeros(self.mjx_model.nv) + qvel_noise  # Start with zero velocity + noise

    # Initial data state
    data = mjx.make_data(self.mjx_model)
    data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.mjx_model.nu),
                        act=jp.zeros(self.mjx_model.na))  # Reset activation
    data = mjx.forward(self.mjx_model, data)  # Compute initial derived quantities

    # Initial reference trajectory time index
    ref_time_idx = jax.random.randint(rng_offset, shape=(1,),
                                      minval=0, maxval=self._ref_traj_len, dtype=jp.int16)[0]
    qpos_ref_t = self._qpos_ref[ref_time_idx]
    qvel_ref_t = self._qvel_ref[ref_time_idx]

    # Initial info dictionary
    info = {
      'rng': rng,
      'ref_time_idx': ref_time_idx,

      # Placeholders for fields for the LL loss calculation
      'desired_torque': jp.zeros(self.mjx_model.nv),
      'actual_torque': jp.zeros(self.mjx_model.nv),
      'jac_torque_act': jp.zeros((self.mjx_model.nv, self.mjx_model.na))
    }

    # Get initial hierarchical observations
    obs = {'hl_obs': self._get_hl_obs(data, info),
           'll_obs': self._get_ll_obs(data, info)
           }

    # Initial reward, done, metrics
    reward, done, zero = jp.zeros(3)
    metrics = {
      'angle_reward': zero,
      'reward_quadctrl': zero
    }

    return State(data, obs, reward, done, metrics, info)

  def hl_step(self, state: State, hl_action: jp.ndarray) -> State:
    """Calculates desired torque based on HL modulation of PD error."""
    # hl_action is the HL_modulation signal
    data = state.data

    ref_time_idx = state.info['ref_time_idx']  # This was incremented after the last physics step

    # Calculate PD errors for the *next* timestep (used in next high_level_step)
    qpos_ref_t = self._qpos_ref[ref_time_idx]
    qvel_ref_t = self._qvel_ref[ref_time_idx]

    # TODO: handle potential ball joints with quat_sub instead
    raw_pos_error = qpos_ref_t - data.qpos
    raw_vel_error = qvel_ref_t - data.qvel

    # Modulate position error
    modulated_pos_error = raw_pos_error + hl_action

    # Calculate desired torque
    desired_torque = self.kp * modulated_pos_error + self.kd * raw_vel_error

    state.info['desired_torque'] = desired_torque
    state.info['ref_time_idx'] = ref_time_idx

    # Prepare observations for the LL policy
    ll_obs = self._get_ll_obs(data, state.info)
    state.obs['ll_obs'] = ll_obs
    return state

  def step(self, state: State, action: jp.ndarray) -> State:
    """Applies LL ctrl action, steps physics, calculates rewards and next state."""
    data = state.data

    next_data = mjx_env.step(self.mjx_model, data, action, self.n_substeps)

    actual_torque = next_data.qfrc_actuator  # Torque resulting from LL ctrl

    # Calculate Jacobian d(torque)/d(act) based on the state *before* the step
    jac_torque_act = self.calculate_torque_activation_jacobian(data)

    # --- Calculate HL Reward ---
    # TODO axis for sum?
    ctrl_cost = (self._config.reward_config.ctrl_cost_weight
                 * jp.sum(jp.square(self._config.reward_config.ctrl_cost_scale * state.info['desired_torque'] ))
                 / self.mjx_model.nv
                 )

    state.info['ref_time_idx'] = (state.info['ref_time_idx'] + 1) % self._ref_traj_len
    qpos_ref_t = self._qpos_ref[state.info['ref_time_idx']]

    angle_error = qpos_ref_t - next_data.qpos
    angle_reward = (self._config.reward_config.angle_reward_weight
                    * jp.exp(-self._config.reward_config.angle_reward_scale
                             * jp.sum(jp.square(angle_error)) / self.mjx_model.nv))
    reward = angle_reward - ctrl_cost

    # Prepare next observations for HL controller
    next_info_for_obs = {'ref_qpos': self._qpos_ref[(state.info['ref_time_idx'] + 1) % self._ref_traj_len]}
    state.obs['hl_obs'] = self._get_hl_obs(next_data, next_info_for_obs)

    # Update metrics
    state.metrics['angle_reward'] = angle_reward
    state.metrics['reward_quadctrl'] = -ctrl_cost

    # Update info dictionary for the final returned state
    state.info['actual_torque'] = actual_torque
    state.info['jac_torque_act'] = jac_torque_act

    # TODO: termination conditions
    done = 0.0

    return state.replace(data=next_data, reward=reward, done=done)

  def _get_hl_obs(self, data: mjx.Data, info: Dict) -> jp.ndarray:
    """Get observations for the high-level policy."""
    ref_qpos = info.get('ref_qpos', jp.zeros_like(data.qpos))  # Get ref info if available
    ref_qvel = info.get('ref_qvel', jp.zeros_like(data.qvel))
    return jp.concatenate([
        data.qpos,
        data.qvel,
        ref_qpos,  # Include reference state
        ref_qvel
      ])

  def _get_ll_obs(self, data, info):
    return jp.concatenate([self._get_ll_obs_base(data), info['desired_torque']])

  def _get_ll_obs_base(self, data: mjx.Data) -> jp.ndarray:
    """Get base state observations for the low-level policy (before desired_torque)."""
    # Example: proprioceptive info relevant for muscle control
    return jp.concatenate([
        data.ctrl,
        data.actuator_length,
        data.actuator_velocity,
      ])


  def calculate_torque_activation_jacobian(self, data: mjx.Data) -> jp.ndarray:
    """Calculates d(torque)/d(act) analytically."""
    gains = mjx._src.scan.flat(
      self.mjx_model,
      mjx._src.support.muscle_gain,
      'uuuuu',
      'u',
      data.actuator_length,
      data.actuator_velocity,
      jp.array(self.mjx_model.actuator_lengthrange),
      jp.array(self.mjx_model.actuator_acc0),
      self.mjx_model.actuator_gainprm,
      group_by='u',
    )
    return gains[None, :] * data.actuator_moment.T

  # --- Properties ---
  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    """Returns the size of the low-level action space (ctrl)."""
    return self._mjx_model.nu

  @property
  def hl_action_size(self) -> int:
    """Returns the size of the high-level action space (modulation)."""
    return self.mjx_model.nv

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model


if __name__ == '__main__':
  env = HierarchicalPlaygroundElbow()
  jit_step = jax.jit(env.step)
  jit_reset = jax.jit(env.reset)
  jit_hl_step = jax.jit(env.hl_step)
  rng = jax.random.key(0)
  rng = jp.array([1708187461, 2772610763], dtype=jp.uint32)

  state = env.reset(rng)
  state = jit_hl_step(state, jp.array([0.0]))
  state = jit_step(state, jp.zeros(6))

  ll_data = LLSupervisedData(
    ll_obs=state.obs,
    activation_designated=jp.zeros(6),
    hl_desired_torque=state.info['desired_torque'],
    torque_designated=state.info['actual_torque'],
    # Pre-computed Jacobian: d(torque)/d(act)
    jacobian=state.info['jac_torque_act']
  )

  def dummy_batch(arr):
    return jp.repeat(jp.repeat(arr[None, None, ...], 256, 0), 20, 1)

  ll_data = jax.tree_util.tree_map(dummy_batch, ll_data)

  from loss_hierarchical import hierarchical_ll_loss
  import functools
  from brax.training import networks
  from brax.training.acme import running_statistics, types

  d_loss = jax.jacrev(hierarchical_ll_loss, has_aux=True)


  ll_network_fn = (
    networks.make_policy_network
  )

  ll_network_factory = functools.partial(
    ll_network_fn, **default_config().rl_config.ll_network_factory
  )

  obs_shape = jax.tree_util.tree_map(lambda x: x.shape, state.obs)

  ll_network = ll_network_factory(
    env.action_size, obs_shape, preprocess_observations_fn=running_statistics.normalize
  )

  obs_shape = jax.tree_util.tree_map(  # TODO: check if we need sub obs shapes?
    lambda x: jp.array(x.shape, jp.dtype('float32')), state.obs
  )
  params = ll_network.init(jax.random.PRNGKey(0))
  normalizer_params = running_statistics.init_state(obs_shape)

  grads = d_loss(params, normalizer_params, ll_data, ll_network)
  pass

