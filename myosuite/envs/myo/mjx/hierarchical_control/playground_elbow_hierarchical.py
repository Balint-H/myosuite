from datetime import datetime
from typing import Any, Dict, Optional, Union, Tuple

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from hierarchical_env import HierarchicalEnv
from mujoco_playground._src import mjx_env


def default_config() -> config_dict.ConfigDict:
    env_config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        healthy_angle_range=(0, 2.1),
        pd_config = config_dict.create(kp=1, kd=1),
        noise_config=config_dict.create(
            reset_noise_scale=1e-1,
        ),
        reward_config=config_dict.create(
            angle_reward_weight=2.5,  # HL reward weight for reaching sampled target
            ctrl_cost_weight=0.1,      # HL control cost weight
            # --- LL Reward Weights (Example - customize as needed) ---
            # ll_angle_reward_weight=2.5, # LL reward for reaching HL target angle
            # ll_ctrl_cost_weight=0.1,   # LL control cost weight
        )
    )

    # Note: RL config might need adjustment for hierarchical setup
    # e.g., separate network factories if HL/LL use different inputs/architectures
    rl_config = config_dict.create(
        num_timesteps=100_000_000,
        num_evals=10,
        reward_scaling=1.0,
        episode_length=env_config.episode_length,
        clipping_epsilon=0.2,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        num_resets_per_eval=1,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0.005,
        num_envs=8192,
        batch_size=256,
        max_grad_norm=1.0,
        hl_network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        ll_network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
        )

    )
    env_config["ppo_config"] = rl_config
    return env_config


class HierarchicalPlaygroundElbow(HierarchicalEnv):
    """Hierarchical elbow environment with internal PD + HL modulation."""

    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            is_msk=True,
            xml_path: Optional[str] = None, # Allow passing xml path
            reference_trajectory: Optional[Tuple[jp.ndarray, jp.ndarray]] = None # Allow passing trajectory
    ) -> None:
        super().__init__(config, config_overrides)
        xml_path = rf"../assets/elbow/myoelbow_1dof{6 if is_msk else 0}muscles_mjx.xml"
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
            # Placeholder: Generate a simple sine wave trajectory
            self._qpos_ref, self._qvel_ref = self._generate_placeholder_trajectory()

        self._ref_traj_len = self._qpos_ref.shape[0]

        # --- PD Gains ---
        self.kp = self._config.pd_gains.kp
        self.kd = self._config.pd_gains.kd

    def _generate_placeholder_trajectory(self) -> Tuple[jp.ndarray, jp.ndarray]:
        """Generates a simple sine wave reference trajectory."""
        cfg = config_dict.create(amplitude=1, frequency=0.5, offset=0.5)
        num_steps = self._config.episode_length # Assuming ref traj matches episode length
        times = jp.arange(num_steps) * self._config.ctrl_dt # Time based on control dt
        qpos_ref = cfg.amplitude * jp.sin(2 * jp.pi * cfg.frequency * times) + cfg.offset
        qvel_ref = cfg.amplitude * 2 * jp.pi * cfg.frequency * jp.cos(2 * jp.pi * cfg.frequency * times)
        # Reshape to (time, dim) - assuming 1 DoF
        return qpos_ref[:, None], qvel_ref[:, None]

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng_noise = jax.random.split(rng)

        # Initial position/velocity noise
        low, hi = -self._config.noise_config.reset_noise_scale, self._config.noise_config.reset_noise_scale
        qpos_noise = jax.random.uniform(rng_noise, (self.mjx_model.nq,), minval=low, maxval=hi)

        rng_noise, rng_vel = jax.random.split(rng_noise)
        qvel_noise = jax.random.uniform(rng_vel, (self.mjx_model.nv,), minval=low, maxval=hi)

        qpos = self.mjx_model.qpos0 + qpos_noise
        qvel = jp.zeros(self.mjx_model.nv) + qvel_noise # Start with zero velocity + noise

        # Initial data state
        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.mjx_model.nu), act=jp.zeros(self.mjx_model.na)) # Reset activation
        data = mjx.forward(self.mjx_model, data) # Compute initial derived quantities

        # Initial reference trajectory time index
        ref_time_idx = 0
        qpos_ref_t = self._qpos_ref[ref_time_idx]
        qvel_ref_t = self._qvel_ref[ref_time_idx]

        # Calculate initial PD errors
        raw_pos_error = qpos_ref_t - data.qpos
        raw_vel_error = qvel_ref_t - data.qvel

        # Initial info dictionary
        info = {
            'rng': rng,
            'ref_time_idx': ref_time_idx,
            'raw_pos_error': raw_pos_error,
            'raw_vel_error': raw_vel_error,

            # Placeholders for fields for the LL loss calculation
            'desired_torque': jp.zeros(self.mjx_model.nv),
            'actual_torque': jp.zeros(self.mjx_model.nv),
            'jac_torque_act': jp.zeros((self.mjx_model.nv, self.mjx_model.na)),
        }

        # Get initial hierarchical observations
        obs = self._get_obs(data, info)

        # Initial reward, done, metrics
        reward, done, zero = jp.zeros(3)
        metrics = {
            'angle_reward': zero,
            'reward_quadctrl': zero,
            'll_torque_error': zero, # Metric for LL performance
        }

        return State(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    def high_level_step(self, state: State, hl_action: jp.ndarray) -> State:
        """Calculates desired torque based on HL modulation of PD error."""
        # hl_action is the HL_modulation signal
        data = state.pipeline_state
        info = state.info

        # Get current PD errors from info (calculated in previous step or reset)
        raw_pos_error = info['raw_pos_error']
        raw_vel_error = info['raw_vel_error']

        # Modulate position error
        modulated_pos_error = raw_pos_error + hl_action

        # Calculate desired torque
        desired_torque = self.kp * modulated_pos_error + self.kd * raw_vel_error

        # Prepare observations for the LL policy
        # Base observations from current state
        ll_obs_base = self._get_ll_obs_base(data)
        # Add desired torque
        ll_observation = {**ll_obs_base, 'desired_torque': desired_torque}

        # Update the observation dict in the state
        # HL obs likely remains the same until the next physics step updates qpos/qvel
        current_obs = state.obs
        new_obs = {
            'hl_obs': current_obs['hl_obs'], # Keep previous HL obs
            'll_observation': ll_observation # Update LL obs with desired torque
        }

        # Update info dictionary
        new_info = info.copy()
        new_info['desired_torque'] = desired_torque
        # Note: raw errors for the *next* step will be calculated in env.step

        # Return the updated state (only obs and info modified, pipeline_state unchanged)
        return state.replace(obs=new_obs, info=new_info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Applies LL ctrl action, steps physics, calculates rewards and next state."""
        # state here is the mid_state from actor_step (already processed by high_level_step)
        # action here is the LL ctrl signal
        data = state.pipeline_state

        # --- Apply LL Control and Step Physics ---
        data = data.replace(ctrl=action)
        # Handle action repeat if necessary (mjx.step doesn't do it automatically)
        def physics_step(data_i, _):
            data_i = data_i.replace(ctrl=action) # Ensure ctrl is applied each substep if needed
            return mjx.step(self.mjx_model, data_i), None

        next_data, _ = jax.lax.scan(physics_step, data, (), length=self._config.action_repeat)
        # next_data = mjx.step(self.mjx_model, data) # If action_repeat=1

        # --- Calculate Results ---
        actual_torque = next_data.qfrc_actuator # Torque resulting from LL ctrl

        # Calculate Jacobian d(torque)/d(act) based on the state *before* the step
        # This state (`data`) had the correct activation (`act`) based on previous `ctrl`
        jac_torque_act = self.calculate_torque_activation_jacobian(data)

        # --- Calculate HL Reward ---
        # Example: Reward based on tracking the reference trajectory point for *this* step
        # Could also be based on a different overall task goal
        ref_time_idx = state.info['ref_time_idx']
        qpos_ref_t = self._qpos_ref[ref_time_idx]
        angle_error = qpos_ref_t[0] - next_data.qpos[0] # Use next state's position
        angle_reward = jp.exp(-self._config.reward_config.angle_reward_weight * angle_error**2)
        # TODO: Decide what ctrl_cost applies to (HL modulation or LL ctrl?)
        # Using LL ctrl cost here as an example:
        ll_ctrl_cost = self._config.reward_config.ctrl_cost_weight * jp.sum(jp.square(action))
        reward = angle_reward - ll_ctrl_cost

        # --- Update State for Next Step ---
        # Update reference trajectory index
        next_ref_time_idx = (ref_time_idx + 1) % self._ref_traj_len

        # Calculate PD errors for the *next* timestep (used in next high_level_step)
        next_qpos_ref_t = self._qpos_ref[next_ref_time_idx]
        next_qvel_ref_t = self._qvel_ref[next_ref_time_idx]
        next_raw_pos_error = next_qpos_ref_t - next_data.qpos
        next_raw_vel_error = next_qvel_ref_t - next_data.qvel

        # Prepare next observations
        next_info_for_obs = {'ref_qpos': next_qpos_ref_t, 'ref_qvel': next_qvel_ref_t} # Pass ref info
        next_obs = self._get_obs(next_data, next_info_for_obs)

        # Update metrics
        current_metrics = state.metrics
        ll_torque_error_val = jp.mean(jp.square(actual_torque - state.info['desired_torque']))
        current_metrics.update(
            angle_reward=angle_reward,
            reward_quadctrl=-ll_ctrl_cost,
            ll_torque_error=ll_torque_error_val,
        )

        # Update info dictionary for the final returned state
        next_info = state.info.copy() # Start with info from mid_state
        next_info.update({
            'rng': state.info['rng'], # Pass RNG state along if needed elsewhere
            'ref_time_idx': next_ref_time_idx,
            'raw_pos_error': next_raw_pos_error,
            'raw_vel_error': next_raw_vel_error,
            'actual_torque': actual_torque,
            'jac_torque_act': jac_torque_act,
            # Keep desired_torque from mid_state info if needed for logging
            # 'desired_torque': state.info['desired_torque']
        })

        # Check for episode termination (e.g., time limit handled by wrapper)
        done = jp.array(0.0) # Add conditions if needed (e.g., joint limits)

        return state.replace(
            pipeline_state=next_data, obs=next_obs, reward=reward, done=done,
            metrics=current_metrics, info=next_info
        )

    def _get_hl_obs(self, data: mjx.Data, info: Dict) -> jp.ndarray:
        """Get observations for the high-level policy."""
        # Example: current state and potentially reference state info
        # Reference info might come from the passed `info` dict
        ref_qpos = info.get('ref_qpos', jp.zeros_like(data.qpos)) # Get ref info if available
        ref_qvel = info.get('ref_qvel', jp.zeros_like(data.qvel))
        return jp.concatenate([
            data.qpos,
            data.qvel,
            ref_qpos, # Include reference state
            ref_qvel
        ])

    def _get_ll_obs_base(self, data: mjx.Data) -> Dict[str, jp.ndarray]:
       """Get base state observations for the low-level policy (before desired_torque)."""
       # Example: proprioceptive info relevant for muscle control
       return {
           'state': jp.concatenate([
               data.ctrl,
               data.actuator_length,
               data.actuator_velocity,
           ])
           # Add other necessary keys if ll_network expects them
       }

    def _get_obs(
            self, data: mjx.Data, info: Dict
    ) -> Dict[str, Union[jp.ndarray, Dict[str, jp.ndarray]]]:
        """Gets the hierarchical observation dictionary."""
        hl_obs = self._get_hl_obs(data, info)
        ll_obs_base = self._get_ll_obs_base(data)
        # Note: 'desired_torque' is NOT added here, it's added in high_level_step
        return {'hl_obs': hl_obs, 'll_obs_base': ll_obs_base}

    #TODO needs to be completely redone
    def calculate_torque_activation_jacobian(self, data: mjx.Data) -> jp.ndarray:
        """Calculates d(torque)/d(act) using JAX AD."""

        # Define function to compute torque from act and static state
        def compute_torque_from_act(act_input, qpos, qvel, model):
            # Create data struct with inputs, ensure other needed fields are present
            # NOTE: mjx.forward recomputes everything, including kinematics.
            # If possible, more efficient calculation might only recompute
            # actuator lengths/velocities/forces based on qpos/qvel/act.
            # This requires deeper knowledge of mjx internal functions.
            # Using mjx.forward is simpler but potentially less efficient here.
            temp_data = data.replace(act=act_input, qpos=qpos, qvel=qvel)
            temp_data = mjx.forward(model, temp_data)
            return temp_data.qfrc_actuator # actuator forces == joint torques for muscles

        # Compute Jacobian w.r.t first arg (act_input)
        jacobian_fn = jax.jacfwd(compute_torque_from_act, argnums=0)

        # Pass static arguments separately if JITting
        jacobian = jacobian_fn(data.act, data.qpos, data.qvel, self.mjx_model)

        expected_shape = (self.mjx_model.nv, self.mjx_model.na)
        # assert jacobian.shape == expected_shape # Shape assertion might fail with JIT
        # Use shape check outside JIT or rely on downstream code
        # print(f"Jacobian shape: {jacobian.shape}")

        return jacobian


    # --- Properties ---
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        """Returns the size of the low-level action space (ctrl)."""
        return self._mjx_model.nu

    @property
    def high_level_action_size(self) -> int:
        """Returns the size of the high-level action space (modulation)."""
        # Assuming modulation has same dim as DoFs
        return self.mjx_model.nv

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model