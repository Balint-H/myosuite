from datetime import datetime
from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State

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
        # Network factory config might need changes for HL/LL policies
        # This example assumes both use the same structure, which might not be ideal
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            # These keys might need updating based on the new obs structure
            # policy_obs_key="state",
            # value_obs_key="privileged_state",
        )
    )
    env_config["ppo_config"] = rl_config
    return env_config


class HierarchicalPlaygroundElbow(mjx_env.MjxEnv): # Renamed for clarity
    """Modified version for hierarchical control."""

    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            is_msk=True
    ) -> None:
        # Pass a copy to avoid modifying the original default config
        super().__init__(config.copy(), config_overrides)
        # Use epath for potentially better path handling
        xml_path = epath.Path(__file__).parent / f"../assets/elbow/myoelbow_1dof{6 if is_msk else 0}muscles_mjx.xml"
        self._xml_path = str(xml_path.resolve()) # Ensure absolute path string

        # Load Model
        try:
             self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not load XML: {self._xml_path}") from e

        self._mj_model.opt.timestep = self.sim_dt

        # Configure Solver (Example settings, adjust as needed)
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        # Consider if disabling Euler damping is desired
        self._mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP

        # Put model onto device using MJX
        self._mjx_model = mjx.put_model(self._mj_model)


    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state for hierarchical control."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._config.noise_config.reset_noise_scale, self._config.noise_config.reset_noise_scale
        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            rng1, (self.mjx_model.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.mjx_model.nv,), minval=low, maxval=hi
        )

        # Sample the ultimate target angle for the episode (used for HL reward)
        sampled_target_angle = jax.random.uniform(
            rng3, (1,), minval=self._config.healthy_angle_range[0], maxval=self._config.healthy_angle_range[1]
        )

        # Initial data state
        data = mjx.make_data(self.mjx_model) # Use mjx.make_data for MJX consistency
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros(self.mjx_model.nu))
        data = mjx.forward(self.mjx_model, data) # Ensure initial state is valid

        # Store episode-specific info
        # 'sampled_target_angle' is the final goal for the episode
        info = {'rng': rng, 'sampled_target_angle': sampled_target_angle}

        # Get initial hierarchical observations
        obs = self._get_obs(data, info) # Pass only data and info

        # Initial reward, done, metrics
        reward, done, zero = jp.zeros(3)
        metrics = {
            'angle_reward': zero,
            'reward_quadctrl': zero,
            'll_reward': zero, # Add placeholder for low-level reward metric
        }

        return State(data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics for hierarchical control."""
        # Action here is the LOW-LEVEL action (e.g., muscle activations)
        data0 = state.data
        data = mjx_env.step(self.mjx_model, data0, action) # Use the base class step if it handles action repeat etc.
        # If mjx_env.step doesn't exist or is insufficient, use mjx directly:
        # data = data0
        # for _ in range(self.action_repeat):
        #     data = data.replace(ctrl=action)
        #     data = mjx.step(self.mjx_model, data)


        # --- High-Level Reward Calculation ---
        # Error based on the episode's sampled target angle
        angle_error = state.info['sampled_target_angle'][0] - data.qpos[0]
        # Smooth fall-off on angle reward (consider alternatives if exp is too slow)
        angle_reward = jp.exp(-self._config.reward_config.angle_reward_weight * angle_error * angle_error)
        # Control cost for the low-level actions
        ctrl_cost = self._config.reward_config.ctrl_cost_weight * jp.sum(jp.square(action))
        hl_reward = angle_reward - ctrl_cost


        # --- Low-Level Reward Calculation (NEEDS DESIGN) ---
        # Option 1: LL gets the same reward as HL (simple starting point)
        ll_reward = hl_reward
        # Option 2: Reward LL for matching HL's command (requires HL action)
        # This requires accessing hl_action (which is state.obs['ll_obs']['hl_input'])
        # ll_target_angle = state.obs['ll_obs']['hl_input'] # Assumes HL action is target angle
        # ll_angle_error = ll_target_angle[0] - data.qpos[0]
        # ll_angle_reward = jp.exp(-self._config.reward_config.ll_angle_reward_weight * ll_angle_error**2)
        # ll_ctrl_cost = self._config.reward_config.ll_ctrl_cost_weight * jp.sum(jp.square(action))
        # ll_reward = ll_angle_reward - ll_ctrl_cost


        # Get next hierarchical observations
        obs = self._get_obs(data, state.info)

        # Update metrics
        # Use hl_reward for the main reward metric compatible with standard eval
        current_metrics = state.metrics
        current_metrics.update(
            angle_reward=angle_reward,
            reward_quadctrl=-ctrl_cost,
            ll_reward=ll_reward, # Track the calculated low-level reward
        )

        # Done is typically 0 unless episode ends early (e.g., unhealthy state)
        # Add termination conditions if needed:
        # is_healthy = ...
        # done = 1.0 - is_healthy
        done = 0.0

        return state.replace(
            data=data, obs=obs, reward=hl_reward, done=done, metrics=current_metrics
        )

    def _get_obs(
            self, data: mjx.Data, info: Dict # Removed action input
    ) -> Dict[str, jp.ndarray]:
        """Observes elbow state for hierarchical policies."""
        position = data.qpos
        velocity = data.qvel
        actuator_force = data.qfrc_actuator

        # High-level observations: Current state and the ultimate goal
        hl_obs = jp.concatenate([
            position,
            velocity,
            info['sampled_target_angle'] # Include the episode's target angle
        ])

        # Low-level observations: Current state + placeholder for HL command
        # The actual HL command ('hl_input') is added dynamically during rollout
        # in the acting script. It needs a defined size here.
        # Assuming HL action is a single float (target angle).
        hl_input_placeholder = jp.zeros(1) # Placeholder size MUST match HL action dim
        ll_obs_dict = {
            'state': jp.concatenate([
                position,
                velocity,
                actuator_force,
                # Add other relevant LL observations if needed
            ]),
            'hl_input': hl_input_placeholder
        }

        # Return hierarchical observation dictionary
        return {'hl_obs': hl_obs, 'll_obs': ll_obs_dict}

    # Accessors.
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        """Returns the size of the low-level action space."""
        return self._mjx_model.nu

    # Optional: Define HL action size if needed for network setup
    @property
    def hl_action_size(self) -> int:
        """Returns the size of the high-level action space (e.g., target angle)."""
        return 1 # Example: HL outputs a single target angle

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model