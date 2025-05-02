from datetime import datetime
from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State

from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src


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
            angle_reward_weight=2.5,
            ctrl_cost_weight=0.1,
        )
    )

    rl_config = config_dict.create(
        num_timesteps=20_000_000,
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
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(512, 256, 128),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )
    )
    env_config["ppo_config"] = rl_config
    return env_config


class PlaygroundArm(mjx_env.MjxEnv):

    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            is_msk=True
    ) -> None:
        super().__init__(config, config_overrides)
        xml_path = rf"../assets/arm/myoarm_relocate_mjx.xml"

        spec = mujoco.MjSpec.from_file(xml_path)
        geoms = spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        ellipsoids = [g for g in geoms if g.type == 4]
        ellipsoids = [e for e in ellipsoids if e.classname.name != "wrap"]
        for e in ellipsoids:
            e.contype = 0
            e.conaffinity = 0
        self._mj_model = spec.compile()
        self._mj_model.opt.timestep = self.sim_dt

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path

        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.disableflags = self._mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._config.noise_config.reset_noise_scale, self._config.noise_config.reset_noise_scale
        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            rng1, (self.mjx_model.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.mjx_model.nv,), minval=low, maxval=hi
        )

        target_angle = jax.random.uniform(
            rng3, (1,), minval=self._config.healthy_angle_range[0], maxval=self._config.healthy_angle_range[1]
        )

        # We store the target angle in the info, can't store it as an instance variable,
        # as it has to be determined in a parallelized manner
        info = {'rng': rng, 'target_angle': target_angle}

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros((self.mjx_model.nu,)))

        obs = self._get_obs(data, jp.zeros(self.mjx_model.nu), info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'angle_reward': zero,
            'reward_quadctrl': zero,
        }
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.data
        data = mjx_env.step(self.mjx_model, data0, action)
        obs = self._get_obs(data, action, state.info)
        reward = self._get_reward(data, action, state.info)
        done = 0.0
        state.metrics.update(
            reward = reward
        )

        return state.replace(
            data=data, obs=obs, reward=reward, done=done
        )

    def _get_reward(self, data: mjx.Data, action: jp.ndarray, info
    ) -> jp.float32:
        return 0.

    def _get_obs(self, data: mjx.Data, action: jp.ndarray, info
    ) -> jp.ndarray:
        """Observes elbow angle, velocities, and last applied torque."""
        position = data.qpos

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            data.qvel,
            data.qfrc_actuator,
            info['target_angle']
        ])

    # Accessors.
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


