# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""Base classes for Berkeley Humanoid."""
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
  return config_dict.create(
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

class PlaygroundElbow(mjx_env.MjxEnv):
    """Made using the Berkeley Humanoid environment as a template."""

    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            is_msk=True
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
            rng3, (1,), minval=self._config.healthy_angle_range[0], maxval=self._config._healthy_angle_range[1]
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
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        angle_error = state.info['target_angle'][0] - data.qpos[0]
        # Smooth fall-off on angle reward. Exp is too costly normally,
        # should replace it later on.
        angle_reward = jp.exp(-self._config.reward_config.angle_reward_weight * angle_error * angle_error)
        ctrl_cost = self._config.reward_config.ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, action, state.info)
        reward = angle_reward - ctrl_cost
        done = 0.0
        state.metrics.update(
            angle_reward=angle_reward,
            reward_quadctrl=-ctrl_cost,
        )

        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done
        )

    def _get_obs(
            self, data: mjx.Data, action: jp.ndarray, info
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


def main(is_msk=True):
    envs.register_environment('elbow', Elbow)

    """## Train Elbow Policy

    Let's now train a policy with PPO to move the elbow to a target angle. Training takes about 9-10 minutes on a Tesla A100 GPU.
    """

    print("Building environment")

    env_name = 'elbow'
    env = envs.get_environment(env_name, is_msk=is_msk)

    def check_env(model, data):
        obs = env._get_obs(data, data.ctrl)
        assert not np.any(np.isnan(obs))
        angle_error = 1 - data.qpos[0]
        angle_reward = np.exp(-env._angle_reward_weight * angle_error * angle_error)
        ctrl_cost = env._ctrl_cost_weight * np.sum(np.square(data.ctrl))
        reward = angle_reward - ctrl_cost
        data.ctrl = np.random.uniform(-1, 1, (env.action_size,))
        assert not np.isnan(reward)

    train_fn = functools.partial(
        ppo.train, num_timesteps=20_000_000, num_evals=5, reward_scaling=0.1,
        episode_length=1000, normalize_observations=True, action_repeat=1,
        unroll_length=10, num_minibatches=1, num_updates_per_batch=8,
        discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=10,
        batch_size=10, seed=0)

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    max_y, min_y = 5000, 0

    # Plot learning curves
    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])

        plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={y_data[-1]:.3f}')

        plt.errorbar(
            x_data, y_data, yerr=ydataerr)
        plt.savefig(f'{num_steps}.png')

    # Instantiate the environment then train
    print("Jitting then training")
    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

    # Save Model
    model_path = './elbow_params.pickle'
    model.save_params(model_path, params)


if __name__ == '__main__':
    main()