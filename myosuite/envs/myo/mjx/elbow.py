import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)
from datetime import datetime
import functools
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mujoco
from mujoco import mjx
from brax import envs
from brax.envs.base import Env, PipelineEnv, State
from brax.training.agents.ppo import train as ppo
from brax.io import mjcf, model
import mujoco.viewer

print(f"Current backend: {jax.default_backend()}")

class Elbow(PipelineEnv):

  def __init__(
      self,
      angle_reward_weight=2.5,
      ctrl_cost_weight=0.1,
      healthy_angle_range=(0, 2.1),
      reset_noise_scale=1e-1,
      is_msk=True,
      **kwargs,
  ):
    path = rf"./myosuite/envs/myo/assets/elbow/myoelbow_1dof{6 if is_msk else 0}muscles_mjx.xml"
    mj_model = mujoco.MjModel.from_xml_path(path)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6
    mj_model.opt.disableflags = mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

    sys = mjcf.load_model(mj_model)

    physics_steps_per_control_step = 5
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self._angle_reward_weight = angle_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_angle_range = healthy_angle_range
    self._reset_noise_scale = reset_noise_scale

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
    
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    target_angle = jax.random.uniform(
        rng3, (1,), minval=self._healthy_angle_range[0], maxval=self._healthy_angle_range[1]
        )

    info = {'rng': rng, 'target_angle': target_angle}

    data = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(data, jp.zeros(self.sys.nu))
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
    angle_reward = jp.exp(-self._angle_reward_weight*angle_error*angle_error)
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(data, action)
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
      self, data: mjx.Data, action: jp.ndarray
  ) -> jp.ndarray:
    """Observes humanoid body position, velocities, and angles."""
    position = data.qpos

    # external_contact_forces are excluded
    return jp.concatenate([
        position,
        data.qvel,
        data.qfrc_actuator,
    ])


def main(is_msk=True):
    envs.register_environment('elbow', Elbow)

    """## Train Elbow Policy
    
    Let's now train a policy with PPO to make the Humanoid run forwards. Training takes about 9-10 minutes on a Tesla A100 GPU.
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

    mj_model = env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)
    mujoco.set_mjcb_control(check_env)
    mujoco.viewer.launch(mj_model, mj_data)

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


    # instantiate the environment
    print("Jitting then training")
    make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')


    #Save Model
    model_path = './elbow_params.pickle'
    model.save_params(model_path, params)

if __name__ == '__main__':
    main()