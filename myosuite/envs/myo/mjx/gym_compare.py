from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
from myosuite.utils import gym
from stable_baselines3 import PPO
from myosuite.envs.myo.mjx.playground_myoElbow import default_config

if __name__ == '__main__':

    config = default_config().ppo_config

    NUM_ENVS = 16  # Adjust based on available CPU cores
    EPISODE_LENGTH = 1000  # Max episode length

    # Function to create monitored environment instances
    def make_env():
        def _init():
            env = gym.make('myoElbowPose1D6MRandom-v0', reset_type='random')
            env = TimeLimit(env, max_episode_steps=EPISODE_LENGTH)  # Apply time limit
            env = Monitor(env)  # Monitor for logging episode rewards & lengths
            return env
        return _init

    # Create vectorized environments with monitoring
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env = VecMonitor(env)  # Ensures rollouts are logged

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy", env,
        n_steps=config.batch_size // NUM_ENVS,  # Adjust batch size per environment
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        ent_coef=config.entropy_cost,
        clip_range=0.3,
        max_grad_norm=config.max_grad_norm,
        gamma=config.discounting,
        policy_kwargs=dict(net_arch=[50, 50, 50]),
        n_epochs=config.num_updates_per_batch,
        device='cpu', verbose=1, tensorboard_log="./elbow_gym/"
    )

    # Add evaluation callback for rollout logging
    eval_env = DummyVecEnv([make_env()])  # Wrap in DummyVecEnv (since it's not parallel)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)  # Match training env
    eval_env = VecMonitor(eval_env)  # Ensures logging
    eval_callback = EvalCallback(eval_env, best_model_save_path="./elbow_gym/",
                                 log_path="./elbow_gym/", eval_freq=100_000, deterministic=True)

    # Train the model
    model.learn(40_000_000, progress_bar=True, callback=eval_callback)

    # Save the model
    model.save("elbow_gym")
