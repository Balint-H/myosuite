# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction
import jax
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as viewer
import numpy as np
from brax import envs
from brax.training.acme import running_statistics
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from video import images_to_video
import matplotlib
matplotlib.use('qtagg')
from functools import partial
from brax.training.agents.ppo import losses as ppo_losses

from myo_legs import MyoLeg

xml = rf"../../../simhive/myo_sim/leg/myolegs_abdomen.xml"

def deterministic_policy (input_data):

    logits = ppo_network.policy_network.apply(*params[:2], np.array([input_data], dtype=np.float32))
    brax_result = ppo_network.parametric_action_distribution.mode(logits)
    return brax_result


def get_obs(data):
    """Observes humanoid body position, velocities, and angles."""
    position = data.qpos
    position = position[2:]

    # external_contact_forces are excluded
    return np.concatenate([
        position,
        data.qvel,
        data.cinert[1:].ravel(),
        data.cvel[1:].ravel(),
        data.qfrc_actuator,
    ])

i = 0

def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system
    global i
    if i%5 == 0:
        i += 1
    observations = get_obs(data)
    data.ctrl = deterministic_policy(observations)
    i+=1
    pass


def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model: mujoco.MjModel = mujoco.MjModel.from_xml_path(filename=xml, assets=None)

    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    physics_steps_per_control_step = 5
    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)

    if model is not None:
        # Can set initial state

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(arm_control)

    return model, data


use_mjx=True
reset_params=False

ppo_network = ppo_networks.make_ppo_networks(
      366,
      86,
      preprocess_observations_fn=normalize)
model_path = '../params/leg_params.pickle'
params = model.load_params(model_path)
if reset_params:
    key = jax.random.PRNGKey(0)
    key_policy, key_value= jax.random.split(key)
    params = [running_statistics.init_state(np.zeros(366)), ppo_network.policy_network.init(key_policy), ppo_network.value_network.init(key_value)]

def mjx_visu():
    envs.register_environment('myoLeg', MyoLeg)
    eval_env = envs.get_environment('myoLeg')

    make_policy = ppo_networks.make_inference_fn(ppo_network)

    inference_fn = make_policy(params)
    jit_inference_fn = jax.jit(inference_fn)

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    # grab a trajectory
    n_steps = 1000
    render_every = 2
    rewards = {
        "forward_reward" : [],
        "reward_alive": [],
        "reward_quadctrl": []
    }

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        for key in rewards.keys():
            rewards[key].append(state.metrics[key])
        if state.done:
            break
    for key in rewards.keys():
        plt.plot(rewards[key], label=key)

    plt.legend()

    images_to_video(eval_env.render(rollout[::render_every], camera="side_view", height=480, width=620, ),
                    fps=int(1.0 / eval_env.dt / render_every))

    plt.show()
    print(len(rewards['reward_alive']))
    print(np.sum([np.sum(rewards[k]) for k in rewards.keys()]))


if __name__ == '__main__':
    if not use_mjx:
        viewer.launch(loader=load_callback)
    else:
        mjx_visu()
