from typing import Dict, Tuple

import mujoco
import mujoco.viewer as viewer
import numpy as np
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from functools import partial
import jax.numpy as jp
from ml_collections.config_dict import config_dict

from mujoco import mjx

from hierarchical_env import make_ll_network
from playground_elbow_hierarchical import MjxElbow

xml = '../../assets/elbow/myoelbow_1dof6muscles_mjx_eval.xml'


def _generate_placeholder_trajectory() -> Tuple[jp.ndarray, jp.ndarray]:
  """Generates a simple sine wave reference trajectory."""
  cfg = config_dict.create(amplitude=1, frequency=0.01, offset=1.2)
  times = jp.arange(1 / cfg.frequency / 0.002) * 0.002  # Time based on control dt
  qpos_ref = cfg.amplitude * jp.sin(2 * jp.pi * cfg.frequency * times) + cfg.offset
  qvel_ref = 0 * cfg.amplitude * 2 * jp.pi * cfg.frequency * jp.cos(2 * jp.pi * cfg.frequency * times)
  # Reshape assuming 1 DoF
  return qpos_ref[:, None], qvel_ref[:, None]

qp_ref, qv_ref = _generate_placeholder_trajectory()

def _get_hl_obs(data: mjx.Data, info: Dict) -> jp.ndarray:
  """Get observations for the high-level policy."""
  ref_qpos = info['ref_qpos']
  ref_qvel = info['ref_qvel']
  return jp.concatenate([
    data.qpos,
    data.qvel,
    ref_qpos,  # Include reference state
    ref_qvel
  ])


def _get_ll_obs(data, info):
  return jp.concatenate([_get_ll_obs_base(data), info['desired_torque']])


def _get_ll_obs_base(data: mjx.Data) -> jp.ndarray:
  """Get base state observations for the low-level policy (before desired_torque)."""
  # Example: proprioceptive info relevant for muscle control
  return jp.concatenate([
    data.ctrl[:-1],
    data.actuator_length[:-1],
    data.actuator_velocity[:-1],
  ])

ppo_network = ppo_networks.make_ppo_networks(
      4,
      1,
       policy_hidden_layer_sizes=[128, 64, 32],
      preprocess_observations_fn=normalize,
      policy_obs_key='hl_obs')


ll_network = make_ll_network(
      6,
      19,
      hidden_layer_sizes=[64, 32, 16],
      preprocess_observations_fn=normalize,
      obs_key='ll_obs')

model_path = '../params/playground_params.pickle'
hl_params, ll_params = model.load_params(model_path)
del hl_params[0].mean['ll_obs']
del hl_params[0].std['ll_obs']
del hl_params[0].summed_variance['ll_obs']
del ll_params[0].mean['hl_obs']
del ll_params[0].std['hl_obs']
del ll_params[0].summed_variance['hl_obs']
def deterministic_hl_policy (input_data):
    logits = ppo_network.policy_network.apply(*hl_params[:2], input_data)
    brax_result = ppo_network.parametric_action_distribution.mode(logits)
    return brax_result

def deterministic_ll_policy (input_data):
    logits = ll_network.apply(*ll_params, input_data)
    return logits



def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system

    idx = int((data.time//0.002)*20 % qp_ref.shape[0])
    qp = qp_ref[idx]
    qv = qv_ref[idx]
    data.ctrl[-1] = qp[0]
    info = {'ref_qpos': qp, 'ref_qvel': qv, 'desired_torque': np.array([0])}
    hl_observations = {'hl_obs': np.array([_get_hl_obs(data, info)], dtype=np.float16)}
    hl_actions = deterministic_hl_policy(hl_observations)
    q_error = qp-data.qpos
    info['desired_torque'] = 4*q_error + 0.05 * (qv-data.qvel)
    ll_observations = {'ll_obs': np.array([_get_ll_obs(data, info)], dtype=np.float16)}
    #data.qfrc_applied=info['desired_torque']
    data.ctrl[:-1] = deterministic_ll_policy(ll_observations)

    pass


def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)
    model.opt.timestep = 0.002
    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)

    if model is not None:
        # Can set initial state

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(arm_control)

    return model, data

def passive():
    m = env.mj_model
    d = mujoco.MjData(m)
    jit_reset = jax.jit(env.reset)
    state = jit_reset(jax.random.key(0))

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step1(m, d)

            # Add your control logic here
            time.sleep(0.0001)

            mujoco.mj_step2(m, d)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                print(time_until_next_step)
                time.sleep(time_until_next_step)
    pass


if __name__ == '__main__':
    viewer.launch(loader=load_callback)