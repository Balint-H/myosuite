# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction

import mujoco
import mujoco.viewer as viewer
import numpy as np
from brax.training.acme.running_statistics import normalize
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from functools import partial


xml = '../assets/elbow/myoelbow_1dof6muscles_mjx_eval.xml'

ppo_network = ppo_networks.make_ppo_networks(
      4,
      6,
      preprocess_observations_fn=normalize)
model_path = '../params/elbow_params.pickle'
params = model.load_params(model_path)
def deterministic_policy (input_data):

    logits = ppo_network.policy_network.apply(*params[:2], np.array([input_data], dtype=np.float32))
    brax_result = ppo_network.parametric_action_distribution.mode(logits)
    return brax_result

def get_obs(data, target):
    """Observes elbow angle, velocities, and last applied torque."""
    position = data.qpos

    # external_contact_forces are excluded
    return np.concatenate([
        position,
        data.qvel,
        data.qfrc_actuator,
        target
    ])


def arm_control(model, data):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system
    observations = get_obs(data, [data.ctrl[-1]])
    data.ctrl[:-1] = deterministic_policy(observations)
    pass


def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_path(filename=xml, assets=None)

    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)

    if model is not None:
        # Can set initial state

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(arm_control)

    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)