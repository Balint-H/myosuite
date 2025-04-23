# Control of planar movements in an arm, using the example scene provided by MuJoCo. For more information on
# the simulator and the key feature of the physics simulation see:
# https://mujoco.readthedocs.io/en/stable/overview.html#introduction
import functools

import mujoco
import mujoco.viewer as viewer
import numpy as np
from numpy.linalg import pinv, inv
import glfw
from mujoco import mjx
import jax
from mujoco.mjx._src.support import muscle_gain, muscle_bias

Kp = 100
Kd = 10

xml = r'''
<mujoco model="2-link 6-muscle arm">
  <option timestep="0.005" iterations="50" solver="Newton" tolerance="1e-10"/>

  <visual>
    <rgba haze=".3 .3 .3 1"/>
  </visual>

  <default>
    <joint type="hinge" pos="0 0 0" axis="0 0 1" limited="true" range="0 120" damping="0.1"/>
    <muscle ctrllimited="true" ctrlrange="0 1"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.6 0.6" rgb2="0 0 0" width="512" height="512"/>

    <texture name="texplane" type="2d" builtin="checker" rgb1=".25 .25 .25" rgb2=".3 .3 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>

    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
  </asset>

  <worldbody>
    <geom name="floor" pos="0 0 -0.5" size="0 0 1" type="plane" material="matplane"/>

    <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>

    <site name="s0" pos="-0.15 0 0" size="0.02"/>
    <site name="x0" pos="0 -0.15 0" size="0.02" rgba="0 .7 0 1" group="1"/>

    <body pos="0 0 0">
      <geom name="upper arm" type="capsule" size="0.045" fromto="0 0 0  0.5 0 0" rgba=".5 .1 .1 1"/>
      <joint name="shoulder"/>
      <geom name="shoulder" type="cylinder" pos="0 0 0" size=".1 .05" rgba=".5 .1 .8 .5" mass="0" group="1"/>

      <site name="s1" pos="0.15 0.06 0" size="0.02"/>
      <site name="s2" pos="0.15 -0.06 0" size="0.02"/>
      <site name="s3" pos="0.4 0.06 0" size="0.02"/>
      <site name="s4" pos="0.4 -0.06 0" size="0.02"/>
      <site name="s5" pos="0.25 0.1 0" size="0.02"/>
      <site name="s6" pos="0.25 -0.1 0" size="0.02"/>
      <site name="x1" pos="0.5 -0.15 0" size="0.02" rgba="0 .7 0 1" group="1"/>

      <body pos="0.5 0 0">
        <geom name="forearm" type="capsule" size="0.035" fromto="0 0 0  0.5 0 0" rgba=".5 .1 .1 1"/>
        <joint name="elbow"/>
        <geom name="elbow" type="cylinder" pos="0 0 0" size=".08 .05" rgba=".5 .1 .8 .5" mass="0" group="1"/>

        <site name="s7" pos="0.11 0.05 0" size="0.02"/>
        <site name="s8" pos="0.11 -0.05 0" size="0.02"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <spatial name="SF" width="0.01">
      <site site="s0"/>
      <geom geom="shoulder"/>
      <site site="s1"/>
    </spatial>

    <spatial name="SE" width="0.01">
      <site site="s0"/>
      <geom geom="shoulder" sidesite="x0"/>
      <site site="s2"/>
    </spatial>

    <spatial name="EF" width="0.01">
      <site site="s3"/>
      <geom geom="elbow"/>
      <site site="s7"/>
    </spatial>

    <spatial name="EE" width="0.01">
      <site site="s4"/>
      <geom geom="elbow" sidesite="x1"/>
      <site site="s8"/>
    </spatial>

    <spatial name="BF" width="0.009" rgba=".4 .6 .4 1">
      <site site="s0"/>
      <geom geom="shoulder"/>
      <site site="s5"/>
      <geom geom="elbow"/>
      <site site="s7"/>
    </spatial>

    <spatial name="BE" width="0.009" rgba=".4 .6 .4 1">
      <site site="s0"/>
      <geom geom="shoulder" sidesite="x0"/>
      <site site="s6"/>
      <geom geom="elbow" sidesite="x1"/>
      <site site="s8"/>
    </spatial>
  </tendon>

  <actuator>
    <muscle name="SF" tendon="SF"/>
    <muscle name="SE" tendon="SE"/>
    <muscle name="EF" tendon="EF"/>
    <muscle name="EE" tendon="EE"/>
    <muscle name="BF" tendon="BF"/>
    <muscle name="BE" tendon="BE"/>
  </actuator>
</mujoco>
'''

actuators = ['SF', 'SE', 'EF', 'EE', 'BF', 'BE']


def clamp_vec(vec, clamp_range, limited, n) :
  for i in range(n):
    if limited[i]:
      vec[i] = mujoco.mju_clip(vec[i], clamp_range[2 * i], clamp_range[2 * i + 1])



def arm_control(model, data, jac_func):
    """
    :type model: mujoco.MjModel
    :type data: mujoco.MjData
    """
    # `model` contains static information about the modeled system, e.g. their indices in dynamics matrices
    # `data` contains the current dynamic state of the system

    # Muscles/cables should only pull
    data.ctrl[-1] = 1
    gains = np.zeros(model.nu)
    biases = np.zeros(model.nu)

    mjx_gains = np.zeros(model.nu)
    mjx_biases = np.zeros(model.nu)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    jac = jac_func(mjx_data.act, mjx_model, mjx_data)
    #mjx._src.forward.fwd_actuation(mjx_model, mjx_data)
    for i, ac in enumerate(actuators):
      actuator_data = data.actuator(ac)
      actuator_model = model.actuator(ac)
      gains[i] = mujoco.mju_muscleGain(actuator_data.length[0],
                                   actuator_data.velocity[0],
                                   actuator_model.lengthrange.reshape(2,1),
                                   actuator_model.acc0[0],
                                   actuator_model.gainprm[:-1].reshape(9,1),)

      biases[i] = mujoco.mju_muscleBias(actuator_data.length[0],
                                        actuator_model.lengthrange.reshape(2,1),
                                        actuator_model.acc0[0],
                                        actuator_model.biasprm[:-1].reshape(9,1),)
      mjx_gains[i] = muscle_gain(actuator_data.length[0],
                              actuator_data.velocity[0],
                              actuator_model.lengthrange.reshape(2, 1),
                              actuator_model.acc0[0],
                              actuator_model.gainprm[:-1].reshape(9, 1), )[0]
      mjx_biases[i] = muscle_bias(actuator_data.length[0],
                                  actuator_model.lengthrange.reshape(2,1),
                                  actuator_model.acc0[0],
                                  actuator_model.biasprm[:-1].reshape(9,1),)[0]


    forces = gains * data.act + biases

    moment_matrix = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(moment_matrix,
                            data.actuator_moment,
                            data.moment_rownnz,
                            data.moment_rowadr,
                            data.moment_colind)
    qfrc = np.zeros((model.nv, 1))
    mujoco.mju_mulMatTVec(qfrc, moment_matrix, forces.reshape(model.nu,1))
    true_qfrc = data.qfrc_actuator


    pass

def load_callback(model=None, data=None):
    # Clear the control callback before loading a new model
    # or a Python exception is raised
    mujoco.set_mjcb_control(None)

    # `model` contains static information about the modeled system
    model = mujoco.MjModel.from_xml_string(xml=xml, assets=None)

    # `data` contains the current dynamic state of the system
    data = mujoco.MjData(model)

    def muscle_step(act, mjx_model, mjx_data):
      mjx_data = mjx_data.replace(act=act)
      mjx_data = mjx.step(mjx_model, mjx_data)

      return mjx_data.qfrc_actuator

    jac_func = jax.jacrev(muscle_step)

    if model is not None:
        # Can set initial state
        data.joint('shoulder').qpos = 0
        data.joint('elbow').qpos =0

        # The provided "callback" function will be called once per physics time step.
        # (After forward kinematics, before forward dynamics and integration)
        # see https://mujoco.readthedocs.io/en/stable/programming.html#simulation-loop for more info
        mujoco.set_mjcb_control(functools.partial(arm_control, jac_func=jac_func))

    return model, data


if __name__ == '__main__':
    viewer.launch(loader=load_callback)

