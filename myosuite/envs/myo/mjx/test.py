import timeit
import mujoco
from mujoco import mjx
import jax


def load_mujoco(mjcf_file=rf"../../../simhive/myo_sim/leg/myolegs_abdomen.xml"):
    mj_model = mujoco.MjModel.from_xml_path(mjcf_file)
    mjx_model = mjx.put_model(mj_model)
    mj_data = mujoco.MjData(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)
    return mjx_data, mjx_model


def load_mjx(mjcf_file=rf"../../../simhive/myo_sim/leg/myolegs_abdomen.xml"):
    mj_model = mujoco.MjModel.from_xml_path(mjcf_file)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)
    return mjx_data, mjx_model


with jax.default_device(jax.devices("cpu")[0]):
    print(timeit.timeit("load_mujoco()", setup="from __main__ import load_mujoco, load_mjx", number=500))
    print(timeit.timeit("load_mjx()", setup="from __main__ import load_mujoco, load_mjx", number=500))
