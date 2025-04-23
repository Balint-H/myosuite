import timeit
from etils import epath
import jax
import mujoco
from mujoco import mjx

path = (epath.Path(epath.resource_path('mujoco')) / (
        'mjx/test_data/actuator/arm26.xml')).as_posix()

def load_with_put(mjcf_file=path):
    mj_model = mujoco.MjModel.from_xml_path(mjcf_file)
    mjx_model = mjx.put_model(mj_model)
    mj_data = mujoco.MjData(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)
    return mjx_data, mjx_model

def load_with_make(mjcf_file=path):
    mj_model = mujoco.MjModel.from_xml_path(mjcf_file)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.make_data(mjx_model)
    return mjx_data, mjx_model

print(timeit.timeit("load_with_put()", setup="from __main__ import load_with_put, load_with_make", number=100))
print(timeit.timeit("load_with_make()", setup="from __main__ import load_with_put, load_with_make", number=100))