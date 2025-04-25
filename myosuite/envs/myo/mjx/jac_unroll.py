import mujoco
from etils import epath
from mujoco import mjx
import jax
from jax import numpy as jp

model_path = (epath.Path(epath.resource_path('mujoco')) / (
        'mjx/test_data/actuator/arm26.xml')).as_posix()

if __name__ == "__main__":

    model = mujoco.MjModel.from_xml_path(model_path)
    model.actuator_actearly[:] = 1
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(mjx_model)

    def muscle_step(ctrl, mjx_data, mjx_model):

        mjx_data = mjx_data.replace(ctrl=ctrl)
        mjx_data = mjx.step(mjx_model, mjx_data)

        return mjx_data.qfrc_actuator


    get_jacobian_muscles_early = jax.jit(jax.jacrev(muscle_step))
    jacobian_muscles_early = get_jacobian_muscles_early(jp.ones((mjx_model.nu,)), mjx_data, mjx_model)

    print(jacobian_muscles_early)
    assert not jp.any(jp.isnan(jacobian_muscles_early)), f"NaNs detected when differentiating through muscles"

