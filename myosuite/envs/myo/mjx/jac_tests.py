import timeit
from etils import epath
import jax
jax.config.update('jax_platform_name', 'cpu')
import mujoco
from mujoco import mjx

path = (epath.Path(epath.resource_path('mujoco')) / (
        'mjx/test_data/actuator/arm26.xml')).as_posix()

model = mujoco.MjModel.from_xml_path(path, assets=None)

# `data` contains the current dynamic state of the system
data = mujoco.MjData(model)



mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

mjx_data = mjx.step(mjx_model, mjx_data)
mjx_data = mjx.fwd_position(mjx_model, mjx_data)
mjx_data = mjx.fwd_velocity(mjx_model, mjx_data)

def muscle_step(act, mjx_model, mjx_data):
    mjx_data = mjx_data.replace(act=act)
    mjx_data = mjx.fwd_actuation(mjx_model, mjx_data)

    return mjx_data.qfrc_actuator

jac_with_jax_func = jax.jit(jax.jacrev(muscle_step))

@jax.jit
def get_gain(*args):
    gain_p, len_, vel, len_range, acc0 = args
    gain = mjx._src.support.muscle_gain(len_, vel, len_range, acc0, gain_p)
    return gain


def analytical_jac(mjx_model, mjx_data: mjx.Data):
    gains = mjx._src.scan.flat(
      mjx_model,
      get_gain,
      'uuuuu',
      'u',
      mjx_model.actuator_gainprm,
      mjx_data.actuator_length,
      mjx_data.actuator_velocity,
      jax.numpy.array(mjx_model.actuator_lengthrange),
      jax.numpy.array(mjx_model.actuator_acc0),
      group_by='u',
    )
    return gains[None, :] * mjx_data.actuator_moment.T

jac_analytical_func = jax.jit(analytical_jac)

key = jax.random.key(1)
act = jax.random.uniform(key, (6,))

jac1 = jac_with_jax_func(jax.random.uniform(key, (6,)), mjx_model, mjx_data )
jac2 = jac_analytical_func( mjx_model, mjx_data )

print(jac1)
print(jac2)


def jac_with_jax(mjx_model=mjx_model, mjx_data=mjx_data, jac_func=jac_with_jax_func, key=key, act=act):
    cur_jac = jac_func(act, mjx_model, mjx_data)
    return mjx_data, mjx_model, cur_jac

def jac_analytical(mjx_model=mjx_model, mjx_data=mjx_data, jac_func=jac_analytical_func, key=key):
    cur_jac = jac_func(mjx_model, mjx_data)
    return mjx_data, mjx_model, cur_jac

print(timeit.timeit("jac_with_jax()", setup="from __main__ import jac_with_jax", number=50000, globals={'key':key}))
print(timeit.timeit("jac_analytical()", setup="from __main__ import jac_analytical", number=50000, globals={'key':key}))