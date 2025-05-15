import timeit
from etils import epath
import jax
jax.config.update('jax_platform_name', 'cpu')
import mujoco
from mujoco import mjx

path = (epath.Path(epath.resource_path('mujoco')) / (
        'mjx/test_data/actuator/arm26.xml')).as_posix()
path = r'../../../simhive/myo_sim/arm/myoarm.xml'

model = mujoco.MjModel.from_xml_path(path, assets=None)
model.opt.iterations = 2
model.opt.ls_iterations = 1

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
      mjx._src.support.muscle_gain,
      'uuuuu',
      'u',
      mjx_data.actuator_length,
      mjx_data.actuator_velocity,
      jax.numpy.array(mjx_model.actuator_lengthrange),
      jax.numpy.array(mjx_model.actuator_acc0),
      mjx_model.actuator_gainprm,
      group_by='u',
    )
    return gains[None, :] * mjx_data.actuator_moment.T

jac_analytical_func = jax.jit(analytical_jac)

key = jax.random.key(1)
act = jax.random.uniform(key, (63,))

jac1 = jac_with_jax_func(jax.random.uniform(key, (63,)), mjx_model, mjx_data )
jac2 = jac_analytical_func( mjx_model, mjx_data )

print(jac1)
print(jac2)
print(jax.numpy.sum(jac1-jac2))


def jac_with_jax(mjx_model=mjx_model, mjx_data=mjx_data, jac_func=jac_with_jax_func, key=key, act=act):
    cur_jac = jac_func(act, mjx_model, mjx_data)
    return mjx_data, mjx_model, cur_jac

def jac_analytical(mjx_model=mjx_model, mjx_data=mjx_data, jac_func=jac_analytical_func, key=key):
    cur_jac = jac_func(mjx_model, mjx_data)
    return mjx_data, mjx_model, cur_jac

print(timeit.timeit("f()",
                    setup="from __main__ import jac_with_jax;"
                          "import jax;"
                          "f=jax.jit(jac_with_jax);"
                          "f()",
                    number=3000, globals={'key':key}))
print(timeit.timeit("f()",
                    setup="from __main__ import jac_analytical;"
                          "import jax;"
                          "f=jax.jit(jac_analytical);"
                          "f()",
                    number=3000,
                    globals={'key':key}))

# mjx_data = mjx.fwd_actuation(mjx_model, mjx_data)
# qfrc = mjx_data.qfrc_actuator
#
# @jax.jit
# def pinv_solve(jac=jac1, qfrc=qfrc):
#   return jax.numpy.linalg.pinv(jac)@qfrc
#
# @jax.jit
# def lststq_solve(jac=jac1, qfrc=qfrc):
#   return jax.numpy.linalg.lstsq(jac, qfrc)
#
# print(timeit.timeit("pinv_solve()", setup="from __main__ import pinv_solve; pinv_solve()", number=10000, globals={'key':key}))
# print(timeit.timeit("lststq_solve()", setup="from __main__ import lststq_solve; lststq_solve()", number=10000, globals={'key':key}))