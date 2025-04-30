import time
import mujoco
from mujoco import mjx
import mujoco.viewer
from mujoco.mjx._src.io import _strip_weak_type, _make_option, _make_statistic
from mujoco_playground._src import mjx_env
from myosuite.envs.myo.mjx.envs.playground_arm import PlaygroundArm
from myosuite.envs.myo.mjx.envs.playground_elbow import PlaygroundElbow
import copy
from typing import List, Tuple, Union

import jax
from jax import numpy as jp
import mujoco
from mujoco.mjx._src import collision_driver
from mujoco.mjx._src import constraint
from mujoco.mjx._src import mesh
from mujoco.mjx._src import support
from mujoco.mjx._src import types
import numpy as np
import scipy


def main():
    xml_path = rf"../assets/arm/myoarm_relocate_mjx.xml"
    env = PlaygroundArm()
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

def put_model(
    m: mujoco.MjModel, device=None, _full_compat: bool = False  # pylint: disable=invalid-name
) -> types.Model:
  """Puts mujoco.MjModel onto a device, resulting in mjx.Model.

  Args:
    m: the model to put onto device
    device: which device to use - if unspecified picks the default device
    _full_compat: put all MjModel fields onto device irrespective of MJX support
      This is an experimental feature.  Avoid using it for now.

  Returns:
    an mjx.Model placed on device
  """

  mesh_geomid = set()
  for g1, g2, ip in collision_driver.geom_pairs(m):
    t1, t2 = m.geom_type[[g1, g2]]
    # check collision function exists for type pair
    if not collision_driver.has_collision_fn(t1, t2) and not _full_compat:
      t1, t2 = mujoco.mjtGeom(t1), mujoco.mjtGeom(t2)
      raise NotImplementedError(f'({t1}, {t2}) collisions not implemented.')
    # margin/gap not supported for meshes and height fields
    no_margin = {mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_HFIELD}
    if no_margin.intersection({t1, t2}):
      if ip != -1:
        margin = m.pair_margin[ip]
      else:
        margin = m.geom_margin[g1] + m.geom_margin[g2]
      if margin.any() and not _full_compat:
        t1, t2 = mujoco.mjtGeom(t1), mujoco.mjtGeom(t2)
        raise NotImplementedError(f'({t1}, {t2}) margin/gap not implemented.')
    for t, g in [(t1, g1), (t2, g2)]:
      if t == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_geomid.add(g)

  # check for unsupported sensor and equality constraint combinations
  sensor_rne_postconstraint = (
      np.any(m.sensor_type == types.SensorType.ACCELEROMETER)
      | np.any(m.sensor_type == types.SensorType.FORCE)
      | np.any(m.sensor_type == types.SensorType.TORQUE)
  )
  eq_connect_weld = np.any(m.eq_type == types.EqType.CONNECT) | np.any(
      m.eq_type == types.EqType.WELD
  )
  if sensor_rne_postconstraint and eq_connect_weld:
    raise NotImplementedError(
        'rne_postconstraint not implemented with equality constraints:'
        ' connect, weld.'
    )

  for enum_field, enum_type, mj_type in (
      (m.actuator_biastype, types.BiasType, mujoco.mjtBias),
      (m.actuator_dyntype, types.DynType, mujoco.mjtDyn),
      (m.actuator_gaintype, types.GainType, mujoco.mjtGain),
      (m.actuator_trntype, types.TrnType, mujoco.mjtTrn),
      (m.eq_type, types.EqType, mujoco.mjtEq),
      (m.sensor_type, types.SensorType, mujoco.mjtSensor),
      (m.wrap_type, types.WrapType, mujoco.mjtWrap),
  ):
    missing = set(enum_field) - set(enum_type)
    if missing and not _full_compat:
      raise NotImplementedError(
          f'{[mj_type(m) for m in missing]} not supported'
      )

  mj_field_names = {
      f.name
      for f in types.Model.fields()
      if f.metadata.get('restricted_to') != 'mjx'
  }
  fields = {f: getattr(m, f) for f in mj_field_names}

  # zero out fields restricted to MuJoCo
  if not _full_compat:
    for f in types.Model.fields():
      if f.metadata.get('restricted_to') == 'mujoco' and isinstance(
          fields[f.name], np.ndarray
      ):
        fields[f.name] = np.zeros((0,), dtype=fields[f.name].dtype)

  fields['dof_hasfrictionloss'] = fields['dof_frictionloss'] > 0
  fields['tendon_hasfrictionloss'] = fields['tendon_frictionloss'] > 0
  fields['geom_rbound_hfield'] = fields['geom_rbound']
  fields['cam_mat0'] = fields['cam_mat0'].reshape((-1, 3, 3))
  fields['opt'] = _make_option(m.opt, _full_compat=_full_compat)
  fields['stat'] = _make_statistic(m.stat)

  # spatial tendon wrap inside
  fields['wrap_inside_maxiter'] = 5
  fields['wrap_inside_tolerance'] = 1.0e-4
  fields['wrap_inside_z_init'] = 1.0 - 1.0e-5
  fields['is_wrap_inside'] = np.zeros(0, dtype=bool)
  if m.nsite:
    # find sphere or cylinder geoms (if any exist)
    (wrap_id_geom,) = np.nonzero(
        (m.wrap_type == mujoco.mjtWrap.mjWRAP_SPHERE)
        | (m.wrap_type == mujoco.mjtWrap.mjWRAP_CYLINDER)
    )
    wrap_objid_geom = m.wrap_objid[wrap_id_geom]
    geom_pos = m.geom_pos[wrap_objid_geom]
    geom_size = m.geom_size[wrap_objid_geom, 0]

    # find sidesites (if any exist)
    side_id = np.round(m.wrap_prm[wrap_id_geom]).astype(int)
    side = m.site_pos[side_id]

    # wrap inside flag
    fields['is_wrap_inside'] = np.array(
        (np.linalg.norm(side - geom_pos, axis=1) < geom_size) & (side_id >= 0)
    )

  # Pre-compile meshes for MJX collisions.
  fields['mesh_convex'] = [None] * m.nmesh
  if not _full_compat:
    for i in mesh_geomid:
      dataid = m.geom_dataid[i]
      if fields['mesh_convex'][dataid] is None:
        fields['mesh_convex'][dataid] = mesh.convex(m, dataid)  # pytype: disable=unsupported-operands
    fields['mesh_convex'] = tuple(fields['mesh_convex'])

  model = types.Model(**{k: copy.copy(v) for k, v in fields.items()})

  model = jax.device_put(model, device=device)
  return _strip_weak_type(model)


if __name__ == '__main__':
    main()