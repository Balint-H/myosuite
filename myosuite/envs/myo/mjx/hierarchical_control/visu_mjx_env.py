import time
import mujoco
import jax
import mujoco.viewer
from playground_hand_hierarchical import MjxHand

# Visualize an MJX environment interactively



def main():
    env = MjxHand()

    m, d = get_mj_model_data(env)

    # Note: the first two steps will be performed much slower as the function is being jitted.
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.PRNGKey(0))

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            state = state.replace(data=state.data.replace(mocap_pos=d.mocap_pos, xfrc_applied=d.xfrc_applied))
            state = jit_step(state, d.ctrl)

            d.qpos = state.data.qpos
            mujoco.mj_forward(m, d)  # We only do forward step, no need to integrate.

            # We'll use the sensordata array to visualize key values as real-time bar-graphs (use F4 in viewer)
            d.sensordata[0] = state.reward

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
    pass


def get_mj_model_data(env):
    # We could get the model from the env, but we want to make some edits for convenience
    spec = mujoco.MjSpec.from_file(env.xml_path)
    spec = env.preprocess_spec(spec)
    # Add in dummy sensor we can write to later to visualize values
    spec.add_sensor(name="reward+target", type=mujoco.mjtSensor.mjSENS_USER, dim=1 + env.mjx_model.nv)


    m = spec.compile()
    d = mujoco.MjData(m)
    return m, d


if __name__ == '__main__':
    main()
