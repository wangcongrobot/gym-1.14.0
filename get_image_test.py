import numpy as np
import matplotlib.pyplot as plt
import mujoco_py
import gym

def print_img(camera_name=None):
    img = env.env.sim.render(width=500, height=500, camera_name=camera_name, depth=False)
    print(img)
    plt.imshow(img)
    plt.show()

mode = 'human' 
# mode = 'rgb_array

env = gym.make("FetchCatch-v1")
env.reset()
env.render()
env.env.viewer == None
# For the confirm that 'env.env.viewer' is 'None'

env.env.viewer = mujoco_py.MjViewer(env.env.sim)
# Without the step above, there will be a "GLEW initialization error". Issue has been submitted.
env.env.sim.render(width=500, height=500, mode='window')
env.env.reset()

env.env._get_viewer(mode).render() # mode = 'human'
#env.env._get_viewer(mode).render(width=48, height=48) # mode = 'rgb_array'

img = env.env.sim.render(width=500, height=500, camera_name='top', depth=False)
print(img)
plt.imshow(img)
plt.show()

print_img('top')
print_img('fixed')
print_img('top_w')


# The image showed here is normal
env.env._get_viewer(mode).render() # mode = 'human'
#env.env._get_viewer(mode).render(width=48, height=48) # mode = 'rgb_array'

# Then start to set the state from given mujoco interface
nq = env.env.sim.model.nq
nv = env.env.sim.model.nv

qpos = gym.spaces.Box(high=np.ones(nq), low=-np.ones(nq), dtype=np.float32).sample()
qvel = gym.spaces.Box(high=np.ones(nv), low=-np.ones(nv), dtype=np.float32).sample()

old_state = env.env.sim.get_state()
new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
env.env.sim.set_state(new_state)

env.env.sim.forward()
# finish set_state

#env.env.sim.render_contexts[0]._set_mujoco_buffers()

print_img('top')
print_img('fixed')
print_img('top_w')

img = env.env.sim.render(width=500, height=500, camera_name='top', depth=False)
plt.imshow(img)
plt.show()
# Then the image showing has only one color


def render(self, width=None, height=None, *, camera_name=None, depth=False,
            mode='offscreen', device_id=-1):
    """
    Renders view from a camera and returns image as an `numpy.ndarray`.

    Args:
    - width (int): desired image width.
    - height (int): desired image height.
    - camera_name (str): name of camera in model. If None, the free
        camera will be used.
    - depth (bool): if True, also return depth buffer
    - device (int): device to use for rendering (only for GPU-backed
        rendering).

    Returns:
    - rgb (uint8 array): image buffer from camera
    - depth (float array): depth buffer from camera (only returned
        if depth=True)
    """
    if camera_name is None:
        camera_id = None
    else:
        camera_id = self.model.camera_name2id(camera_name)

    if mode == 'offscreen':
        with _MjSim_render_lock:
            if self._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(
                    self, device_id=device_id)
            else:
                render_context = self._render_context_offscreen

            render_context.render(
                width=width, height=height, camera_id=camera_id)
            return render_context.read_pixels(
                width, height, depth=depth)
    elif mode == 'window':
        if self._render_context_window is None:
            from mujoco_py.mjviewer import MjViewer
            render_context = MjViewer(self)
        else:
            render_context = self._render_context_window

        render_context.render()

    else:
        raise ValueError("Mode must be either 'window' or 'offscreen'.")