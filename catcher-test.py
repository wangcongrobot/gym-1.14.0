import gym
from gym.envs.registration import register

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

# add Catcher3d environment
'''
register(
    id='Catcher3d{}-v1'.format(suffix),
    entry_point='gym.envs.robotics:Catcher3dEnv',
    max_episode_steps=250,
    kwargs={**kwargs, **dict(
        add_high_res_output=False,
        no_movement=False,
        stack_frames=False,
        camera_3=False
    )}
)
'''

env = gym.make("Catcher3d-v1")

for i in range(10):
  env.reset()
  env.render()
  for i in range(200):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs)
    env.render()

