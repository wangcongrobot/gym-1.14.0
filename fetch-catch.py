import gym
import matplotlib.pyplot as plt

mode = 'rgb_array'

env = gym.make("FetchCatch-v1")
#env = gym.wrappers.Monitor(env, './video', force=True)
#env.render(mode)
#plt.imshow(env.render(mode='rgb_array'))
#plt.show()
for i in range(10):
  env.reset()
 # env.render(mode)
  for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
#    print(obs)
    print(reward)
    print(done)
    print(info)
  #  env.render(mode)

