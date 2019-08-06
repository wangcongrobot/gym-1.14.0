import gym
import matplotlib.pyplot as plt

mode = 'human'
#mode = 'rgb_array'

env = gym.make("HandManipulateEggRotateTouchSensors-v0")
#env = gym.make('FetchSlide-v1')
#env = gym.wrappers.Monitor(env, './video', force=True)
#env.render('human')
#plt.imshow(env.render(mode='rgb_array', camera_id=-1))
#plt.show()
for i in range(20):
  env.reset()
  env.render('human')
  for i in range(200):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs)
    print(reward)
    print(done)
    print(info)
    env.render('human')

