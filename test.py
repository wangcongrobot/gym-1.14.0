import gym
import argparse
from matplotlib import pylab
from pylab import *

ion()

parser = argparse.ArgumentParser()
parser.add_argument('env_id')
args = parser.parse_args()

env = gym.make(args.env_id)
while True:
    env.reset()
    done = False
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        imshow(env.render(mode='rgb_array'))
        pause(1/30)
