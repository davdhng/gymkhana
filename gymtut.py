import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('CartPole-v0')
highscore = 0
# q_table = np.zeros([env.observation_space.n, env.action_space.n])
n_actions = 2
n_obs = 4
q = np.zeros((n_obs, n_actions))
lr = 0.9
discount_factor = 0.8

for episode in range(1, 4001):
    state = env.reset()
    points = 0
    while True:
        env.render()
        noise = np.random.random((1, n_actions)) / (episode **2.)
        action = np.argmax(q[state, :] + noise)
        newstate, reward, done, _ = env.step(action)
        qtarget = reward + discount_factor * np.max(q[newstate, :])
        q[observation, action] = (1 - lr) * q[state, action] + lr * qtarget
        points += reward
        state = newstate
        if done:
            break
env.close()
