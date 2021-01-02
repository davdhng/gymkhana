import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(512, 10)
        self.hidden2 = nn.Linear(256, 10)
        self.hidden3 = nn.Linear(64, 10)
        self.hidden4 = nn.Linear(2)
    def forward(self, x):
        x = self.hidden2(self.hidden1(x))
        x = self.hidden2(F.relu(self.hidden3(x)))

env = gym.make('CartPole-v0')
highscore = 0
# q_table = np.zeros([env.observation_space.n, env.action_space.n])

for i_episode in range(20):
    observation = env.reset()
    points = 0
    while True:
        env.render()
        # print(observation)
        # action = env.action_space.sample()
        action = 1 if observation[2] > 0 else 0
        observation, reward, done, info = env.step(action)
        points += reward
        if done:
            if points > highscore:
                highscore = points
                print("New high score: " + str(highscore))
            break
env.close()
