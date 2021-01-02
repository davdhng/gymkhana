import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')
rewards = []
discount_factor = 0.8
lr = 0.9
report_interval = 500
report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Average: %.2f ' \
         '(Episode %d)'

def print_report(rewards, episode):
    print(report % (
        np.mean(rewards[-100:]),
        max([np.mean(rewards[i:i+100]) for i in range(len(rewards) - 100)]),
        np.mean(rewards),
        episode))

q = np.zeros((env.observation_space.n, env.action_space.n))
for episode in range(1, 4001):
    state = env.reset()
    episode_reward = 0
    while True:
        noise = np.random.random((1, env.action_space.n)) / (episode**2.)
        action = np.argmax(q[state, :] + noise) 
        state2, reward, done, _ = env.step(action)
        qtarget = reward + discount_factor * np.max(q[state2, :])
        q[state, action] =(1 - lr) * q[state, action] + lr * qtarget
        episode_reward += reward
        state = state2
        if done:
            rewards.append(episode_reward)
            if episode % report_interval == 0:
                print_report(rewards, episode)
            break
print_report(rewards, -1)
