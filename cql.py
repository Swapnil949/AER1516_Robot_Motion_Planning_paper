# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 20:15:51 2023

@author: Alejandro
"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

"""

"8x8": [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
"""



env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)

def soft_policy(eps, S, A, Q):
    # Initialize soft policy and set all values to eps/A
    b = np.full((S,A), eps/A)
    pi = np.argmax(Q, axis=1) #used in next row to make epsilon greedy policy based on Q
    #Make epison greedy policy based on Q
    b[np.arange(S), pi] = 1- eps + b[np.arange(S), pi]
    return b #SbyA array



#off-policy Q learning
#Define variables
rng=np.random.RandomState(0)
gamma = 0.999
A = env.action_space.n
S = 8*8 #Based on the get_state function which discretizes the states
alpha = 0.9
eps = 0.9999
eps_decay = 0.999
eps_final = 0.1
#Initialize
Q = rng.random([S, A])
#Loop for each episode
for _ in range(10000):
    #Initialize state
    obs = env.reset()
    
    state = obs[0] #Get discretized state
    #Loop for each step of episode
    while True:
        #Choose A from S using policy derived from Q (eg. e-greedy)
        e_greedy = soft_policy(eps, S, A, Q)
        action = rng.choice(A, p=e_greedy[state])
        #if eps > eps_final:
         #   eps *= eps_decay
        
        #Take action A, observe R, S'
        obs, reward, done, info, prob = env.step(action)
        if done:
            reward = -1
        if obs == 63:
            reward = 1

        state_new = obs
        #Apply Q-learning equation:
        Q[state, action] = Q[state, action] + alpha*(reward + gamma*np.max(Q[state_new]) - Q[state, action])
        state = state_new
        if done:
            break

print(Q)  
#%%

env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode='human')
env = gym.wrappers.RecordVideo(env, 'video')
#Test out Q-Learning policy
pi = np.argmax(Q, axis = 1)
obs = env.reset()

i = 0
Complete = False
while not Complete:
    env.render()
    if i == 0:
        state = obs[0]
    else:
        state = obs
    action = pi[state]
    obs, reward, done, info, prob = env.step(action)
    if done:
        Complete = True
        print(obs)
    i = i + 1
env.close()
print('episode lasted', i, 'steps')
mapping = {0:"\u2190", 1:"\u2193", 2:"\u2192", 3:"\u2191"}
print(np.array([mapping[p] for p in pi]).reshape(8, 8))

# %%
