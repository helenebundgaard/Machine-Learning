import gymnasium as gym
import numpy as np
import pandas as pd
import random

#Game: Cliff Walking
game = "CliffWalking-v0"

env = gym.make(game, render_mode="ansi")


# Parameters
gamma = 0.9
alpha = 0.4
training_Length = 1000
epsilon = 1.0

#Q-table
Q_table=np.zeros([env.observation_space.n,env.action_space.n])


# Training mode
observation, info = env.reset()

#Training loop
for i in range(training_Length):
    done = False
    #Episode loop
    observation, info = env.reset()
    while not done:
        # Taking action
        current_state = observation

        # Epsilon Greed
        action = int(np.argmax(Q_table[current_state,])) # Exploit

        if action == 0: 
            action = env.action_space.sample() # Explore

        #Enviroment Interaction
        observation,reward,terminated,truncated,info = env.step(action)
        next_state = observation

        # Temporal difference
        Td = reward + gamma *Q_table[next_state,np.argmax(Q_table[next_state,])] - Q_table[current_state,action]
        #Update Q-value
        Q_table[current_state,action] =  Q_table[current_state,action] + alpha * Td

        done = terminated or truncated 

#Data table
df = pd.DataFrame(Q_table, columns=['Up', 'Right', 'Down', 'Left'])
print(df)

env.close()

# Inference mode
observation, info = env.reset()
#Game: Cliff Walking
env = gym.make(game, render_mode="human")
#Inference loop
for i in range(15):
    #Episode loop
    done = False
    observation, info = env.reset()
    while not done:
        #Taking actions
        current_state = observation
        action = int(np.argmax(Q_table[current_state,]))
        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

env.close()

