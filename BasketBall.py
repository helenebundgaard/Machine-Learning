import numpy as np


# Setting conversion rates (how likely you are to hit the hoop) and the number of shots and hoops
conversion_rates = [0.5, 0.33, 0.1]
N = 10000
d = len(conversion_rates)

# Define reward values for each hoop based on difficulty level
hoop_rewards = {
    "easy": 1,
    "medium": 2,
    "hard": 3
}

# Succes rate taking conversion rates and points into account
succes_rates = [0.5*1, 0.33*2, 0.1*3]

# Creating the dataset
X = np.zeros((N, d))
for i in range(N):
    for j in range(d):
        if np.random.rand() < succes_rates[j]:
            X[i][j] = 1

# Making arrays to count our hits and misses
nPosReward = np.zeros(d)
nNegReward = np.zeros(d)

# Thompson Sampling: beta distibution and updating its hits and misses
for i in range(N):
    selected = 0
    maxRandom = 0
    for j in range(d):
        randomBeta = np.random.beta(nPosReward[j] + 1, nNegReward[j] + 1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            selected = j
    if X[i][selected] == 1:
        nPosReward[selected] += 1
    else:
        nNegReward[selected] += 1

# Printing scored and best hoop
nSelected = nPosReward + nNegReward 
for i in range(d):
    print('Hoop number ' + str(i + 1) + ' got ' + str(nSelected[i]) + ' points')
print('Conclusion: Best hoop is hoop number ' + str(np.argmax(nSelected) + 1))
