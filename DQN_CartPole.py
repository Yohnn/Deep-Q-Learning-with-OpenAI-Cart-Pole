# Second attempt at implementing DQN to the CartPole environment

# Rewriting the code to clean it up

import gym
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random

#create function to play random games. Used to inspect the information
#acquired from the cartpole environment such as observations,
#rewards, done, info

epsilon = 1.0
epsilon_f = 0.1
epsilon_decay = 0.995
batch_size = 64

def random_games(states,actions,rewards,next_states,dones):
    for episode in range(10): # One game is 1 episode. Play for 10 games
        state = env.reset()
        for _ in range(500): # Maximium 500 time steps if the game is not lost
            states.append(state) #get s_t
            action = env.action_space.sample()
            actions.append(action) #get a_t
            new_state, reward, done, info = env.step(action)
            dones.append(done) #get "if done"
            next_states.append(new_state) #get s_t+1
            rewards.append(reward) #get r_t+1
            state = new_state
            if done:
                break
        
#create a function that creates a model

def create_model(a_space):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(128, activation = 'relu'))
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(a_space, activation = 'linear'))

    model.compile(optimizer = tf.keras.optimizers.RMSprop(),
                  loss = tf.keras.losses.MeanSquaredError(),
                  metrics = tf.keras.metrics.Accuracy())
    return model


#create the epsilon-decay greedy policy which prioritizes exploration
#initially but slowly plrioritizes exploitation as the agent gets better

'''
def epsilon_greedy_policy(model,state,j,decay): #inputs a counter that will be the decay variable
    epsilon_i = 1.0
    epsilon_f = 0.1
    epsilon = epsilon_f + (epsilon_i - epsilon_f)*(decay**j)

    #generate random number
    p = np.random.random()

    if p < epsilon: #event with probablity epsilon (initially 100%) (exploration)
        return np.random.randint(2), epsilon #return random action (0 or 1)
    else: #exploitation
        return np.argmax(model.predict(state[np.newaxis,...])), epsilon
'''

# create a function that allows the model to play some games
# and store the <s_o,a_o,r_1,s_1> info from the game
# same structure as random_games function but instead uses the model
# to decide the next action

def practice_game(states,actions,rewards,new_states,dones,model,epsilon):
    while len(states) < batch_size: #play practice games until memory batch is filled
        state = env.reset()
        done = False
        while not done: # Maximium 500 time steps if the game is not lost
            states.append(state) #get s_t

            #Use epsilon-greedy policy for selecting the action
            p = np.random.random()
            if p < epsilon: #event with probablity epsilon (initially 100%) (exploration)
                action =  np.random.randint(2) #return random action (0 or 1)
            else: #exploitation
                action =  np.argmax(model.predict(state[np.newaxis,...]))
            
            actions.append(action) #get a_t
            new_state, reward, done, info = env.step(action)
            dones.append(done) #get "if done"
            new_states.append(new_state) #get s_t+1
            rewards.append(reward) #get r_t+1
            state = new_state
            if done:
                #rewards[-1] = -50 #if lose: punish
                break
    print("epsilon: ", epsilon)
    states,actions,rewards,new_states,dones = np.array(states),np.array(actions),np.array(rewards),np.array(new_states),np.array(dones)
    memory_batch = np.concatenate((states,actions[...,np.newaxis],rewards[...,np.newaxis],new_states,dones[...,np.newaxis]),axis = 1)

    #decay the epsilon
    if epsilon > epsilon_f:
        epsilon *= epsilon_decay
    return memory_batch, epsilon


# define function that recalls portions of the information from the practice
# games and learns from it. This is where the Bellman Optimality Equation is
# applied and the targets calculated


def experience_replay(memory_batch,model,train):
    gamma = 0.9 #decay factor gamma
    
    #shuffle the batch of memories
    memory_batch = np.array(random.sample(memory_batch.tolist(),len(memory_batch)//2))
    
    #extract the <s_o,a_o,r_1,s_1> from the memory batch into separate variables
    state = memory_batch[:,0:4]
    action = memory_batch[:,4]
    reward = memory_batch[:,5]
    new_state = memory_batch[:,6:-1]
    done = memory_batch[:,-1]

    targets = train.predict(state)
    new_targets = train.predict(new_state)

    for i in range(len(targets)):
        if done[i]:
            targets[i,np.argmax(targets[i])] = -100 #WHAT IF LOSING IS BAD
        else:
            targets[i,np.argmax(targets[i])] = reward[i] + gamma*np.max(new_targets[i])

    #after acquiring targets, train the model
    model.fit(state,targets,epochs = 3)
    return model

# play one game that evaluates total score achieved. used to check
# performance after every training episode

def evaluation_game(model):
    total_rewards = []
    for episode in range(10): # One game is 10 episode. Play for 10 episodes
        total_reward = 0
        state = env.reset()
        for _ in range(500): # Maximium 500 time steps if the game is not lost
            action = np.argmax(model.predict(state[np.newaxis,...]))
            new_state, reward, done, info = env.step(action)
            total_reward += reward
            state = new_state
            if done:
                total_rewards.append(total_reward)
                break
    return np.array(total_rewards).mean()



# Reinforcement Learning Program Proper

#create the Cart Pole v0 environment
env = gym.make("CartPole-v0")

a_space = env.action_space.n

main = create_model(a_space)
target = create_model(a_space)

#initally evaluate performance of model
print("Before training, score is: ", evaluation_game(main))

#Training proper

episodes = 1000 #train for 1000 episodes
decay = 0.995 #decay rate for epsilon 

j = 0 #counter for the epsilon decay
i = 0 #couhter for target update
for episode in range(episodes):
    #initate variables
    states,actions,rewards,new_states,dones = [],[],[],[],[]

    #play one practice game and acquire memory
    memory, epsilon = practice_game(states,actions,rewards,new_states,dones,main, epsilon)

    #learn from the memory and train the model
    main = experience_replay(memory,main,target)

    if i == 5:
        target.set_weights(main.get_weights())
        i = 0
        
    #evaluate thhe performance of the model by getting the total score possible

    total = evaluation_game(main)
    print("Average Total Score: {score} at Episode {episode}".format(score = total, episode = j+1))

    if total >= 195:
        main.save("cartpole_2.h5")
        print("Model achieved goal score of >= 195")
        print("Saving model. Exiting training")
        break
    j += 1
    i += 1






