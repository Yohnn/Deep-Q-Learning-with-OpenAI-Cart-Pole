'''
Implementation of a Deep Q Learning Agent for the Cart Pole v0 
environment of OpenAI.

We use the Double DQN for better stability compared to the 
normal DQN. The Double DQN using two networks, the main network 
and the target network. The target network is used as the reference 
for the training of the main network. This is done for short regular
intervals (in this case, every 5 episodes) then the target network weights 
are updated based on the main network weights.
'''

#import the necessary libraries
import gym
import numpy as np
import tensorflow as tf
import random

#define the necessary functions
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

'''
create a function that allows the model to play some games
and store the <s_o,a_o,r_1,s_1> info from the game
same structure as random_games function but instead uses the model
to decide the next action
'''

def practice_game(states,actions,rewards,new_states,dones,model,epsilon,memory):
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
            memory.append((state,action,reward,new_state,done))
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

'''
 define function that recalls portions of the information from the practice
 games and learns from it. This is where the Bellman Optimality Equation is
 applied and the targets calculated
'''

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
    model.fit(state,targets,epochs = 1)
    return model


'''
play 10 games that evaluates the average total score achieved.
used for checking the performance of the model after every episode
'''

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

#Defining the global variables

'''
We want the agent to first prioritize exploration then slowly
prioritize exploitation, thus the initial epsilon, final epsilon values,
and decay rate are defined below
'''
epsilon = 1.0 #initial epsilon value
epsilon_f = 0.1 #final epsilon value
epsilon_decay = 0.995 #decay rate of epsilon


batch_size = 64 #the size of the training data
memory = [] #for storing the entire experience of the agent

#create the Cart Pole v0 environment
env = gym.make("CartPole-v0")

a_space = env.action_space.n #possible actions of the agent based on the environment

#create the main and target networks
main = create_model(a_space)
target = create_model(a_space)

#initally evaluate performance of model
print("Before training, score is: ", evaluation_game(main))

#Training proper

episodes = 1000 #train for 1000 episodes
i = 0 #counter for target update
update_target_interval = 5 #we update the target network based on the main network weights every 5 episodes

for episode in range(episodes):
    #initate variables
    states,actions,rewards,new_states,dones = [],[],[],[],[]

    #play one practice game and acquire memory
    memory_batch, epsilon = practice_game(states,actions,rewards,new_states,dones,main, epsilon, memory)

    #learn from the memory and train the model
    main = experience_replay(memory_batch,main,target)

    if i == update_target_interval:
        target.set_weights(main.get_weights())
        i = 0
        
    #evaluate thhe performance of the model by getting the total score possible

    total = evaluation_game(main)
    print("Average Total Score: {score} at Episode {episode}".format(score = total, episode = episode+1))

    '''
    We use a standard score of 195 to determine if the agent has
    "solved" the environment. If the evaluation shows a score >= standard,
    then we stop the training and save the model.
    '''

    if total >= 195:
        main.save("cartpole_2.h5")
        print("Model achieved goal score of >= 195")
        print("Saving model. Exiting training")
        break

    i += 1






