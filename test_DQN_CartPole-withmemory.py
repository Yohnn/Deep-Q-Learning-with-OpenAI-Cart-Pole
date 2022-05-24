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
from matplotlib import pyplot as plt
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

def practice_game(model,epsilon,memory):
    memory_initial_size = len(memory)
    while int(len(memory) - memory_initial_size) < batch_size: #play practice games until ideal batch_size is met
        state = env.reset() #initiate cartpole environment
        done = False
        while not done: #finish one game
            #Use epsilon-greedy policy for selecting the action
            p = np.random.random()
            if p < epsilon: #event with probablity epsilon (initially 100%) (exploration)
                action =  np.random.randint(2) #return random action (0 or 1)
            else: #exploitation
                action =  np.argmax(model.predict(state[np.newaxis,...]))
            
            
            new_state, reward, done, info = env.step(action)
            
            memory.append((state,action,reward,new_state,done))
            state = new_state
            if done:
                #rewards[-1] = -50 #if lose: punish
                break
    #print("epsilon: ", epsilon)

    #decay the epsilon
    if epsilon > epsilon_f:
        epsilon *= epsilon_decay
    return epsilon


'''
 define function that recalls portions of the information from the practice
 games and learns from it. This is where the Bellman Optimality Equation is
 applied and the targets calculated
'''


def experience_replay(memory,model,train):
    gamma = 0.9 #decay factor gamma
    X = [] #holds the inputs for training
    Y = [] #holds the outputs for training

    #randomly sample a few data from the memory
    memory_batch = np.array(random.sample(memory,batch_size))

    '''
    extract the states and newstates from the memory batch
    then calculate the Q and Q' values based on the states
    '''

    states = np.array([t[0] for t in memory_batch])
    current_qs_list= model.predict(states) #Q(s,a) value based on the main network
    new_states = np.array([t[3] for t in memory_batch])
    future_qs_list = train.predict(new_states)#Q(s',a) based on the target network

    #memory is a list of tuples (state,action,reward,new_state,done)
    for i,tuple in enumerate(memory_batch):
        state = tuple[0]
        action = tuple[1]
        reward = tuple[2]
        done = tuple[4]
        
        '''
        Modified reward system:
        The reward system of CartPole-v0 is always reward = 1 for every
        action that the game is still ongoing. Even if the last action caused
        the game to end, the reward is still 1. Thus, we want to impress
        upon our agent that losing the game is bad, thus whenever the game is "done",
        we assign a negatie q value for that action
        '''
        if done:
            new_q = -20 #Punish Losing
        else:
            new_q = reward + gamma*np.max(future_qs_list[i]) #new q value based on the Bellman Optimality equation

        #update the q values on the current qs list
        current_qs = current_qs_list[i]
        current_qs[action] = new_q

        X.append(state[np.newaxis,...])
        Y.append(current_qs[np.newaxis,...])

    #reshape the data to properly feed the model
    X = np.squeeze(np.array(X))
    Y = np.squeeze(np.array(Y))

    model.fit(X,Y,epochs = 1) #train the model with the updated Q values from the memory batch
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
decay = 0.995 #decay rate for epsilon 

i = 0 #counter for target update
update_target_interval = 5 #we update the target network based on the main network weights every 5 episodes

for episode in range(episodes):
    #initate variables

    #play one practice game and acquire memory
    epsilon = practice_game(main, epsilon,memory)
    
    #print("Length of memory: ", len(memory))
    #learn from the memory and train the model
    main = experience_replay(memory,main,target)
    #break #exit muna for debugging
    if i == 5:
        target.set_weights(main.get_weights())
        i = 0
        
    #evaluate the performance of the model by getting the total score possible

    total = evaluation_game(main)
    print("Average Total Score: {score}, total memory size: {total_memory}, epsilon: {epsilon}  at Episode {episode}".format(score = total, total_memory = len(memory), epsilon = epsilon, episode = episode+1))


    '''
    We use a standard score of 195 to determine if the agent has
    "solved" the environment. If the evaluation shows a score >= standard,
    then we stop the training and save the model.
    '''

    if total >= 195:
        main.save("cartpole_model.h5")
        print("Model achieved goal score of >= 195")
        print("Saving model. Exiting training")
        break
    i += 1






