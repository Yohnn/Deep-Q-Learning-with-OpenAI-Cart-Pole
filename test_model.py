import tensorflow as tf
import gym
import numpy as np


#load the models

print("Load Model")
main = tf.keras.models.load_model("cartpole_model.h5")

print("Play Some Games")
#Initiate the Cartpole Environment
env = gym.make('CartPole-v0')

episodes = 5 #number of games to play
goal_steps = 500 #number of max time steps per game

for episode in range(episodes):
    state = env.reset()
    for _ in range(goal_steps):
        env.render()
        action = main.predict(state[np.newaxis,...])
        new_state, reward, done, ___ = env.step(np.argmax(action))
        state = new_state
        if done:
            break
        
env.close()
print("games finished")