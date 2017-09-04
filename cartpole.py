import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression 


env = gym.make('CartPole-v0')
env.reset()
# Number of steps to take
goal_steps = 500
# Minimum score requirement
score_requirement = 50
# Number of episodes for training
training_episodes = 10000
# Number of episodes for testing
required_episodes = 100

def generate_training_data():
    # Training data
    training_data = []
    # Scores that met/above threshold
    scores_above_threshold = []
    # Training episodes
    for episode in range(training_episodes):
        score = 0
        # Data specific to episode
        episode_data = []
        # Previous observation
        prev_observation = []
        # Take action
        for steps in range(goal_steps):
            # Random action (0 - move left or 1 - move right)
            action = random.randrange(0, 2)
            # Take step
            observation, reward, done, info = env.step(action)
            # Append to episode data
            if len(prev_observation) > 0:
                episode_data.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        # Check if score above requirement
        if score >= score_requirement:
            scores_above_threshold.append(score)
            for data in episode_data:
                # Convert to one-hot
                if data[1] == 1:
                    # Balanced - score: 1
                    output = [0, 1]
                elif data[1] == 0:
                    # Unbalanced - score: 0
                    output = [1, 0]

                # Save training data
                training_data.append([data[0], output])

        # Reset env
        env.reset()

    return training_data


# Neural Network model
def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')
    # Convolution
    branch1 = tflearn.conv_1d(network, 128, 2, padding='valid', activation='relu', regularizer="L2")
    branch2 = tflearn.conv_1d(network, 128, 2, padding='valid', activation='relu', regularizer="L2")
    branch3 = tflearn.conv_1d(network, 128, 2, padding='valid', activation='relu', regularizer="L2")
    branch4 = tflearn.conv_1d(network, 128, 2, padding='valid', activation='relu', regularizer="L2")
    branch5 = tflearn.conv_1d(network, 128, 2, padding='valid', activation='relu', regularizer="L2")
    network = tflearn.merge([branch1, branch2, branch3, branch4, branch5], mode='concat', axis=1)
    # LSTM
    network = tflearn.lstm(network, 128, activation='relu', dropout=0.8)
    # Fully connected
    network = tflearn.fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = tflearn.fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = tflearn.fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
    network = tflearn.fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = tflearn.fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = tflearn.regression(network, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
    model = tflearn.DNN(network, tensorboard_verbose=0)
    return model


# Train model
def train_model(training_data, model=False):
    # Observation
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]),1)
    # Action
    y = [i[1] for i in training_data]

    # Model
    if not model:
        model = neural_network_model(input_size = len(X[0]))
    # Train the model
    model.fit(X, y, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model


def render_and_test():
    training_data = generate_training_data()
    model = train_model(training_data)
    scores = []
    actions_done = []
    # Moniter gym
    for game in range(required_episodes):
        score = 0
        episode_data = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs), 1))[0])

            actions_done.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            episode_data.append([new_observation, action])
            score += reward
            if done: break
            
        scores.append(score)

    print('Average Score:', sum(scores) / len(scores))

# Render and test
render_and_test()