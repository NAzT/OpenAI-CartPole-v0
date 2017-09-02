import gym
import random
import statistics as stat
import numpy as np
from collections import Counter

env = gym.make('CartPole-v0')
env.reset()

# Number of steps to take, we won't reach that far
goal_steps = 500
# Minimum score requirement
score_requirement = 50
# Number of episodes for training
training_episodes = 10000


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

    # Convert training_data to numpy array
    training_data_save = np.array(training_data)
    # Save training data
    np.save('trainingdata.npy', training_data_save)

    return training_data


# Generate training data
generate_training_data()