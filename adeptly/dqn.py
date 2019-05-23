"""Simple adaptive agents with finite, countable action spaces"""

import random
from collections import deque
import numpy
import keras


# Deep Q-learning Agent
class DQNAgent:
    """Create and adaptive agent with a simple categorical list of actions.

    # Arguments
    observation_size (int): number of degrees of freedom the agents observes
    action_size (int): number of actions availabe to the agent
    action_weights (list): initial probabilities for the different actions

    # Attributes
    observation_size (int): see constructor argument of same name
    action_size (int): see constructor argument of same name
    action_weights (list): see constructor argument of same name
    memory (collections.deque): memorized past observations, chosen actions and achieved rewards
    gamma (float): discount rate
    epsilon (float): exploration rate
    epsilon_min (float): minimum exploration rate
    epsilon_decay (float): how slowly the exploration rate decreases
    learning_rate (float): rate at which the policy is altered each learning cycle
    model (keras.sequential): the agents keras model

    """

    def __init__(self, observation_size, action_size, action_weights):
        self.observation_size = observation_size
        self.action_size = action_size
        self.action_weights = action_weights
        self.memory = deque(maxlen=1500)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.03
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.observation_size, activation="softsign"))
        model.add(keras.layers.Dense(24, activation="softplus"))
        model.add(keras.layers.Dense(24, activation="softplus"))
        model.add(keras.layers.Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, observation, action, reward, next_observation, done):
        """Append an observation, chosen action and concluding reward to memory."""
        self.memory.append((observation, action, reward, next_observation, done))

    def predict_best_action(self, observation):
        """Based on the agents current policy chose the best action based on an observation."""
        if numpy.random.rand() <= self.epsilon:
            return random.choices(range(self.action_size), weights=self.action_weights)[0]
        predicted_rewards = self.model.predict(observation)
        return numpy.argmax(predicted_rewards[0])  # returns action index

    def replay(self, batch_size):
        """Learn from the past and adapt the policy."""
        minibatch = random.sample(self.memory, batch_size)
        for observation, action, reward, next_observation, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * numpy.amax(self.model.predict(next_observation)[0])
            target_f = self.model.predict(observation)
            target_f[0][action] = target
            self.model.fit(observation, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
