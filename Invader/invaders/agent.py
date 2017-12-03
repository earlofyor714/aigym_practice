import random
import numpy as np

from Invader.invaders.agents.practice_lstm_agent import LstmAgent


class Agent(object):
    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        self.env = env
        self.mem_state = None
        input_size = 160*210
        self.ai = LstmAgent(input_size)

        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha

        self.timer = 1.0

    def reset(self, testing=False):
        if testing:
            self.epsilon = 0
            self.alpha = 0
            return
        self.epsilon = 1.0 / (self.timer * self.timer)
        self.timer += 0.001

    def update(self):
        state = self.build_state()
        action = self.choose_action(state)
        next_state, reward, is_terminated, _ = self.env.act(action)
        self.learn(state, action, reward, next_state)
        return next_state, reward, is_terminated

    def build_state(self):
        state = np.divide(np.mean(self.env.current_state, axis=2), 256)
        return state

    def choose_action(self, state):
        action = None
        x = random.uniform(0, 1)
        if x < self.epsilon or not self.learning:
            action = self.env.get_action_space().sample()
        else:
            # TODO: choose initial action via AI logic
            action = self.env.get_action_space().sample()
        return action

    def learn(self, state, action, reward, next_state):
        if self.learning:
            # TODO: add support for AI logic here
            return
        return

    def tensorflow_update(self, state, tolerance, n_frames, is_display=True):
        self.epsilon, is_quit = self.ai.tensorflow_learn(state, self.env, tolerance, n_frames, is_display)
        return is_quit

    def get_state(self):
        return self.mem_state
