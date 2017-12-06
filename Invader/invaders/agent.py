import random
import numpy as np

from Invader.invaders.agents.practice_lstm_agent import LstmAgent


class Agent(object):
    def __init__(self, env, graph=None, learning=False, epsilon=1.0, alpha=0.5):
        self.env = env
        self.mem_state = None
        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha
        self.timer = 1.0
        input_size = 160*210
        self.ai = LstmAgent(input_size, graph=graph)

    """ 3.472135955 / timer = # trials """
    def reset(self, testing=False):
        if testing:
            self.epsilon = 0
            self.alpha = 0
            self.ai.alpha = 0
            return
        self.epsilon = 1.0 / (self.timer * self.timer)
        # self.timer += 0.001
        self.timer += 0.011

    def update(self, session=None):
        state = self.build_state(self.env.current_state)
        action, allQ = self.choose_action(state, session=session)
        next_state, reward, is_terminated, _ = self.env.act(action)
        next_built_state = self.build_state(next_state)
        self.learn(state, action, allQ, reward, next_built_state, session=session)
        return next_state, reward, is_terminated

    def build_state(self, state):
        mean_state = np.mean(state, axis=2)
        state = np.divide(mean_state, 256)
        state = [state.flatten()]
        return state

    def choose_action(self, state, session=None):
        output_size = 6
        x = random.uniform(0, 1)
        if x < self.epsilon or not self.learning:
            action = self.env.get_action_space().sample()
            allQ = np.identity(output_size)[action:action+1]
            return action, allQ
        return self.ai.choose_action(state, session)

    def learn(self, state, action, allQ, reward, next_state, session=None):
        if self.learning:
            self.ai.learn(state, action, allQ, reward, next_state, session)
        return

    def get_state(self):
        return self.mem_state

    def save(self, filepath, session=None):
        return self.ai.save_vars(session, filepath)

    def load(self, filepath, session=None):
        self.ai.restore_vars(session, filepath)
