import gym

from Invader.invaders.agents.practice_lstm_agent import LstmAgent


class Environment:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        # input_size = 210*160
        # self.agent = LstmAgent(self.env, input_size)
        self.agent = None
        self.trial_data = {
            'testing': False,
            'parameters': dict(),
            'initial_time': 0.0,
            'final_time': 0.0,
            'net_reward': 0.0,
            'actions': 0,
            'success': False
        }

    def reset(self, testing=False):
        self.env.reset()

        self.trial_data['testing'] = testing
        self.trial_data['parameters'] = {'e': self.agent.epsilon, 'a': self.agent.alpha}
        self.trial_data['initial_time'] = 0.0
        self.trial_data['final_time'] = 0.0
        self.trial_data['net_reward'] = 0.0
        self.trial_data['success'] = False

    def step(self):
        action = self.agent.update()
        state, reward, is_terminated, info = self.env.step(action)

        self.trial_data['final_time'] += 1
        self.trial_data['net_reward'] += reward
        self.trial_data['success'] = False

        return is_terminated

    def render(self):
        self.env.render()

    def set_agent(self, agent):
        self.agent = agent

    def get_action_space(self):
        return self.env.action_space
