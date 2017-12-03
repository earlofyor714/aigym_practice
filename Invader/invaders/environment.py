import gym


class Environment:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        # input_size = 210*160
        # self.agent = LstmAgent(self.env, input_size)
        self.agent = None
        self.current_state = None
        self.done = False
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
        self.current_state = self.env.reset()

        self.trial_data['testing'] = testing
        self.trial_data['parameters'] = {'e': self.agent.epsilon, 'a': self.agent.alpha}
        self.trial_data['initial_time'] = 0.0
        self.trial_data['final_time'] = 0.0
        self.trial_data['net_reward'] = 0.0
        self.trial_data['success'] = False

    def step(self, n_frames, is_display=True):
        is_quit = False
        time = 0.0
        for _ in range(n_frames):
            try:
                time += 1.0
                if is_display:
                    self.env.render()
                self.current_state, reward, self.done = self.agent.update()
                self.trial_data['final_time'] += 1
                self.trial_data['net_reward'] += reward
            except KeyboardInterrupt:
                is_quit = True
            finally:
                if time >= (n_frames - 1):
                    self.trial_data['success'] = True
                if is_quit or self.done:
                    return is_quit
        # self.current_state, reward, self.done = self.agent.update()
        return is_quit

    """Tensorflow requires everything be done in a single session, which forces game logic onto the agent :("""
    def tensorflow_step(self, tolerance, n_frames, is_display=True):
        self.reset(False)
        is_quit = self.agent.tensorflow_update(self.current_state, tolerance, n_frames, is_display)
        return is_quit

    def act(self, action):
        state, reward, is_terminated, info = self.env.step(action)
        return state, reward, is_terminated, info

    def render(self):
        self.env.render()

    def set_agent(self, agent):
        self.agent = agent

    def get_action_space(self):
        return self.env.action_space
