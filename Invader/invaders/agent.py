import gym

from Invader.invaders.simulator import Simulator


class Agent(object):
    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        self.valid_actions = env.action_space.n
        self.mem_state = None

        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha

    def update(self):
        pass

    def reset(self):
        pass

    def get_state(self):
        return self.mem_state

def run():
    env = gym.make("SpaceInvaders-v0")
    # env = gym.make("Breakout-v0")
    observation = env.reset()

    agent = Agent(env)
    sim = Simulator(env, display=True, log_metrics=False, optimized=False)
    sim.run()


if __name__ == "__main__":
    run()
