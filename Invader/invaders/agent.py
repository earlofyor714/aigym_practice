import gym

from Invader.invaders.simulator import Simulator


class LearningAgent(object):
    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        self.valid_actions = env.action_space.n

        self.learning = learning
        self.epsilon = epsilon
        self.alpha = alpha

def run():
    env = gym.make("SpaceInvaders-v0")
    # env = gym.make("Breakout-v0")
    observation = env.reset()

    agent = LearningAgent(env)
    sim = Simulator(env, agent, display=True, log_metrics=True, optimized=False)
    sim.run()


if __name__ == "__main__":
    run()
