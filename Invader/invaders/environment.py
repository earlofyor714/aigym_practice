import gym

from Invader.invaders.agents.practice_lstm_agent import LstmAgent


class Environment:
    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')

        # input_size = 210*160
        # self.agent = LstmAgent(self.env, input_size)

    def reset(self):
        self.env.reset()

    def step(self):
        # self.agent.run()
        action = self.env.action_space.sample()
        state, reward, terminated, info = self.env.step(action)
        self.env.render()
        return state, reward, terminated, info


if __name__=="__main__":
    env = Environment()
    # env = gym.make('SpaceInvaders-v0')
    state = env.reset()
    for _ in range(9000):
        # action = env.action_space.sample()
        # state, reward, terminated, info = env.step(action)
        state, reward, terminated, info = env.step()
        # env.render()
        if terminated:
            print("terminated")
            break
