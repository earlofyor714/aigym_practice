import os
import csv
from tqdm import tqdm

from Invader.invaders.agent import Agent
from Invader.invaders.environment import Environment


class Simulator(object):
    """Handles simulation + logging"""
    def __init__(self, env, display=True, log_metrics=False, optimized=False):
        self.env = env
        self.agent = env.agent
        self.log_metrics = log_metrics
        self.optimized = optimized
        self.display = display
        self.quit = False

        if self.log_metrics:
            if self.agent.learning:
                if self.optimized:
                    self.log_filename = os.path.join("../logs", "sim_improved-learning.csv")
            else:
                self.log_filename = os.path.join("../logs", "sim_basic.csv")

            self.log_fields = ['trial', 'testing', 'parameters', 'initial_time', 'final_time', 'net_reward', 'actions', 'success']
            self.log_file = open(self.log_filename, 'w', newline='')
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.log_fields)
            self.log_writer.writeheader()

    def run(self, tolerance=0.05, n_test=0, n_frames=3000):
        self.quit = False
        testing = False
        total_trials = 1
        trial = 1

        while True:
            if not testing:
                if total_trials > 20:
                    if self.agent.learning:
                        if self.agent.epsilon < tolerance:
                            testing = True
                            trial = 1
                    else:
                        testing = True
                        trial = 1
            else:
                if trial > n_test:
                    break

            self.env.reset(testing)
            time = 0.0

            print("trial {}:".format(trial))
            for _ in range(n_frames):
                try:
                    time += 1.0
                    if self.display:
                        self.env.render()
                    state, reward, done, info = self.env.step()

                except KeyboardInterrupt:
                    self.quit = True
                finally:
                    if time >= (n_frames-1):
                        success = True
                    if self.quit or done:
                        break

            if self.log_metrics:
                self.log_writer.writerow({
                    'trial': trial,
                    'testing': self.env.trial_data['testing'],
                    'parameters': self.env.trial_data['parameters'],
                    'initial_time': self.env.trial_data['initial_time'],
                    'final_time': self.env.trial_data['final_time'],
                    'net_reward': self.env.trial_data['net_reward'],
                    'actions': self.env.trial_data['actions'],
                    'success': self.env.trial_data['success']
                })

            if self.quit:
                break

            total_trials += 1
            trial += 1

        if self.log_metrics:
            self.log_file.close()

# To do: 3 game loops: testing/training, trials, game frame loops

if __name__=="__main__":
    env = Environment()
    agent = Agent(env)
    env.set_agent(agent)
    sim = Simulator(env, display=True, log_metrics=False, optimized=False)

    sim.run(n_test=1)
