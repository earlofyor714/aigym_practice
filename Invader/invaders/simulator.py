import gym
import os
import csv
from tqdm import tqdm


class Simulator(object):
    """Handles simulation + logging"""
    def __init__(self, env, agent, display=True, log_metrics=False, optimized=False):
        self.env = env
        self.agent = agent
        self.log_metrics = log_metrics
        self.optimized = optimized
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
        parameters = {}

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

            observation = self.env.reset()
            time = 0.0
            net_reward = 0.0
            success = False

            print("trial {}:".format(trial))
            for _ in range(n_frames):
                try:
                    time += 1.0
                    # self.env.render()
                    action = self.env.action_space.sample()
                    observation, reward, done, info = self.env.step(action)
                    net_reward += reward

                except KeyboardInterrupt:
                    self.quit = True
                finally:
                    if time >= (n_frames-1):
                        success = True
                    if self.quit or done:
                        break

            parameters["a"] = self.agent.alpha
            parameters["e"] = self.agent.epsilon
            if self.log_metrics:
                self.log_writer.writerow({
                    'trial': trial,
                    'testing': testing,
                    'parameters': parameters,
                    'initial_time': 0.0,
                    'final_time': time,
                    'net_reward': net_reward,
                    'actions': 0,
                    'success': success
                })

            if self.quit:
                break

            total_trials += 1
            trial += 1

        if self.log_metrics:
            self.log_file.close()
