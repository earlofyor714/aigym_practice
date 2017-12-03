import os
import csv
import time
# from tqdm import tqdm

from Invader.invaders.agent import Agent
from Invader.invaders.environment import Environment


class Simulator(object):
    """Handles simulation + logging"""
    def __init__(self, env, display=True, log_metrics=False, optimized=False, filename="sim"):
        self.env = env
        self.agent = env.agent
        self.log_metrics = log_metrics
        self.optimized = optimized
        self.display = display
        self.quit = False

        if self.log_metrics:
            if self.agent.learning:
                if self.optimized:
                    self.log_filename = os.path.join("../logs", filename+"_improved-learning.csv")
                else:
                    self.log_filename = os.path.join("../logs", filename+"_default-learning.csv")
            else:
                self.log_filename = os.path.join("../logs", filename+"_basic.csv")

            self.log_fields = ['trial', 'testing', 'parameters', 'initial_time', 'final_time', 'net_reward', 'actions', 'success']
            self.log_file = open(self.log_filename, 'w', newline='')
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.log_fields)
            self.log_writer.writeheader()

    def run(self, tolerance=0.05, n_test=0, n_frames=3000):
        self.quit = False
        is_testing = False
        total_trials = 1
        trial = 1
        start_time = time.time()

        while True:
            if not is_testing:
                is_testing, trial = self.determine_testing_status(trial, total_trials, tolerance)
            else:
                if trial > n_test:
                    break

            self.env.reset(is_testing)
            print("trial {}:".format(trial))
            # self.quit = self.env.step(n_frames, is_display=self.display)
            self.quit = self.env.tensorflow_step(n_frames, is_display=self.display)
            self.log_trial(trial)

            if self.quit:
                break
            total_trials += 1
            trial += 1
        if self.log_metrics:
            self.log_file.close()
        final_time = time.time()
        print("Total time: {} seconds".format(final_time - start_time))

    def tensorflow_run(self, tolerance=0.05, n_test=0, n_frames=3000):
        start_time = time.time()

        print("training set:")
        self.quit = self.env.tensorflow_step(tolerance, n_frames, is_display=self.display)

        print("validation set:")

        if self.log_metrics:
            self.log_file.close()
        final_time = time.time()
        print("Total time: {} seconds".format(final_time - start_time))

    def determine_testing_status(self, trial, total_trials, tolerance):
        if total_trials > 20:
            if self.agent.learning:
                if self.agent.epsilon < tolerance:
                    return True, 1
            else:
                return True, 1
        return False, trial

    def log_trial(self, trial):
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

# To do: 3 game loops: testing/training, trials, game frame loops

if __name__=="__main__":
    environment = Environment()
    agent = Agent(environment)
    environment.set_agent(agent)
    sim = Simulator(environment, display=False, log_metrics=True, optimized=False)

    sim.tensorflow_run(n_test=1)
