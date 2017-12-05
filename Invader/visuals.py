import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import ast


def plot_trials(csv):
    data = pd.read_csv(os.path.join("logs", csv))
    all_data = append_features_to(data)

    if len(data) < 10:
        print("Not enough data collected to create a visualization.")
        print("At least 20 trials are required.")
        return

    training_data = all_data[all_data['testing'] == False]
    testing_data = all_data[all_data['testing'] == True]

    plt.figure(figsize=(12, 8))

    ###############
    ### Average step reward plot
    ###############

    ax = plt.subplot2grid((6, 6), (0, 3), colspan=3, rowspan=2)
    ax.set_title("10-Trial Rolling Average Reward per Action")
    ax.set_ylabel("Reward per Action")
    ax.set_xlabel("Trial Number")
    ax.set_xlim((10, len(training_data)))

    # Create plot-specific data
    step = training_data[['trial', 'average_reward']].dropna()

    ax.axhline(xmin=0, xmax=1, y=0, color='black', linestyle='dashed')
    ax.plot(step['trial'], step['average_reward'])

    ###############
    ### Parameters Plot
    ###############

    ax = plt.subplot2grid((6, 6), (2, 3), colspan=3, rowspan=2)

    # Check whether the agent was expected to learn
    if csv != 'sim_basic.csv':
        ax.set_ylabel("Parameter Value")
        ax.set_xlabel("Trial Number")
        ax.set_xlim((1, len(training_data)))
        ax.set_ylim((0, 1.05))

        ax.plot(training_data['trial'], training_data['epsilon'], color='blue', label='Exploration factor')
        ax.plot(training_data['trial'], training_data['alpha'], color='green', label='Learning factor')

        ax.legend(bbox_to_anchor=(0.5, 1.19), fancybox=True, ncol=2, loc='upper center', fontsize=10)

    else:
        ax.axis('off')
        ax.text(0.52, 0.30, "Simulation completed\nwith learning disabled.", fontsize=24, ha='center', style='italic')

    ###############
    ### Rolling Success-Rate plot
    ###############

    ax = plt.subplot2grid((6, 6), (4, 0), colspan=4, rowspan=2)
    ax.set_title("10-Trial Rolling Rate of Reliability")
    ax.set_ylabel("Rate of Reliability")
    ax.set_xlabel("Trial Number")
    ax.set_xlim((10, len(training_data)))
    ax.set_ylim((-5, 105))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    # Create plot-specific data
    trial = training_data.dropna()['trial']
    rate = training_data.dropna()['reliability_rate']

    # Rolling success rate
    ax.plot(trial, rate, label="Reliability Rate", color='blue')

    ###############
    ### Test results
    ###############

    ax = plt.subplot2grid((6, 6), (4, 4), colspan=2, rowspan=2)
    ax.axis('off')

    if len(testing_data) > 0:
        # safety_rating, safety_color = calculate_safety(testing_data)
        reliability_rating, reliability_color = calculate_reliability(testing_data)

        # Write success rate
        ax.text(0.40, .9, "{} testing trials simulated.".format(len(testing_data)), fontsize=14, ha='center')
        # ax.text(0.40, 0.7, "Safety Rating:", fontsize=16, ha='center')
        # ax.text(0.40, 0.42, "{}".format(safety_rating), fontsize=40, ha='center', color=safety_color)
        ax.text(0.40, 0.27, "Reliability Rating:", fontsize=16, ha='center')
        ax.text(0.40, 0, "{}".format(reliability_rating), fontsize=40, ha='center', color=reliability_color)

    else:
        ax.text(0.36, 0.30, "Simulation completed\nwith testing disabled.", fontsize=20, ha='center', style='italic')

    # Plot everything
    plt.tight_layout()
    plt.show()


def append_features_to(data):
    average_reward = (data['net_reward'] / (data['final_time'] - data['initial_time'])).rolling(window=10).mean()
    reliability_rate = (data['success'] * 100).rolling(window=10).mean()
    epsilon = data['parameters'].apply(lambda x: ast.literal_eval(x)['e'])
    alpha = data['parameters'].apply(lambda x: ast.literal_eval(x)['a'])

    new_data = pd.DataFrame({'average_reward': average_reward,
                             'reliability_rate': reliability_rate,
                             'epsilon': epsilon,
                             'alpha': alpha})

    result = pd.concat([data, new_data], axis=1)

    return result


def calculate_reliability(data):
    """ Calculates the reliability rating of the smartcab during testing. """

    success_ratio = data['success'].sum() * 1.0 / len(data)

    if success_ratio == 1: # Always meets deadline
        return ("A+", "green")
    else:
        if success_ratio >= 0.90:
            return ("A", "green")
        elif success_ratio >= 0.80:
            return ("B", "green")
        elif success_ratio >= 0.70:
            return ("C", "#EEC700")
        elif success_ratio >= 0.60:
            return ("D", "#EEC700")
        else:
            return ("F", "red")

# TODO: graph out avg time of runs, and high scores
