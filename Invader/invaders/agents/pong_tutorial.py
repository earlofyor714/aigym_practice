import numpy as np
import _pickle as pickle
import gym

# Policy gradient optimizes for short term rewards

# hyperparameters
num_hidden_neurons = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.9  # discount factor for policy gradients
decay_rate = 0.99  # gradient descent
resume = True

# init model
input_dim = 80*80
if resume:
    model = pickle.load(open('save.p', 'rb'))
    print("model loaded")
else:
    # xavier initialization. solves: If start init values too small, becomes insignificant. If too big, gets amplified.)
    model = {'W1': np.random.randn(num_hidden_neurons, input_dim) / np.sqrt(input_dim),
             'W2': np.random.randn(num_hidden_neurons) / np.sqrt(num_hidden_neurons)}
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # way to store gradients for back-propagation

rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # store value for rms prop formula, inc. multipliers
# Note: rmsprop is stochastic gradient descent

np.seterr(divide='ignore')


# activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # squashing


def preprocess(game_frame):
    """ Image converted to 0 or 1 values """
    game_frame = game_frame[35:195]  # columns that contain the paddles
    game_frame = game_frame[::2, ::2, 0]
    game_frame[game_frame==144] = 0  # erase background
    game_frame[game_frame==109] = 0  # erase more background
    game_frame[game_frame != 0] = 1  # paddles, and balls set to 1
    return game_frame.astype(np.float).ravel()


def discount_reward(rewards):
    """ If y = discount rate, and rewards = [r0, r1, r2, r3],
        returns [d0, d1, d2, d3] where
        d0 = r0 + y * r1 + y^2 * r2 + y^3 * r3
        d1 = r1 + y * r2 + y^2 * r3
        d2 = r2 + y * r3
        d3 = r3
        31:10
    """
    discounted_r = np.zeros_like(rewards)
    running_reward_sum = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0: running_reward_sum = 0
        running_reward_sum = running_reward_sum * gamma + rewards[t]
        discounted_r[t] = running_reward_sum
    return discounted_r


def policy_forward(game_pixels):
    hidden_state = np.dot(model['W1'], game_pixels)
    hidden_state[hidden_state < 0] = 0  # ReLU.Note:ReLU used during calc, sigmoid used @ end of network (probabilities)
    logp = np.dot(model['W2'], hidden_state)
    probabilities = sigmoid(logp)
    return probabilities, hidden_state


def policy_backward(epx, eph, epdlogp):
    """ epdlogp is going to modulate the gradient with advantage
        eph is array of intermediate hidden states """
    d_W2 = np.dot(eph.T, epdlogp).ravel()
    d_hidden_state = np.outer(epdlogp, model['W2'])
    d_hidden_state[eph <= 0] = 0  # ReLU
    d_W1 = np.dot(d_hidden_state.T, epx)

    return {'W1': d_W1, 'W2': d_W2}


# Implementation details
env = gym.make('Pong-v0')
observation = env.reset()
prev_frame = None  # Want to calculate motion of ball, so going to take difference between 2 frames n use that
obs_storage, hid_state_storage, grad_storage, reward_storage = [], [], [], []
running_reward = None
reward_sum = 0
episode_num = 0

# Begin training
while True:
    cur_frame = preprocess(observation)
    d_frame = cur_frame - prev_frame if prev_frame is not None else np.zeros(input_dim)
    prev_frame = cur_frame

    # forward!!
    aprob, h = policy_forward(d_frame)
    # stochastic
    action = 2 if np.random.uniform() < aprob else 3
    obs_storage.append(d_frame)
    hid_state_storage.append(h)
    y = 1 if action == 2 else 0  # a "fake label"
    grad_storage.append(y - aprob)

    # step the environment
    env.render()
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    reward_storage.append(reward)

    if done:
        episode_num += 1

        # stack i/o
        ep_obs = np.vstack(obs_storage)
        ep_hid_states = np.vstack(hid_state_storage)
        ep_grad = np.vstack(grad_storage)
        ep_rewards = np.vstack(reward_storage)
        obs_storage, hid_state_storage, grad_storage, reward_storage = [], [], [], []

        # discounted reward computation
        discounted_ep_reward = discount_reward(ep_rewards)
        discounted_ep_reward -= np.mean(discounted_ep_reward)
        discounted_ep_reward /= np.std(discounted_ep_reward)

        # advantage - qty which describes how good the action is compared to the avg of all the action
        ep_grad *= discounted_ep_reward
        grad = policy_backward(ep_obs, ep_hid_states, ep_grad)
        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_num % batch_size == 0:
            for k, v in model.items():
                gradient = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * gradient**2
                model[k] += learning_rate * gradient / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        # book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env.  Episode reward total was {}. running mean: {}'.format(reward_sum, running_reward))
        if episode_num % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()
        prev_frame = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends
        print('ep {}: game finished, reward: {}'.format(episode_num, reward))


# My notes on model:
# How to handle more than 2 actions?
# Alternatives to rmsprop?
# Ways to speed it up? i.e. multiple games running at once, using gpu,
