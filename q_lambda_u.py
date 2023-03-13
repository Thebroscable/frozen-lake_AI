import numpy as np
import random
import hickle


class Agent:
    def __init__(self, n_states, n_actions,
                 alpha, gamma, lambda_, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_ = lambda_

        self.n_states = n_states
        self.n_actions = n_actions

        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.eligibility_trace = self.q_table.copy()

    def train(self, state, action, reward, next_state, next_action, done):
        old_value = self.q_table[state, action]

        if done:
            next_value = 0
        else:
            next_value = np.amax(self.q_table[next_state])

        delta = reward + self.gamma * next_value - old_value
        self.eligibility_trace[state, action] += 1

        self.q_table += self.alpha * delta * self.eligibility_trace
        self.eligibility_trace *= self.lambda_ * self.gamma

        if done:
            self.eligibility_trace = np.zeros([self.n_states, self.n_actions])

    def make_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            action = self.create_action(state)

        return action

    def create_action(self, state):
        action_list = self.q_table[state].copy()
        index_list = [i for i in range(4)]

        if len(self.replay_buffer) > 0:
            last_mem = self.replay_buffer[-1]
            if state == last_mem[0]:
                action_list = np.delete(action_list, last_mem[1])
                index_list = np.delete(index_list, last_mem[1])
            else:
                action_list = np.delete(action_list, (last_mem[1]+2) % 4)
                index_list = np.delete(index_list, (last_mem[1]+2) % 4)

        best_action_value = np.amax(action_list)
        valid_action_list = [index_list[i] for i, x in enumerate(action_list) if x == best_action_value]
        return random.choice(valid_action_list)

    def reset(self):
        self.q_table = np.zeros([self.n_states, self.n_actions])

    def save_data(self):
        hickle.dump(self.q_table, 'model/q_lambda_u.hkl')

    def load_data(self):
        self.q_table = hickle.load('model/q_lambda_u.hkl')
