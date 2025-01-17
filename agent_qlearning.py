# Q-learning algorithm. It is the brain behind the agent of reinforcement learning.
import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learning_rate=0.05, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions  # number of possible actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        # add this observation to the table
        self.add_state(observation)

        # action selection based on greedy policy
        if np.random.uniform() < self.epsilon:
            # choose a random action
            action = np.random.choice(self.actions)
        else:
            # choose the best action for the given observation
            state_str = str(observation)
            state_action = self.q_table.loc[state_str, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        return action

    def learn(self, s, a, r, s_):
        # add the current and next observations to the table
        self.add_state(s)
        self.add_state(s_)

        # choose the best q-value for the given pair of (s, a); Q(s, a)
        q_predict = self.q_table.loc[str(s), a]

        # check if the next state is a terminal state or not and get the expected q value
        if s_ != 'terminal':
            # next state is not terminal
            # approximate the expected future reward based on Bellman equation:
            # Q'() = r + gamma * [max_a' Q(s',a')]
            q_target = r + self.gamma * self.q_table.loc[str(s_), :].max()
        else:
            q_target = r  # next state is terminal

        # Update q-value in the table
        # Q(s, a) = Q(s, a) + learning_rate [r + gamma max_a' Q(s', a') - Q(s, a)]
        self.q_table.loc[str(s), a] += self.lr * (q_target - q_predict)

    def add_state(self, state):
        if str(state) not in self.q_table.index:
            # append new state to q table
            row = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=str(state)
            )
            self.q_table = pd.concat([self.q_table, row.to_frame().T])

    def get_q_table(self):
        return self.q_table
