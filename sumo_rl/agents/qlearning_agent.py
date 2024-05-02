"""Q-learning Agent class."""
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
import pickle

class QLAgent:
    """Q-learning Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
            reward + self.gamma * max(self.q_table[s1]) - self.q_table[s][a]
        )
        self.state = s1
        self.acc_reward += reward
        
    def eval_step(self, next_state, reward):
        self.state = next_state
        self.acc_reward += reward
        
    @staticmethod
    def save_all(agents, name):
        name = ".\\models\\" + name + ".pkl"
        q_tables = {ts: agent.q_table for ts, agent in agents.items()}
        with open(name, 'wb') as f:
            pickle.dump(q_tables, f)

    @staticmethod
    def load_all(agents, name):
        name = ".\\models\\" + name + ".pkl"
        with open(name, 'rb') as f:
            q_tables = pickle.load(f)
        for ts, agent in agents.items():
            agent.q_table = q_tables[ts]