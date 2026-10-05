from collections import defaultdict
import random
import pickle
from algorithms import MOVES

class QLearningNavigator:
    """
    Tabular Q-learning navigation agent.

    The state is local and target-relative:
      relative row (clipped)
      relative column (clipped)
      obstacle mask around the truck

    The task planner supplies the target.
    """

    def __init__(self, alpha=0.15, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, seed=42):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q = defaultdict(lambda: [0.0] * 4)

    def state(self, env, target):
        r, c = env.truck.position
        tr, tc = target
        dr = max(-(env.rows - 1), min(env.rows - 1, tr - r))
        dc = max(-(env.cols - 1), min(env.cols - 1, tc - c))

        mask = []
        for action in range(4):
            rr = r + MOVES[action][0]
            cc = c + MOVES[action][1]
            blocked = (
                rr < 0 or rr >= env.rows or
                cc < 0 or cc >= env.cols or
                (rr, cc) in env.obstacles
            )
            mask.append(int(blocked))

        return (dr, dc, *mask)

    def choose_action(self, state, greedy=False):
        values = self.q[state]
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(4)
        best = max(values)
        candidates = [i for i, v in enumerate(values) if v == best]
        return self.rng.choice(candidates)

    def update(self, state, action, reward, next_state, done):
        current = self.q[state][action]
        future = 0.0 if done else max(self.q[next_state])
        target = reward + self.gamma * future
        self.q[state][action] += self.alpha * (target - current)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q), f)

    def load(self, path):
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        self.q.update(loaded)
