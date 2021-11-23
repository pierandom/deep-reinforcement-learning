import numpy as np

def discount_cumsum(v: np.ndarray, discount: float) -> np.ndarray:
    """
    For a given vector
    v = [
        v0,
        v1,
        v2,
        ...
    ]
    and discount factor, computes the vector
    w = [
        v0 + discount * v1 + discount^2 * v2 + ...,
        v1 + discount * v2 + ...,
        v2 + ...,
        ...
    ]
    """

    u = np.power(discount, range(len(v)))
    return np.array([np.sum(v[-i-1:]*u[:i+1]) for i in reversed(range(len(v)))])


class Score:
    """Compute the average return for the last <size> episodes and keep track of the best score."""
    
    def __init__(self, size=100, score_to_beat=None):
        self.to_beat = score_to_beat if score_to_beat is not None else np.inf #score to beat could be 0
        self.episode_returns = np.zeros(size, dtype=np.float32)
        self.episode = 0
        self.size = size
        self.best = -np.inf
    
    def update(self, ep_ret):
        self.episode_returns[self.episode % self.size] = ep_ret
        self.episode += 1
        if self.episode >= self.size:
            self.best = max(self.episode_returns.mean(), self.best)
    
    @property
    def solved(self):
        return self.best > self.to_beat
    