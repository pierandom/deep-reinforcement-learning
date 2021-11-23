import numpy as np
import torch as th
from utils import discount_cumsum

class PolicyMemory:
    """A buffer to store the agent experience for on-policy learning"""

    def __init__(self, obs_dim, size, gamma, lambda_):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lambda_ = gamma, lambda_
        self.idx, self.path_start_idx, self.max_size = 0, 0, size
    

    def store(self, obs, act, rew, val, logp):
        self.obs_buf[self.idx] = obs
        self.act_buf[self.idx] = act
        self.rew_buf[self.idx] = rew
        self.val_buf[self.idx] = val
        self.logp_buf[self.idx] = logp
        self.idx += 1
    

    def end_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.idx)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # compute GAE-lambda (Generalized Advantage Estimation) estimates
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.lambda_)

        # compute reward to go
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.idx
    

    def get(self):
        self.idx, self.path_start_idx = 0, 0
        data = {
            'obs': th.from_numpy(self.obs_buf),
            'act': th.from_numpy(self.act_buf),
            'ret': th.from_numpy(self.ret_buf),
            'adv': th.from_numpy(self.adv_buf),
            'val': th.from_numpy(self.val_buf),
            'logp': th.from_numpy(self.logp_buf)
        }
        return data



class ReplayMemory:
    """A buffer to store the agent experience for off-policy learning"""

    def __init__(self, obs_shape, act_shape, size):
        self.obs_buf = np.zeros((size, obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((size, act_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_shape), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.idx, self.size, self.max_size = 0, 0, size
    
    def store(self, obs, act, next_obs, rew, done):
        self.obs_buf[self.idx] = obs
        self.act_buf[self.idx] = act
        self.next_obs_buf[self.idx] = next_obs
        self.rew_buf[self.idx] = rew
        self.done_buf[self.idx] = done
        self.idx = (self.idx+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {
            'obs': th.from_numpy(self.obs_buf[idxs]),
            'act': th.from_numpy(self.act_buf[idxs]),
            'next_obs': th.from_numpy(self.next_obs_buf[idxs]),
            'rew': th.from_numpy(self.rew_buf[idxs]),
            'done': th.from_numpy(self.done_buf[idxs])
        }
        return batch
