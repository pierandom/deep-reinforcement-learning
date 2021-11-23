import time
import gym
import numpy as np
import torch as th
from torch import nn
from torch.optim import AdamW
from memory import PolicyMemory
from actor_critic import MLPCategoricalActor, MLPValueCritic
from utils import Score



class CartPoleConfig:
    ENV_NAME = 'CartPole-v0'
    HID_DIMS = [64,64]
    ACTIV = nn.Tanh
    EPOCHS = 20
    STEPS_PER_EPOCH = 4000
    OPTIM_STEPS = 80
    GAMMA = 0.99
    LAMBDA = 0.98
    LR = 0.01

    SCORE_TO_BEAT = 196
    SCORE_SIZE = 100

    TEST_EPISODES = 3
    MAX_EPISODE_STEPS = 1000


config = CartPoleConfig



class VPG:
    """
    Vanilla Policy Gradient.
    
    - On-Policy learning algorithm.
    - Stochastic policy.
    - Both discrete and continuous action spaces.
    """

    def __init__(self):
        self.env = gym.make(config.ENV_NAME)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.score = Score(config.SCORE_SIZE, config.SCORE_TO_BEAT)
        self.memory = PolicyMemory(obs_dim, config.STEPS_PER_EPOCH, config.GAMMA, config.LAMBDA)
        self.actor = MLPCategoricalActor(obs_dim, act_dim, config.HID_DIMS, config.ACTIV)
        self.critic = MLPValueCritic(obs_dim, config.HID_DIMS, config.ACTIV)
        self.actor_optimzer = AdamW(self.actor.parameters(), lr=config.LR)
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=config.LR)
    
    def compute_policy_loss(self, obs, act, adv):
        policy = self.actor(obs)
        logp = policy.log_prob(act)
        loss = -th.mean(logp * adv)
        return loss

    def compute_value_loss(self, obs, ret):
        return th.mean((self.critic(obs) - ret)**2)
    
    def update(self):
        data = self.memory.get()
        obs, act, ret, adv = data['obs'], data['act'], data['ret'], data['adv']

        self.actor_optimzer.zero_grad()
        policy_loss = self.compute_policy_loss(obs, act, adv)
        policy_loss.backward()
        self.actor_optimzer.step()

        for _ in range(config.OPTIM_STEPS):
            self.critic_optimizer.zero_grad()
            val_loss = self.compute_value_loss(obs, ret)
            val_loss.backward()
            self.critic_optimizer.step()


    def train(self):
        print("============ TRAINING ============")
        if hasattr(config, "MAX_EPISODE_STEPS"):
            self.env._max_episode_steps = config.MAX_EPISODE_STEPS
        tot_time = 0
        solved_flag = False
        epochs_num_digits = 1 + int(np.log10(config.EPOCHS))
        for epoch in range(config.EPOCHS):
            start_time = time.time()
            epoch_returns = []
            episode_ret, episode_len = 0, 0
            obs = self.env.reset()
            for t in range(config.STEPS_PER_EPOCH):
                with th.no_grad():
                    obs_tensor = th.tensor(obs.reshape(1,-1), dtype=th.float32)
                    policy = self.actor(obs_tensor)
                    act = policy.sample().squeeze()
                    logp = policy.log_prob(act)
                    val = self.critic(obs_tensor)
                next_obs, rew, done, _ = self.env.step(act.numpy())
                self.memory.store(obs, act.numpy(), rew, val.numpy(), logp.numpy())
                obs = next_obs
                episode_ret += rew
                episode_len += 1
                timeout = episode_len == self.env._max_episode_steps
                epoch_ended = t == config.STEPS_PER_EPOCH-1
                if timeout or epoch_ended or done:
                    obs_tensor = th.tensor(obs.reshape(1,-1), dtype=th.float32)
                    last_val = self.critic(obs_tensor).detach().numpy() if timeout or epoch_ended else 0
                    self.memory.end_path(last_val)
                    epoch_returns.append(episode_ret)
                    self.score.update(episode_ret)
                    episode_ret, episode_len = 0, 0
                    obs = self.env.reset()
            
            self.update()
            epoch_time = time.time() - start_time
            tot_time += epoch_time
            tot_time_min, tot_time_sec = map(lambda s: str(round(s)).zfill(2), divmod(tot_time, 60))

            epoch_returns = np.array(epoch_returns)
            print(
                f"Epoch: {epoch+1:{epochs_num_digits}}/{config.EPOCHS} - "
                f"Time: {epoch_time:2.1f}s/{tot_time_min}.{tot_time_sec}min - "
                f"MinRet: {epoch_returns.min():8.2f} - AvgRet: {epoch_returns.mean():8.2f} - "
                f"MaxRet: {epoch_returns.max():8.2f} - StdRet: {epoch_returns.std():8.2f}"
            )
            if self.score.solved and not solved_flag:
                solved_flag = True
                print(
                    f"Environment solved in {self.score.episode-self.score.size} episodes with "
                    f"{self.score.best:.2f} average return over {self.score.size} consecutive episodes"
                )
        
        if not solved_flag:
            print(
                f"Best score is {self.score.best:.2f} average return over "
                f"{self.score.size} consecutive episodes after {self.score.episode} total episodes"
            )
        
    
    def test(self):
        if not config.TEST_EPISODES: return
        print("============ TESTING ============")
        test_episodes_num_digits = 1 + int(np.log10(config.TEST_EPISODES))
        for episode in range(1, 1+config.TEST_EPISODES):
            episode_ret, episode_len = 0, 0
            done = False
            obs = self.env.reset()
            while not done:
                self.env.render()
                with th.no_grad():
                    obs_tensor = th.tensor(obs.reshape(1,-1), dtype=th.float32)
                    policy = self.actor(obs_tensor)
                    act = policy.sample().squeeze()
                obs, rew, done, _ = self.env.step(act.numpy())
                episode_ret += rew
                episode_len += 1
            
            print(
                f"Episode: {episode:{test_episodes_num_digits}}/{config.TEST_EPISODES} - "
                f"Return: {episode_ret:8.2f} - Duration: {episode_len:4} steps"
            )
        self.env.close()



if __name__ == '__main__':
    learner = VPG()
    learner.train()
    learner.test()
