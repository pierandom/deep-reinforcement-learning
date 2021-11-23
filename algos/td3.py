from copy import deepcopy
import time
import gym
import numpy as np
import torch as th
from torch import nn
from torch.optim import AdamW
from memory import ReplayMemory
from actor_critic import MLPDeterministicActor, MLPQualityCritic
from utils import Score


class PendulumConfig:
    ENV_NAME = "Pendulum-v0"
    BATCH_SIZE = 64
    MEMORY_SIZE = int(1e6)
    EPOCHS = 20
    STEPS_PER_EPOCH = 4000
    UPDATE_EVERY = 200
    NUM_UPDATES = 100
    WARMUP_EPOCHS = 1
    POLICY_DELAY = 2
    HID_DIMS = [128,128]
    ACTIV = nn.ReLU
    LR = 0.001
    GAMMA = 0.99
    RHO = 0.99

    SCORE_TO_BEAT = None
    SCORE_SIZE = 100

    TEST_EPISODES = 5



config = PendulumConfig




class TD3:
    """
    Twin Delayed DDPG.
    
    - Off-Policy learning algorithm.
    - Deterministic policy.
    - Continuous-time action spaces only.
    """

    def __init__(self):
        self.env = gym.make(config.ENV_NAME)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.score = Score(config.SCORE_SIZE, config.SCORE_TO_BEAT)
        self.memory = ReplayMemory(obs_dim, act_dim, config.MEMORY_SIZE)
        self.actor = MLPDeterministicActor(obs_dim, act_dim, config.HID_DIMS, config.ACTIV)
        self.critic1 = MLPQualityCritic(obs_dim, act_dim, config.HID_DIMS, config.ACTIV)
        self.critic2 = MLPQualityCritic(obs_dim, act_dim, config.HID_DIMS, config.ACTIV)
        self.actor_target = deepcopy(self.actor)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.actor_optimizer = AdamW(self.actor.parameters(), lr=config.LR)
        self.critic1_optimizer = AdamW(self.critic1.parameters(), lr=config.LR)
        self.critic2_optimizer = AdamW(self.critic2.parameters(), lr=config.LR)
        

    def compute_policy_loss(self, obs):
        self.critic1.requires_grad_(False)
        self.critic2.requires_grad_(False)
        q1 = self.critic1(obs, self.actor(obs))
        q2 = self.critic2(obs, self.actor(obs))
        loss = -th.mean((q1 + q2)/2)
        self.critic1.requires_grad_(True)
        self.critic2.requires_grad_(True)
        return loss


    def compute_quality_loss(self, obs, act, next_obs, rew, done):
        with th.no_grad():
            q1_target = self.critic1_target(next_obs, self.actor_target(next_obs))
            q2_target = self.critic2_target(next_obs, self.actor_target(next_obs))
            target = rew + config.GAMMA*(1-done)*th.minimum(q1_target, q2_target)
        loss1 = th.mean((self.critic1(obs, act) - target)**2)
        loss2 = th.mean((self.critic2(obs, act) - target)**2)
        return loss1, loss2
    

    def update_target(self, model, target):
        with th.no_grad():
            for p, p_target in zip(model.parameters(), target.parameters()):
                p_target.data.mul_(config.RHO)
                p_target.data.add_((1-config.RHO) * p.data)
    

    def update(self, timer):
        data = self.memory.sample_batch(config.BATCH_SIZE)
        obs, act, next_obs, rew, done = data['obs'], data['act'], data['next_obs'], data['rew'], data['done']

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        q_loss1, q_loss2 = self.compute_quality_loss(obs, act, next_obs, rew, done)
        q_loss1.backward()
        q_loss2.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        self.update_target(self.critic1, self.critic1_target)
        self.update_target(self.critic2, self.critic2_target)
        
        if timer % config.POLICY_DELAY == 0:
            self.actor_optimizer.zero_grad()
            policy_loss = self.compute_policy_loss(obs)
            policy_loss.backward()
            self.actor_optimizer.step()
        
            self.update_target(self.actor, self.actor_target)
    

    def train(self):
        print("============ TRAINING ============")
        epochs_digits = 1 + int(np.log10(config.EPOCHS))
        episode_ret, episode_len = 0, 0
        obs = self.env.reset()
        solved_flag = False
        tot_time = 0
        for epoch in range(1, 1+config.EPOCHS):
            start_time = time.time()
            epoch_returns = []
            for t in range(1, 1+config.STEPS_PER_EPOCH):
                if epoch > config.WARMUP_EPOCHS:
                    with th.no_grad():
                        obs_tensor = th.tensor(obs.reshape(1,-1), dtype=th.float32)
                        act = self.actor(obs_tensor).squeeze(0).numpy()
                else:
                    act = self.env.action_space.sample()
                
                next_obs, rew, done, _ = self.env.step(act)
                episode_ret += rew
                episode_len += 1
                timeout = episode_len == self.env._max_episode_steps
                done = False if timeout else done
                self.memory.store(obs, act, next_obs, rew, done)
                obs = next_obs

                if epoch > config.WARMUP_EPOCHS and t % config.UPDATE_EVERY == 0:
                    for i in range(1, 1+config.NUM_UPDATES):
                        self.update(timer=i)
                
                if timeout or done:
                    epoch_returns.append(episode_ret)
                    self.score.update(episode_ret)
                    obs, episode_ret, episode_len = self.env.reset(), 0, 0
            
            epoch_time = time.time() - start_time
            tot_time += epoch_time
            tot_time_min, tot_time_sec = map(lambda x: str(round(x)).zfill(2), divmod(tot_time, 60))
            epoch_returns = np.array(epoch_returns)
            print(
                f"Epoch: {epoch:{epochs_digits}}/{config.EPOCHS} - "
                f"Time: {epoch_time:4.1f}s/{tot_time_min}.{tot_time_sec}min - "
                f"MinRet: {epoch_returns.min():8.2f} - AvgRet: {epoch_returns.mean():8.2f} - "
                f"MaxRet: {epoch_returns.max():8.2f} - StdRet: {epoch_returns.std():8.2f}" 
            )
            if self.score.solved and not solved_flag:
                solved_flag = True
                print(
                    f"Environment solved in {self.score.episode-self.score.size} episodes with "
                    f"{self.score.best:.2f} average return over {self.score.size} consecutive episodes"
                )
        
        print(
            f"Best score is {self.score.best:.2f} average return over "
            f"{self.score.size} consecutive episodes after {self.score.episode} total episodes"
        )



    def test(self):
        if not config.TEST_EPISODES: return
        print("============ TESTING ============")
        test_episodes_digits = 1 + int(np.log10(config.TEST_EPISODES))
        for episode in range(1, 1+config.TEST_EPISODES):
            episode_ret, episode_len = 0, 0
            done = False
            obs = self.env.reset()
            while not done:
                self.env.render()
                with th.no_grad():
                    obs_tensor = th.tensor(obs.reshape(1,-1), dtype=th.float32)
                    act = self.actor(obs_tensor).squeeze(0)
                obs, rew, done, _ = self.env.step(act.numpy())
                episode_ret += rew
                episode_len += 1
            
            print(
                f"Episode: {episode:{test_episodes_digits}}/{config.TEST_EPISODES} - "
                f"Return: {episode_ret:8.2f} - Duration: {episode_len:4} steps"
            )
        self.env.close()



if __name__ == '__main__':
    learner = TD3()
    learner.train()
    learner.test()
