from argparse import Namespace
import time
import gym
import wandb
import numpy as np
import torch as th
from torch import nn
from torch.optim import AdamW
from memory import PolicyMemory
from actor_critic import MLPCategoricalActor, MLPValueCritic
from utils import Score


LunarLanderConfig = Namespace(
    ENV_NAME = "LunarLander-v2",
    HID_DIMS = [128,128],
    ACTIV = nn.ReLU,
    EPOCHS = 50,
    STEPS_PER_EPOCH = 8_000,
    OPTIM_STEPS = 100,
    GAMMA = 0.99,
    LAMBDA = 0.99,
    LR = 0.001,
    CLIP_EPS = 0.2,
    TARGET_KL = 0.02,

    SCORE_TO_BEAT = 200,
    SCORE_SIZE = 100,

    TEST_EPISODES = 5)

CartPoleConfig = Namespace(
    ENV_NAME = "CartPole-v0",
    HID_DIMS = [64,64],
    ACTIV = nn.ReLU,
    EPOCHS = 20,
    STEPS_PER_EPOCH = 4_000,
    OPTIM_STEPS = 100,
    GAMMA = 0.999,
    LAMBDA = 0.99,
    LR = 0.001,
    CLIP_EPS = 0.2,
    TARGET_KL = 0.02,

    SCORE_TO_BEAT = 195,
    SCORE_SIZE = 100,

    TEST_EPISODES = 1,
    MAX_EPISODE_STEPS = 10_000)



config = LunarLanderConfig
# config = CartPoleConfig



class PPO:
    """
    Proximal Policy Optimization.

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
    

    def compute_policy_loss(self, obs, act, adv, logp_old):
        policy = self.actor(obs)
        logp = policy.log_prob(act)
        ratio = th.exp(logp - logp_old)
        clip_ratio = th.clamp(ratio, 1-config.CLIP_EPS, 1+config.CLIP_EPS)
        loss = -th.mean(th.min(ratio * adv, clip_ratio * adv))
        kl = th.mean(logp_old - logp.detach()).item()
        return loss, kl


    def compute_value_loss(self, obs, ret):
        return th.mean((self.critic(obs) - ret)**2)
    

    def update(self):
        data = self.memory.get()
        obs, act, ret, adv, old_logp = data['obs'], data['act'], data['ret'], data['adv'], data['logp']
        self.actor.train()
        self.critic.train()

        for _ in range(config.OPTIM_STEPS):
            self.actor_optimzer.zero_grad()
            policy_loss, kl = self.compute_policy_loss(obs, act, adv, old_logp)
            if kl > config.TARGET_KL: break
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
        epochs_num_digits = 1 + int(np.log10(config.EPOCHS))
        solved_flag = False
        tot_time = 0
        for epoch in range(config.EPOCHS):
            start_time = time.time()
            epoch_returns = []
            episode_ret, episode_len = 0, 0
            obs = self.env.reset()
            for t in range(config.STEPS_PER_EPOCH):
                with th.no_grad():
                    obs_tensor = th.tensor(obs.reshape(1,-1), dtype=th.float32)
                    policy = self.actor(obs_tensor)
                    act = policy.sample()
                    logp = policy.log_prob(act)
                    val = self.critic(obs_tensor)
                next_obs, rew, done, _ = self.env.step(act.squeeze(0).numpy())
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
                    obs, episode_ret, episode_len = self.env.reset(), 0, 0
            
            self.update()
            epoch_time = time.time() - start_time
            tot_time += epoch_time
            tot_time_min, tot_time_sec = map(lambda s: str(round(s)).zfill(2), divmod(tot_time, 60))

            epoch_returns = np.array(epoch_returns)
            stats = {
                'Min Return': epoch_returns.min(),
                'Avg Return': epoch_returns.mean(),
                'Max Return': epoch_returns.max(),
                'Std Return': epoch_returns.std()}
            wandb.log(stats, step=epoch*config.STEPS_PER_EPOCH)
            print(
                f"Epoch: {epoch+1:{epochs_num_digits}}/{config.EPOCHS} - "
                f"Time: {epoch_time:4.1f}s/{tot_time_min}.{tot_time_sec}min - "
                f"MinRet: {epoch_returns.min():8.2f} - AvgRet: {epoch_returns.mean():8.2f} - "
                f"MaxRet: {epoch_returns.max():8.2f} - StdRet: {epoch_returns.std():8.2f}")
            if self.score.solved and not solved_flag:
                solved_flag = True
                print(
                    f"Environment solved in {self.score.episode-self.score.size} episodes with "
                    f"{self.score.best:.2f} average return over {self.score.size} consecutive episodes")
        
        print(
            f"Best score is {self.score.best:.2f} average return over "
            f"{self.score.size} consecutive episodes after {self.score.episode} total episodes")


    def test(self):
        if not config.TEST_EPISODES: return
        print("============ TESTING ============")
        test_episodes_num_digits = 1 + int(np.log10(config.TEST_EPISODES))
        frames = []
        for episode in range(1, 1+config.TEST_EPISODES):
            episode_ret, episode_len = 0, 0
            done = False
            obs = self.env.reset()
            while not done:
                frame = self.env.render(mode='rgb_array')
                frames.append(frame)
                with th.no_grad():
                    obs_tensor = th.tensor(obs.reshape(1,-1), dtype=th.float32)
                    policy = self.actor(obs_tensor)
                    act = policy.sample().squeeze()
                obs, rew, done, _ = self.env.step(act.numpy())
                episode_ret += rew
                episode_len += 1
            
            print(
                f"Episode: {episode:{test_episodes_num_digits}}/{config.TEST_EPISODES} - "
                f"Return: {episode_ret:8.2f} - Duration: {episode_len:4} steps")
        self.env.close()
        
        frames = np.stack(frames)
        frames = np.transpose(frames, (0,3,1,2))
        wandb.log({'test': wandb.Video(frames, fps=60, format='mp4')})



def main():
    wandb.init(
        project='deep-reinforcement-learning',
        config=vars(config),
        group=config.ENV_NAME,
        tags=['PPO'])
    
    learner = PPO()
    learner.train()
    learner.test()



if __name__ == '__main__':
    main()
