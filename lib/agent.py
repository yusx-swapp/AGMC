
import itertools

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from lib.memory import SequentialMemory
from lib.Utils import to_numpy, to_tensor
from models.encoder_gcn import GraphEncoder_GCN
import os


import torch as T,torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

criterion = nn.MSELoss()
USE_CUDA = torch.cuda.is_available()


class Actor(nn.Module):
    def __init__(self, node_feature_size,nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Actor, self).__init__()
        self.graph_encoder = GraphEncoder_GCN(node_feature_size, nb_states, nb_states)
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, g):
        out = self.graph_encoder(g)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, node_feature_size,nb_states, nb_actions, hidden1=400, hidden2=300):
        super(Critic, self).__init__()
        self.graph_encoder_critic = GraphEncoder_GCN(node_feature_size, nb_states, nb_states)

        self.fc11 = nn.Linear(nb_states, hidden1)
        self.fc12 = nn.Linear(nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, xs):
        x, a = xs
        g = self.graph_encoder_critic(x)
        out = self.fc11(g) + self.fc12(a)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class DDPG(object):
    def __init__(self, node_feature_size,nb_states, nb_actions, args):
        # for key, value in params.items():
        #     setattr(self, key, value)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden1,
            'hidden2': args.hidden2,
            # 'init_w': args.init_w
        }
        self.actor = Actor(node_feature_size,self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(node_feature_size,self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_a)

        self.critic = Critic(node_feature_size,self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(node_feature_size,self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_c)

        self.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        self.hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        # self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu,
        #                                                sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.lbound = 0.  # args.lbound
        self.rbound = 1.  # args.rbound

        # noise
        self.init_delta = args.init_delta
        self.delta_decay = args.delta_decay
        self.warmup = args.warmup

        #
        self.epsilon = 1.0
        # self.s_t = None  # Most recent state
        # self.a_t = None  # Most recent action
        self.is_training = True

        #
        if USE_CUDA: self.cuda()

        # moving average baseline
        self.moving_average = None
        self.moving_alpha = 0.5  # based on batch, so small

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # normalize the reward
        batch_mean_reward = np.mean(reward_batch)
        if self.moving_average is None:
            self.moving_average = batch_mean_reward
        else:
            self.moving_average += self.moving_alpha * (batch_mean_reward - self.moving_average)
        reward_batch -= self.moving_average
        # if reward_batch.std() > 0:
        #     reward_batch /= reward_batch.std()

        # Prepare for the target q batch
        with torch.no_grad():

            next_q_values = self.critic_target([
                next_state_batch,
                self.actor_target(next_state_batch),
            ])

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([state_batch, to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            state_batch,
            self.actor(state_batch)
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t, s_t1, a_t, done):
        if self.is_training:
            self.memory.append(s_t, a_t, r_t, done)  # save to memory
            # self.s_t = s_t1

    def random_action(self):
        # action = np.random.uniform(self.lbound, self.rbound, self.nb_actions)
        action = np.random.uniform(0, 0.3, self.nb_actions)
        # self.a_t = action
        return action

    def select_action(self, s_t, episode):
        # assert episode >= self.warmup, 'Episode: {} warmup: {}'.format(episode, self.warmup)
        # action = to_numpy(self.actor(to_tensor(np.array(s_t).reshape(1, -1)))).squeeze(0)
        action = to_numpy(self.actor(s_t)).squeeze(0)
        delta = self.init_delta * (self.delta_decay ** (episode - self.warmup))
        #action += self.is_training * max(self.epsilon, 0) * self.random_process.sample()
        # action=np.array(action)
        action = self.sample_from_truncated_normal_distribution(lower=self.lbound, upper=self.rbound, mu=np.mean(action), sigma=delta,size=action.shape[0])
        action = np.clip(action, self.lbound, self.rbound)

        # self.a_t = action
        return action

    def reset(self, obs):
        pass
        # self.s_t = obs
        # self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def sample_from_truncated_normal_distribution(self, lower, upper, mu, sigma, size=1):
        from scipy import stats
        return stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=size)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
class ActorNetwork(nn.Module):
    def __init__(self, node_feature_size,nb_states, nb_actions, hidden1=400, hidden2=300,
                 chkpt_dir='tmp/rl'):
        super(ActorNetwork, self).__init__()

        self.graph_encoder = GraphEncoder_GCN(node_feature_size, nb_states, nb_states)
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        out = self.graph_encoder(g)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        actions = self.tanh(out)

        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self,node_feature_size,nb_states, nb_actions, hidden1=400, hidden2=300,
                 chkpt_dir='tmp/rl'):
        super(CriticNetwork, self).__init__()

        self.graph_encoder_critic = GraphEncoder_GCN(node_feature_size, nb_states, nb_states)

        self.fc11 = nn.Linear(nb_states, hidden1)
        self.fc12 = nn.Linear(nb_actions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()

        # self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, xs):
        x, a = xs
        g = self.graph_encoder_critic(x)
        out = self.fc11(g) + self.fc12(a)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        actor_cfg = {
            'g_in_size':state_dim,
            'g_hidden_size':50,
            'g_embedding_size':50,
            'hidden_size':200,
            'nb_actions':action_dim,

        }
        critic_cfg = {

            'g_in_size':state_dim,
            'g_hidden_size':50,
            'g_embedding_size':50,
        }
        self.actor = ActorNetwork(**actor_cfg)
        self.critic = CriticNetwork(**critic_cfg)

        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class Agent:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.policy.parameters()), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_states = memory.states
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()


        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            print('Epoches {} \t loss: {} \t '.format(_, loss.mean()))
            # start = time.time()
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # end = time.time()
            # times.append(end-start)
        # print("time:",times)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
