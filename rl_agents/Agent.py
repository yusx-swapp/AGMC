import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_agents.Actor import Actor
from rl_agents.Critic import Critic
import random
import itertools

import numpy as np
class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.s_dim
        a_dim = self.a_dim

        self.actor = Actor(s_dim, 256, a_dim).to(self.device)
        self.actor_target = Actor(s_dim, 256, a_dim).to(self.device)
        self.critic = Critic(s_dim + a_dim, 256, a_dim).to(self.device)
        self.critic_target = Critic(s_dim + a_dim, 256, a_dim).to(self.device)
        self.actor_optim = optim.Adam(itertools.chain(self.actor.parameters(),self.env.parameters(),self.graph_encoder.parameters()), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self, s0):
        #s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0).to(self.device)
        s0 = s0.float().unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().cpu().numpy()
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor([item.cpu().detach().numpy() for item in s0]).view(self.batch_size, -1).to(self.device)
        #s0 = torch.FloatTensor(list(s0))
        a0 = torch.FloatTensor(a0).view(self.batch_size, -1).to(self.device)
        #print("r1",r1)
        r1=torch.tensor(r1, dtype=torch.float).to(self.device)
        r1 = r1.view(self.batch_size, -1).to(self.device)
        s1 = torch.tensor([item.cpu().detach().numpy() for item in s1]).view(self.batch_size, -1).to(self.device)
        def critic_learn():
            #print("s1",s1.shape)
            a1 = self.actor_target(s1).detach()

            #print("a1",a1.shape)
            #print("s1",s1.shape)
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

