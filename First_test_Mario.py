#!/usr/bin/env python
# coding: utf-8




import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from atari_wrappers import wrap_environment
from train_info import TrainInformation
from os.path import join





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor



Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')

)


class ReplayMemory():
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, num_of_eps_to_final_value=1000000):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.beta_start = beta_start
        self.final = num_of_eps_to_final_value
        self.beta = beta_start
    
    def push(self, experience):
        max_prio = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
            
            
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, device=device,dtype=torch.float32)

            
        return samples, weights, indices
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def update_beta(self,timestep):
        self.beta = min(self.beta_start + timestep * (1.0 - self.beta_start) / self.final,1.0)





class DQN(nn.Module):

    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self._input_shape = input_shape
        self._num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.vaue_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        
    def forward(self, x):
        x = self.features(x)
        value = self.vaue_stream(x)
        advantage = self.advantage_stream(x)
        q_vals = value + (advantage - advantage.mean())
        return q_vals
    
    @property
    def feature_size(self):
        x = self.features(torch.zeros(1, *self._input_shape))
        return x.view(1, -1).size(1)





class EpsilonGreedyStrategy():
    def __init__(self, start, end,num_of_steps ):
        self.start = start
        self.end = end
        self.num_of_steps = num_of_steps
    
    def get_exploration_rate(self, current_step):
        return max(self.start - (current_step * (self.start-self.end) / self.num_of_steps),self.end)





class Agent():
    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
    
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return torch.tensor([random.randrange(self.num_actions)],device=device) 
            # explore      
        else:
            with torch.no_grad():
                return torch.tensor([policy_net(state).argmax(dim=1).item()],device=device) 
                # exploit  





class MarioEnvManager():
    def __init__(self,max_episode_steps):
        self.env = wrap_environment(gym_super_mario_bros.make('SuperMarioBros-v0'),RIGHT_ONLY)
        self.done = False
    
    def reset(self):
        return (torch.tensor(self.env.reset(),device=device,dtype=torch.float32)).unsqueeze(0)

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n
    
    def take_action(self, action):        
        next_state, reward, self.done, info = self.env.step(action.item())
        return ((torch.tensor(next_state,device=device,dtype=torch.float32)).unsqueeze(0)                 ,torch.tensor([reward], device=device,dtype=torch.float32),info)
        


    





class QValues():
    
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
    
    @staticmethod        
    def get_next(target_net, next_states): 
        q_actions = policy_net(next_states).argmax(dim=1).detach()
        return target_net(next_states).gather(dim=1,index = q_actions.unsqueeze(-1)).squeeze(-1).detach()





def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)





batch_size = 32
gamma = 0.90
eps_start = 1
eps_end = 0.01
num_of_steps_epsilon = 1000000
beta_start = 0.4
target_update = 10000
memory_size = 40000
lr = 0.00025
num_episodes = 50000
max_episode_steps = 10000
initial_learning_step = 10000





#---------------- Initiallization
em = MarioEnvManager(max_episode_steps)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, num_of_steps_epsilon)
agent = Agent(strategy, em.num_actions_available())
memory = ReplayMemory(memory_size)

policy_net = DQN((4,84,84),em.num_actions_available()).to(device)
target_net = DQN((4,84,84),em.num_actions_available()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

information = TrainInformation()



#---------------- Training loop
for episode in range(num_episodes):
    episode_reward = 0.0
    state = em.reset()
    print("episode: ",episode)
    
    
    while True:
        information.update_index()
        action = agent.select_action(state, policy_net)
        next_state ,reward,info = em.take_action(action)
        episode_reward += reward.item()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state
    
        if memory.can_provide_sample(batch_size):

            
            
            if len(memory.memory) > initial_learning_step:
                memory.update_beta(information.index)
                if information.index % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            
                optimizer.zero_grad()
                experiences, weights, indices = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma)*(1-em.done) + rewards


                loss = (current_q_values - target_q_values).pow(2) * weights
                prios = (current_q_values - target_q_values).abs() + 1e-5
                loss = loss.mean()
                loss.backward()
                memory.update_priorities(indices, prios.data.cpu().numpy())
                optimizer.step()
        
        if em.done: 
            if info['flag_get']:
                torch.save(policy_net.state_dict(),join("model_save",'SuperMarioBros-v0-gotflag.dat'))
            break
        

    
    if(information.update_rewards(episode_reward)):
        torch.save(policy_net.state_dict(),join("model_save",'SuperMarioBros-v0.dat'))
    
    print('Episode %s - Reward: %s, Best: %s, Average: %s '
          'Epsilon: %s Beta = %s' % (episode,
                           round(episode_reward, 3),
                           round(information.best_reward, 3),
                           round(information.average, 3),
                           round(strategy.get_exploration_rate(information.index), 4),
                           round(memory.beta, 4)))
    
    

       


    
em.close()





