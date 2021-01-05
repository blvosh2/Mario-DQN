#!/usr/bin/env python
# coding: utf-8




import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
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
from time import sleep





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





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





def take_action( action,env):        
    next_state, reward, done, _ = env.step(action.item())
    return ((torch.tensor(next_state,dtype=torch.float32)).unsqueeze(0)             ,torch.tensor([reward],dtype=torch.float32),done)




env = wrap_environment(gym_super_mario_bros.make('SuperMarioBros-v0'),RIGHT_ONLY)
policy_net = DQN((4,84, 84),env.action_space.n)
policy_net.load_state_dict(torch.load("/home/barak/Git_stuff/model_save/SuperMarioBros-v0-gotflag2.dat"))
policy_net.eval()





state = (torch.tensor(env.reset(),dtype=torch.float32)).unsqueeze(0)
for timestep in count():
        action = torch.tensor([policy_net(state).argmax(dim=1).item()])
        next_state ,reward,done = take_action(action, env)
        state = next_state
        sleep(0.042)
        env.render()

        
        if done:
            break
        
env.close()


