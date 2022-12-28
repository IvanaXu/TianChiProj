# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.config import PolicyParam
from train.tools import initialize_weights


class CnnFeatureNet(nn.Module):
    def __init__(self):
        super(CnnFeatureNet, self).__init__()
        self.act_cnn1 = nn.Conv2d(15, 16, 3)
        self.act_cv1_pool = nn.MaxPool2d(3, stride=2)
        self.act_cnn2 = nn.Conv2d(16, 16, 3)
        self.act_cv2_pool = nn.MaxPool2d(3, stride=2)
        self.act_cnn3 = nn.Conv2d(16, 16, 3)
        self.act_cv3_pool = nn.MaxPool2d(3, stride=2)
        self.act_cnn4 = nn.Conv2d(16, 16, 3)
        self.act_cv4_pool = nn.MaxPool2d(3, stride=2)
        self.act_cnn5 = nn.Conv2d(16, 16, 3)
        self.out_size = 1296

    def forward(self, img_state):
        mt_tmp = F.relu(self.act_cnn1(img_state))
        mt_tmp = self.act_cv1_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn2(mt_tmp))
        mt_tmp = self.act_cv2_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn3(mt_tmp))
        mt_tmp = self.act_cv3_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn4(mt_tmp))
        mt_tmp = self.act_cv4_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn5(mt_tmp))
        mt_tmp = torch.flatten(mt_tmp, 1)
        return mt_tmp


class VecFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_cnn1 = nn.Conv2d(1, 1, (5, 1))
        self.out_size = 70

    def forward(self, vec_state):
        vec_state = F.relu(self.act_cnn1(vec_state))
        vec_state = torch.flatten(vec_state, 1)
        return vec_state


class PPOPolicy(nn.Module):
    def __init__(self, num_outputs):
        nn.Module.__init__(self)
        self.policy_param = PolicyParam
        self.feature_net = (
            VecFeatureNet() if self.policy_param.obs_type is "vec" else CnnFeatureNet()
        )
        self.actor_logit = nn.Sequential(
            OrderedDict(
                [
                    ("actor_1", nn.Linear(self.feature_net.out_size + 40, 64)),
                    ("actor_relu_1", nn.ReLU()),
                    ("actor_2", nn.Linear(64, 64)),
                    ("actor_relu_2", nn.ReLU()),
                    ("actor_3", nn.Linear(64, 64)),
                    ("actor_relu_3", nn.ReLU()),
                ]
            )
        )

        self.critic_value = nn.Sequential(
            OrderedDict(
                [
                    ("critic_1", nn.Linear(self.feature_net.out_size + 40, 64)),
                    ("critic_relu_1", nn.ReLU()),
                    ("critic_2", nn.Linear(64, 64)),
                    ("critic_relu2", nn.ReLU()),
                    ("critic_3", nn.Linear(64, 64)),
                    ("critic_relu_3", nn.ReLU()),
                    ("critic_output", nn.Linear(64, 1)),
                ]
            )
        )

        self.logit = nn.Sequential(nn.Linear(64, num_outputs * 2), nn.Tanh())

        initialize_weights(self, "orthogonal")

    def _forward_actor(self, env_obs, vec_obs):
        env_feature = self.feature_net(env_obs)

        env_feature = env_feature.reshape(env_obs.shape[0], -1)
        complex_feature = torch.cat((env_feature, vec_obs.reshape(env_feature.shape[0], -1)), 1)

        logits = self.actor_logit(complex_feature)
        action_logit = self.logit(logits)
        action_mean, action_logstd = torch.chunk(action_logit, 2, dim=-1)
        return action_mean, action_logstd

    def _forward_critic(self, env_obs, vec_obs):
        env_feature = self.feature_net(env_obs)
        env_feature = env_feature.reshape(env_obs.shape[0], -1)
        complex_feature = torch.cat((env_feature, vec_obs.reshape(env_feature.shape[0], -1)), 1)
        values = self.critic_value(complex_feature)
        return values

    def forward(self, env_obs, vec_obs):
        action_mean, action_logstd = self._forward_actor(env_obs, vec_obs)
        critic_value = self._forward_critic(env_obs, vec_obs)
        return action_mean, action_logstd, critic_value

    def select_action(self, action_mean, action_logstd):
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = -0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, env_obs, vec_obs, actions):
        action_mean, action_logstd = self._forward_actor(env_obs, vec_obs)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        entropy = torch.distributions.Normal(action_mean, torch.exp(action_logstd)).entropy()
        return logproba, entropy
