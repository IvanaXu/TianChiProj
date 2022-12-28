# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

# from tensorboardX import SummaryWriter
from train.config import PolicyParam
from train.policy import PPOPolicy
from train.workers import MemorySampler
from geek.env.logger import Logger

logger = Logger.get_logger(__name__)

class MulProPPO:
    def __init__(self, obs_type, logger) -> None:
        self.args = PolicyParam
        self.obs_type = obs_type
        self.logger = logger
        self.global_sample_size = 0
        self._make_dir()

        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.device == "cuda":
            torch.cuda.manual_seed(self.args.seed)

        self.sampler = MemorySampler(self.args, self.logger)
        self.model = PPOPolicy(2)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.args.lr)

        self.clip_now = self.args.clip
        self.start_episode = 0
        self._load_model(self.args.model_path)

    def _load_model(self, model_path: str = None):
        if not model_path:
            return
        pretrained_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage.cuda(self.args.device)
        )
        if self._check_keys(self.model, pretrained_dict):
            self.model.load_state_dict(pretrained_dict, strict=False)

    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # filter 'num_batches_tracked'
        missing_keys = [x for x in missing_keys if not x.endswith("num_batches_tracked")]
        if len(missing_keys) > 0:
            logger.info("[Warning] missing keys: {}".format(missing_keys))
            logger.info("missing keys:{}".format(len(missing_keys)))
        if len(unused_pretrained_keys) > 0:
            logger.info("[Warning] unused_pretrained_keys: {}".format(unused_pretrained_keys))
            logger.info("unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
        logger.info("used keys:{}".format(len(used_pretrained_keys)))

        assert len(used_pretrained_keys) > 0, "check_key load NONE from pretrained checkpoint"
        return True

    def _make_dir(self):
        # current_dir = os.path.abspath(".")
        current_dir = "/myspace"

        self.exp_dir = current_dir + "/results/exp_{}/".format("10086") # time.time()
        self.model_dir = current_dir + "/results/model_{}/".format("10086") # time.time()
        try:
            os.makedirs(self.exp_dir)
            os.makedirs(self.model_dir)
        except:
            print("file is existed")
        # self.writer = SummaryWriter(self.exp_dir)

    def train(self):
        for i_episode in range(self.args.num_episode):
            self.logger.info(
                "----------------------" + str(i_episode) + "-------------------------"
            )
            memory = self.sampler.sample(self.model)
            batch = memory.sample()
            batch_size = len(memory)
            self.global_sample_size += batch_size

            rewards = torch.from_numpy(np.array(batch.reward))
            values = torch.from_numpy(np.array(batch.value))
            masks = torch.from_numpy(np.array(batch.mask))
            actions = torch.from_numpy(np.array(batch.action))
            env_state = torch.from_numpy(np.array(batch.env_state))
            vec_state = torch.from_numpy(np.array(batch.vec_state))
            oldlogproba = torch.from_numpy(np.array(batch.logproba))

            returns = torch.Tensor(batch_size)
            deltas = torch.Tensor(batch_size)
            advantages = torch.Tensor(batch_size)
            prev_return = 0
            prev_value = 0
            prev_advantage = 0

            for i in reversed(range(batch_size)):
                returns[i] = rewards[i] + self.args.gamma * prev_return * masks[i]
                deltas[i] = rewards[i] + self.args.gamma * prev_value * masks[i] - values[i]
                advantages[i] = (
                    deltas[i] + self.args.gamma * self.args.lamda * prev_advantage * masks[i]
                )

                prev_return = returns[i]
                prev_value = values[i]
                prev_advantage = advantages[i]
            if self.args.advantage_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + self.args.EPS)

            env_state = env_state.to(self.args.device)
            values = values.to(self.args.device)
            vec_state = vec_state.to(self.args.device)
            actions = actions.to(self.args.device)
            oldlogproba = oldlogproba.to(self.args.device)
            advantages = advantages.to(self.args.device)
            returns = returns.to(self.args.device)
            for i_epoch in range(int(self.args.num_epoch * batch_size / self.args.minibatch_size)):
                minibatch_ind = np.random.choice(
                    batch_size, self.args.minibatch_size, replace=False
                )
                minibatch_env_state = env_state[minibatch_ind]
                minibatch_vec_state = vec_state[minibatch_ind]
                minibatch_actions = actions[minibatch_ind]
                minibatch_values = values[minibatch_ind]
                minibatch_oldlogproba = oldlogproba[minibatch_ind]
                minibatch_newlogproba, entropy = self.model.get_logproba(
                    minibatch_env_state, minibatch_vec_state, minibatch_actions
                )
                minibatch_advantages = advantages[minibatch_ind]
                minibatch_returns = returns[minibatch_ind]
                minibatch_newvalues = self.model._forward_critic(
                    minibatch_env_state, minibatch_vec_state
                ).flatten()

                assert minibatch_oldlogproba.shape == minibatch_newlogproba.shape
                ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
                assert ratio.shape == minibatch_advantages.shape
                surr1 = ratio * minibatch_advantages
                surr2 = ratio.clamp(1 - self.clip_now, 1 + self.clip_now) * minibatch_advantages
                loss_surr = -torch.mean(torch.min(surr1, surr2))

                if self.args.use_clipped_value_loss:
                    value_pred_clipped = minibatch_values + (
                        minibatch_newvalues - minibatch_values
                    ).clamp(-self.args.vf_clip_param, self.args.vf_clip_param)
                    value_losses = (minibatch_newvalues - minibatch_returns).pow(2)
                    value_loss_clip = (value_pred_clipped - minibatch_returns).pow(2)
                    loss_value = torch.max(value_losses, value_loss_clip).mean()
                else:
                    loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

                if self.args.lossvalue_norm:
                    minibatch_return_6std = 6 * minibatch_returns.std()
                    loss_value = (
                        torch.mean((minibatch_newvalues - minibatch_returns).pow(2))
                        / minibatch_return_6std
                    )
                else:
                    loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

                loss_entropy = -torch.mean(entropy)

                total_loss = (
                    loss_surr
                    + self.args.loss_coeff_value * loss_value
                    + self.args.loss_coeff_entropy * loss_entropy
                )
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.schedule_clip == "linear":
                ep_ratio = 1 - ((i_episode) / self.args.num_episode)
                self.clip_now = self.args.clip * ep_ratio

            if self.args.schedule_adam == "linear":
                ep_ratio = 1 - ((i_episode) / self.args.num_episode)
                lr_now = self.args.lr * ep_ratio
                for g in self.optimizer.param_groups:
                    g["lr"] = lr_now
                iteration_reduce = self.args.lr * (1 - ep_ratio)

            if self.args.schedule_adam == "layer":
                for item in self.args.lr_schedule:
                    if self.global_sample_size >= item[0]:
                        lr_now = item[1]
                for g in self.optimizer.param_groups:
                    g["lr"] = lr_now
                iteration_reduce = 0.0

            if self.args.schedule_adam == "layer_linear":
                for idx, item in enumerate(self.args.lr_schedule):
                    if self.global_sample_size >= item[0]:
                        lr_max = item[1]
                        data_num_min = item[0]
                        lr_min = self.args.lr_schedule[idx + 1][1]
                        data_num_max = self.args.lr_schedule[idx + 1][0]
                num_iteration = int((data_num_max - data_num_min) / self.args.batch_size)
                iteration_reduce = float((lr_max - lr_min) / num_iteration)
                self.args.lr = self.args.lr - iteration_reduce
                lr_now = self.args.lr
                for g in self.optimizer.param_groups:
                    g["lr"] = lr_now

            if self.args.schedule_adam == "fix":
                lr_now = self.args.lr
                iteration_reduce = 0.0

            if i_episode % self.args.log_num_episode == 0:
                mean_reward = (torch.sum(rewards) / memory.num_episode).data
                mean_step = len(memory) // memory.num_episode
                reach_goal_rate = memory.arrive_goal_num / memory.num_episode

                reward = mean_reward.cpu().data.item()
                total_loss = total_loss.cpu().data.item()
                loss_surr = loss_surr.cpu().data.item()
                loss_value = loss_value.cpu().data.item()
                loss_entropy = loss_entropy.cpu().data.item()
                self.logger.info("Finished iteration: " + str(i_episode))
                self.logger.info("reach goal rate: " + str(reach_goal_rate))
                self.logger.info("reward: " + str(reward))
                self.logger.info(
                    "total loss: "
                    + str(total_loss)
                    + " = "
                    + str(loss_surr)
                    + "+"
                    + str(self.args.loss_coeff_value)
                    + "*"
                    + str(loss_value)
                    + "+"
                    + str(self.args.loss_coeff_entropy)
                    + "*"
                    + str(loss_entropy)
                )
                self.logger.info("Step: " + str(mean_step))
                self.logger.info("total data number: " + str(self.global_sample_size))
                self.logger.info(
                    "lr now: " + str(lr_now) + "  lr reduce per iteration: " + str(iteration_reduce)
                )
                # self.writer.add_scalar("reward", reward, i_episode)
                # self.writer.add_scalar("total_loss", total_loss, i_episode)
                # self.writer.add_scalar("reach_goal_rate", reach_goal_rate, i_episode)
            if i_episode % self.args.save_num_episode == 0:
                torch.save(
                    self.model.state_dict(), self.model_dir + "network_{}.pth".format(i_episode)
                )
        self.sampler.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    
    obs_type = "cnn"
    mpp = MulProPPO(obs_type=obs_type, logger=logger)
    mpp.train()
