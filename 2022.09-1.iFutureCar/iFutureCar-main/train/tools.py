# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import collections
import math
import os
from typing import Dict

import numpy
import torch
import torch as ch
import torch.nn as nn

from cvxpy import vec
from train.config import PolicyParam
from train.np_image import NPImage

STD = 2 ** 0.5


def file_name(file_dir, file_type):
    L = []
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == file_type:
                L.append(os.path.join(dirpath, file))
    return L


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = numpy.zeros(shape)
        self.S = numpy.zeros(shape)
        self.std = numpy.sqrt(self.S)

    def update(self, x):
        x = numpy.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = numpy.sqrt(self.S / self.n)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = numpy.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = numpy.zeros(self.shape)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


def initialize_weights(mod, initialization_type, scale=STD):
    """
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    """
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")


def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    u, s, v = ch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with ch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


class TianchiCNN(object):
    def __init__(
        self,
    ):
        self.base_image_size = 500
        self.crop_size = 224
        self.resize_size = 224
        self.base_image_reso = 1.0
        self.draw_count = 0

    def draw_from_obs(self, obs, center_dict=None):

        undrivable_img = NPImage(
            meter_per_pixel=self.base_image_reso,
            width=self.base_image_size,
            height=self.base_image_size,
        )

        drivable_img = NPImage(
            meter_per_pixel=self.base_image_reso,
            width=self.base_image_size,
            height=self.base_image_size,
        )

        obs_img = NPImage(
            meter_per_pixel=self.base_image_reso,
            width=self.base_image_size,
            height=self.base_image_size,
        )

        center_pose_x = obs["player"]["status"][0]
        center_pose_y = obs["player"]["status"][1]
        center_pose_h = obs["player"]["status"][2]

        ego_width = obs["player"]["property"][0]
        ego_length = obs["player"]["property"][1]

        undrivable_img.set_center_pose((center_pose_x, center_pose_y))
        drivable_img.set_center_pose((center_pose_x, center_pose_y))
        obs_img.set_center_pose((center_pose_x, center_pose_y))

        obs_img.draw_rect((center_pose_x, center_pose_y, center_pose_h), ego_length, ego_width, 255)

        for npc in obs["npcs"]:
            obs_img.draw_rect((npc[2], npc[3], npc[4]), npc[10], npc[9], 128)

        if center_dict:
            for centers in center_dict.values():
                drivable_img.draw_polyline(centers, 128)

        merge_img = undrivable_img.merge([drivable_img, obs_img])

        heading_angle = 90 - center_pose_h * 180.0 / math.pi

        merge_img.resize(self.resize_size).rotate(
            (self.resize_size / 2, self.resize_size / 2), heading_angle, self.resize_size
        ).flip()

        return merge_img.img_data


class EnvPostProcsser:
    def __init__(self) -> None:
        self.args = PolicyParam

        self.target_speed = self.args.target_speed
        self.dt = self.args.dt
        self.history_length = self.args.history_length
        self.img_width = self.args.img_width
        self.img_length = self.args.img_length
        self.vec_length = self.args.ego_vec_length
        self.surr_vec_length = self.args.surr_vec_length
        self.surr_number = self.args.surr_agent_number
        self.obs_type = self.args.obs_type

        self.prev_distance = None
        self.surr_cnn_normalize = Normalization(shape=(self.img_width, self.img_length, 3))
        self.vec_normalize = Normalization(shape=self.vec_length)
        self.surr_vec_normalize = Normalization(shape=(1, self.surr_number * 7))
        self.reward_scale = RewardScaling(shape=1, gamma=self.args.gamma)
        self.surr_img_deque = collections.deque(maxlen=5)
        self.surr_vec_deque = collections.deque(maxlen=5)
        self.vec_deque = collections.deque(maxlen=5)
        
        for i in range(self.history_length):
            self.surr_img_deque.append(numpy.zeros((self.img_width, self.img_length, 3)))
            self.surr_vec_deque.append(numpy.zeros((1, self.surr_number * 7)))
            self.vec_deque.append(numpy.zeros(self.vec_length))
        self.tianchi_cnn = TianchiCNN()  # not used

    def assemble_surr_cnn_obs(self, observation, env):
        """
        not used
        """
        centers = {}
        for lane in observation["map"].lanes:
            lane_id = lane.lane_id
            centers[lane_id] = env.centers_by_lane_id(lane_id)
            
        img_obs = self.tianchi_cnn.draw_from_obs(observation, centers).astype(numpy.float32)
        img_obs = self.surr_cnn_normalize(img_obs)
        self.surr_img_deque.append(img_obs)
        cnn_obs = numpy.concatenate(list(self.img_deque), axis=2)
        env_state = torch.Tensor(cnn_obs).float().unsqueeze(0).permute(0, 3, 2, 1)
        return env_state

    def assemble_surr_vec_obs(self, observation):
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])
        npc_info_dict = {}
        
        for npc_info in observation["npcs"]:
            if int(npc_info[0]) == 0:
                continue
            npc_info_dict[
                numpy.sqrt((npc_info[2] - curr_xy[0]) ** 2 + (npc_info[3] - curr_xy[1]) ** 2)
            ] = [
                npc_info[2] - curr_xy[0],
                npc_info[3] - curr_xy[1],
                npc_info[4],
                numpy.sqrt(npc_info[5] ** 2 + npc_info[6] ** 2),
                numpy.sqrt(npc_info[7] ** 2 + npc_info[8] ** 2),
                npc_info[9],
                npc_info[10],
            ]
        sorted_npc_info_dict = dict(sorted(npc_info_dict.items(), key=lambda x: x[0]))
        surr_obs_list = list(sorted_npc_info_dict.values())
        for _ in range(self.surr_number - len(surr_obs_list)):
            surr_obs_list.append(list(numpy.zeros(self.surr_vec_length)))
            
        curr_surr_obs = numpy.array(surr_obs_list).reshape(-1)[: self.surr_number * 7]
        curr_surr_obs = self.surr_vec_normalize(curr_surr_obs)
        self.surr_vec_deque.append(curr_surr_obs.reshape(1, -1))
        surr_vec_obs = numpy.concatenate(list(self.surr_vec_deque), axis=0)[numpy.newaxis, :, :]
        env_state = torch.Tensor(surr_vec_obs).float().unsqueeze(0)
        return env_state

    def assemble_ego_vec_obs(self, observation):
        target_xy = (
            (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,
            (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,
        )
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])
        delta_xy = (target_xy[0] - curr_xy[0], target_xy[1] - curr_xy[1])
        curr_yaw = observation["player"]["status"][2]
        curr_velocity = observation["player"]["status"][3]
        prev_steer = observation["player"]["status"][7]
        prev_acc = observation["player"]["status"][8]
        lane_list = []
        
        for lane_info in observation["map"].lanes:
            lane_list.append(lane_info.lane_id)
        current_lane_index = lane_list.index(observation["map"].lane_id)
        current_offset = observation["map"].lane_offset
        vec_obs = numpy.array(
            [
                delta_xy[0],
                delta_xy[1],
                curr_yaw,
                curr_velocity,
                prev_steer,
                prev_acc,
                current_lane_index,
                current_offset,
            ]
        )
        vec_obs = self.vec_normalize(vec_obs)
        self.vec_deque.append(vec_obs)
        mlp_obs = numpy.concatenate(list(self.vec_deque), axis=0)
        vec_state = torch.Tensor(mlp_obs).float().unsqueeze(0)
        return vec_state

    def assemble_reward(self, observation: Dict, info: Dict) -> float:
        target_xy = (
            (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,
            (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,
        )
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])
        distance_with_target = numpy.sqrt(
            (target_xy[0] - curr_xy[0]) ** 2 + (target_xy[1] - curr_xy[1]) ** 2
        )
        if self.prev_distance is None:
            self.prev_distance = distance_with_target
        distance_reward = (self.prev_distance - distance_with_target) / (
            self.target_speed * self.dt
        )
        self.prev_distance = distance_with_target
        step_reward = -0.5

        if info["collided"]:
            end_reward = -200
        elif info["reached_stoparea"]:
            end_reward = 200
        elif info["timeout"]:
            end_reward = -200
        else:
            end_reward = 0.0
        return distance_reward + end_reward + step_reward

    def assemble_surr_obs(self, observation, env):
        if self.obs_type is "cnn":
            return self.assemble_surr_cnn_obs(observation=observation, env=env)
        elif self.obs_type is "vec":
            return self.assemble_surr_vec_obs(observation=observation)
        else:
            raise Exception("error observation type")

    def reset(self):
        self.prev_distance = None
        self.reward_scale.reset()
        self.img_deque = collections.deque(maxlen=5)
        self.vec_deque = collections.deque(maxlen=5)
        
        for _ in range(self.history_length):
            self.img_deque.append(numpy.zeros((self.img_width, self.img_length, 3)))
            self.vec_deque.append(numpy.zeros(self.vec_length))
