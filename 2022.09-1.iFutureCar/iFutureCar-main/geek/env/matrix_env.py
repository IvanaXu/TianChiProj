from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import gym
import numpy as np
from gym import spaces

SCENE_LIST = ["事故场景绕行", "同车道施工", "压速", "直道cutin"]


class Scenarios(Enum):
    TRAINING = "train"
    INFERENCE = "inference"


class DoneReason:
    COLLIDED = "collided"
    TIMEOUT = "Timeout"
    MAX_EXP_STEP = "MaxExpStep"
    INFERENCE_DONE = "InferenceDone"
    Runtime_ERROR = "RuntimeError"


Observations = namedtuple("Observations", ["obs", "reward", "done", "info"])


@dataclass
class LaneInfo:
    lane_id: str
    lane_offset: float  # player 相对当前车道的横向偏倚，单位米

    @dataclass
    class Lane:
        lane_id: str
        width: float  # 单位 米
        speed_limit: float  # 单位 米/秒平方
        outgoing_lanes: List[LaneInfo.Lane]
        incoming_lane: List[LaneInfo.Lane]

    lanes: List[Lane]
    target_lanes: List[str]


class MatrixEnv(ABC, gym.Env):
    """MatrixEnv.
    MatrixEnv 客户端
    """

    def __init__(self, 
                 scenarios: Scenarios, 
                 scene_list: List[str], 
                 render_id: str) -> None:
        """
        Args:
            scenarios: Scenarios 训练模式使用Scenarios.TRAINING
                                 评估模式使用Scenarios.INFERENCE
            scene_list: List[str] 指定训练场景列表, 默认使用所有场景, 仅在训练模式生效
            render_id: 实例的实时回放id
        """

        self.action_space = spaces.Box(low=-6.0, high=2.0, shape=(2,))

        self.action_space = spaces.Box(
            np.array([-math.pi / 4.0, -6]).astype(np.float32),
            np.array([math.pi / 4.0, 2]).astype(np.float32),
        )

        self.observation_space = spaces.Dict(
            {
                "player": spaces.Dict(
                    {
                        "status": spaces.Box(-1e4, 1e4, shape=(9,)),
                        "property": spaces.Box(-1e4, 1e4, shape=(9,)),
                        "target": spaces.Box(-1e4, 1e4, shape=(8,)),
                    }
                ),
                "npc": spaces.Box(-1e4, 1e4, shape=(32, 11)),
            }
        )
        self.observation_space.contains = lambda x: True

    @abstractmethod
    def reset(self) -> Dict:
        """
        重置场景，每次会随机到不同考题场景中，agent的初始状态带速度

        Returns:
            Dict:  Obsevation 定义见 step 函数
        """
        return dict()

    @abstractmethod
    def step(self, actions: np.ndarray) -> Observations:
        """step.

        Args:
            actions (nd.ndarray): shape为(2,)的ndarray，actions[0]为前轮转角，
                                  actions[1]为加速度命令，分别由[-1,1]线性映射至
                                  实际前轮转角与加速度命令范围，具体见下面车辆参数

        Returns:
            Tuple[Dict, float, bool, Dict]:  Observation, Reward, Done, Info
                Observation 定义如下：
                                        {"player":{
                                                "status": np.ndarray(9,), 分别是
                                                                车辆后轴中心位置 x,
                                                                车辆后轴中心位置 y,
                                                                车体朝向,
                                                                车辆后轴中心纵向速度，
                                                                车辆后轴中心纵向加速度，
                                                                车辆后轴中心横向加速度，
                                                                当前前轮转角，
                                                                上一个前轮转角命令，
                                                                上一个加速度命令，
                                                "property": np.ndarray(9,), 分别是
                                                                车辆宽度，
                                                                车辆长度，
                                                                车辆轴距
                                                                车辆前悬
                                                                车辆后悬，
                                                                最小加速度指令
                                                                最大加速度指令
                                                                最小前轮转角
                                                                最大前轮转角
                                                "target": np.ndarray(8), 目标区域，x y 坐标交替
                                        }
                                        "npcs": ndarray(32, 11), agent附近障碍物 axis[0]为每一个障碍物, 附近障碍物数量
                            不足32个是以0 padding
                                                        axis[1]定义为:
                                                                id
                                                                类型
                                                                中心点x
                                                                中心点y
                                                                车体朝向
                                                                中心点速度x
                                                                中心点速度y
                                                                中心点加速度x
                                                                中心点加速度y
                                                                宽度
                                                                长度
                                        "map": LaneInfo() 地图定义见LaneInfo
                                        }

                                Reward: 默认输出0，由选手自行设计

                                Done: 以下条件会Done会为True
                                                （1）主车后轴中心进入目标区域
                                                （2）主车bounding box任意点出地图或与环境车辆碰撞
                                                （3）超时，时限根据场景不同

                                Info：{"DoneReason": 
                                     # "collided", 
                                     # "Timeout", 
                                     # "MaxExpStep"
                                     # "RuntimeError", 运行时异常, 此时可忽略当前帧
                                    "TotalExpStepRemain": int(), #总剩余步数，
                                    "MaxStep": int(), #场景最大步数
                                    "CurrentStep": int()， #场景当前步数
                                    }

        车辆参数：
            宽度：2.110 米
            长度：4.933 米
            轴距：2.670 米
            前悬：1.220 米
            后悬：1.043 米
            加速命令范围：[-10, 10] 包含两端，单位米/秒平方
            前轮转角范围：[-PI/4, PI/4]

        """
        return dict(), 0.0, False, dict()

    def centers_by_lane_id(self, lane_id: str) -> List[Tuple[float, float]]:
        assert self.matrix_env_, "MatrixEnv should not be none."

        rpc_status, centers = self.matrix_env_.centers_by_lane_id(lane_id)
        if rpc_status != 0:
            raise Exception("call centers_by_lane_id with rpc error.")
        return centers

    def instance_key(self) -> str:
        return self.key_
