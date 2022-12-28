# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

from multiprocessing import Pool

import gym
import numpy

from geek.env.logger import Logger
from geek.env.matrix_env import DoneReason, Scenarios

logger = Logger.get_logger(__name__)


def run(worker_index):
    try:
        env = gym.make("MatrixEnv-v1", scenarios=Scenarios.INFERENCE)
        obs = env.reset()
        while True:
            observation, reward, done, info = env.step(numpy.array([0.1, 0]))
            infer_done = DoneReason.INFERENCE_DONE == info.get("DoneReason", "")
            if done and not infer_done:
                obs = env.reset()
            elif infer_done:
                break
    except Exception as e:
        logger.info(f"{worker_index}, error: {str(e)}")


if __name__ == "__main__":
    num_workers = 12

    pool = Pool(num_workers)
    pool_result = pool.map_async(run, list(range(num_workers)))
    pool_result.wait(timeout=3000)

    logger.info("inference done.")
