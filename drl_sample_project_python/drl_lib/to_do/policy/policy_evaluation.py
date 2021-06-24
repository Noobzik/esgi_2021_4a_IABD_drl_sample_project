import numpy as np
from tqdm import tqdm
from ...do_not_touch.contracts import SingleAgentEnv
from ...do_not_touch.result_structures import ValueFunction


def policy_evaluation(
        env: SingleAgentEnv,
        epsilon: float,
        gamma: float,
        max_iter: int) -> ValueFunction:
    pass

