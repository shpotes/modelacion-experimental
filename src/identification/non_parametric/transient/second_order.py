from typing import Any, Dict, Tuple
import math
import warnings

import numpy as np
import sympy as sy

from src import ContinuousModel
from src.identification.non_parametric.transient import identify_first_order

def print_eq(
        k: float, w0: float, z: float
) -> sy.Expr:
    s = sy.symbols('s')
    return k * w0**2 / (s**2 + 2 * z * w0 + w0 ** 2)

def compute_second_order_params(
        k: float, Mp: float, tp: float
) -> Tuple[float, float]:
    z = 1 / math.sqrt(1 + math.pi ** 2 / math.log(Mp / k) ** 2)
    w0 = math.pi / (tp * math.sqrt(1 - z**2))

    return z, w0

def create_second_order_model_from_params(
        k: float, w0: float, z: float, tau: float,
) -> ContinuousModel:
    return ContinuousModel(
        [k * w0 ** 2],
        [1, 2 * z * w0, w0 ** 2],
        delay=tau
    )

def identify_second_order(
        system: ContinuousModel,
        simulation_parameters: Dict[str, Any] = {},
) -> Dict[str, Any]:

    time, yout = system.simulate(**simulation_parameters)

    for i, val in enumerate(yout):
        if val != 0:
            tau = time[i - 1]
            break

    K = yout[-1]
    Mp = yout.max() - K
    tp = time[yout.argmax()] - tau

    # return (K, Mp, tp)

    if Mp == 0:
        warnings.warn('the system is overdamped, a first order model will be used')
        return identify_first_order(system, simulation_parameters)

    z, w0 = compute_second_order_params(K, Mp, tp)

    return {
        'params': (K, w0, z),
        'model': create_second_order_model_from_params(K, w0, z, tau),
        'equation': print_eq(K, w0, z)
    }
