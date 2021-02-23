from typing import Any, Dict
import warnings

import numpy as np
from scipy import signal, stats
import sympy as sy

from src import ContinuousModel

EPS = 1e-4

def identify_first_order(
    system: ContinuousModel,
    method: str = 'graphic',
    cons: float = 1,
    simulation_parameters: Dict[str, Any] = {},
) -> Dict[str, Any]:

    def print_eq(K, T, tau, cons):
        s = sy.symbols('s')
        return cons * K * sy.exp(-tau * s) / (cons * T * s + cons)

    time, yout = system.simulate(**simulation_parameters)

    # Estimate tau
    for i, val in enumerate(yout):
        if val != 0:
            tau = time[i - 1]
            break

    if yout[-1] - yout[-2] > EPS:
        warnings.warn('maybe you want to simulate for longer')

    K = yout[-1]

    if method == 'graphic':
        T = time[abs(yout - K * 0.632).argmin()] - tau
    elif method == 'linear':
        mask = (time > tau) & (yout < K)
        x_lr = time[mask]
        y_lr = np.log(1 - yout[mask] / K)

        reg = stats.linregress(x_lr, y_lr)

        T = -1/reg.slope
        tau = -reg.intercept / reg.slope
    else:
        raise NotImplementedError

    return {
        'params': (K, T, tau),
        'model': ContinuousModel([K], [T, 1], delay=tau),
        'equation': print_eq(K, T, tau, cons)
    }
