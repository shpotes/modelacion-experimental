from typing import Optional, Tuple, Union
from functools import partial

import numpy as np
from scipy import signal

from src import DiscreteModel

def compute_theoretical_weighting_func(
        model: DiscreteModel,
        time: Optional[np.array] = None,
) -> Tuple[np.array]:
    if time is None:
        time = np.arange(0, 10, model.system.dt)

    time, (weighting,) = signal.dimpulse(
        model.system,
        t=time
    )

    return time, weighting.reshape(-1)

def identify_weighting_sequence(
        model: DiscreteModel,
        max_value: Union[int, float] = 10,
) -> Tuple[np.array]:
    time = np.arange(0, max_value, model.system.dt)
    input_signal = np.random.normal(size=time.shape)

    time, yout = model.simulate(time, input_signal)
    weighting_seq = compute_weighting_from_signal(
        input_signal, yout, len(time)
    )

    return time, weighting_seq

def compute_weighting_from_signal(
        input_signal: np.array,
        data: np.array,
        seq_length: int = 10
) -> np.array:
    def corr(y, u, N, tau):
        return sum(y[t] * u[t - tau] for t in range(tau + 1, N)) / N

    std = np.mean(input_signal ** 2)
    autocorr = partial(corr, data, input_signal, seq_length)

    return np.array([autocorr(tau) / std for tau in range(seq_length)])

def simulate_from_seq(weighting, input_signal):
    return np.convolve(weighting, input_signal)
