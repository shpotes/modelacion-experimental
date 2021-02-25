from typing import Optional, Tuple, Union
from functools import partial
import logging

import numpy as np
from scipy import signal
import sympy as sy
from sympy.abc import t, k

from src import DiscreteModel
from src import utils

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
        input_type: str = 'prbs'
) -> Tuple[np.array]:
    time = np.arange(0, max_value, model.system.dt)

    if input_type == 'gaussian':
        input_signal = np.random.normal(size=time.size)
    elif input_type == 'prbs':
        input_signal = utils.prbs(size=time.size)

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

##

def manual_correlation_method(input_signal, data, seq_len=5):
    Eu = manual_Eu_computation(input_signal)
    g_hat = partial(manual_g_computation, input_signal, data, Eu)
    
    g = []
    for tau in range(min(len(input_signal), seq_len)):
        g.append(g_hat(tau))
    
    return g

def manual_Eu_computation(u_data):
    N = len(u_data)
    u = sy.IndexedBase('u')

    Eu = sy.Sum(u[t] ** 2, (t, 1, N)) / N
    logging.info(f'Eu²(t) = {Eu.doit()}')

    Eu_vals = Eu.doit().subs([(u[i + 1], u_value) for i, u_value in enumerate(u_data)])
    logging.info(f'Eu²(t) = {Eu_vals}')
    
    return Eu_vals

def manual_g_computation(u_data, y_data, Eu_vals, tau):
    y = sy.IndexedBase('y')
    u = sy.IndexedBase('u')

    N = len(u_data)
    
    cross = sy.Sum(y[t]*u[t - tau], (t, 1, N)) / N
    logging.info(f'ĝ({tau}) = {cross}')

    inter_step = lambda y_data: (
        cross
        .doit()
        .subs([(u[i + 1], u_value) for i, u_value in enumerate(u_data)])
        .subs([(y[i + 1], y_value) for i, y_value in enumerate(y_data)])
    ) / Eu_vals

    g_sym = inter_step(map(lambda x: sy.symbols(str(x)), y_data))
    #if g_sym.args:
    #    g_sym = g_sym.args[0]
    logging.info(f'ĝ({tau}) = {g_sym}')

    g_val = inter_step(y_data)
    logging.info(f'ĝ({tau}) = {g_val}')

    if g_val.args:
        g_val = g_val.args[0]
    
    return g_val

def manual_simulation(weighting_seq, input_signal, n_points=3):
    y_hat = partial(manual_y_computation, weighting_seq, input_signal)
    
    y = []
    for t in range(1, n_points + 1):
        y.append(y_hat(t))
        
    return y

def manual_y_computation(weighting_seq, input_signal, T):
    logging.info('ŷ(t) = ĝ(t) * u(t)')

    g_ = sy.IndexedBase('ĝ')
    u = sy.IndexedBase('u')

    y_ = sy.Sum(g_[k] * u[T - k], (k, 0, T))
    logging.info(f'ŷ({T}) = {y_}')

    inter_step = lambda weighting_seq: (
            y_
            .doit()
            .subs([(u[i], u_value) for i, u_value in enumerate(input_signal)])
            .subs([(g_[i], g_value) for i, g_value in enumerate(weighting_seq)])
    )

    g_sym = inter_step(map(lambda x: sy.symbols(str(x)), weighting_seq))
    logging.info(f'ŷ({T}) = {g_sym}')
    g_val = inter_step(weighting_seq)
    logging.info(f'ŷ({T}) = {g_val}')
    
    return g_val
