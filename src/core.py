from typing import Tuple

import numpy as np
from scipy import signal, stats

EPS = 1e-4

class Model:
    def __str__(self):
        return f'{self.system}\n with delay {self.delay}'

    def __repl__(self):
        return f'{self.system}\n with delay {self.delay}'

class ContinuousModel(Model):
    def __init__(self, *args, delay=0):
        self.system = signal.lti(*args)
        self.delay = delay
    
    def simulate(
            self,
            time: np.array = None,
            input_signal: np.array = None,
    ) -> Tuple[np.array]:
        if time is None:
            time = np.linspace(0, 10, 1000)

        if input_signal is None:
            input_signal = np.ones_like(time)

        time, yout, _ = signal.lsim(
            self.system,
            input_signal,
            time
        )

        new_yout = np.zeros_like(yout)
        new_yout[(time < self.delay).sum():] = yout[:(time >= self.delay).sum()]
        yout = new_yout

        return time, yout

    def get_transfer_function(self):
        return signal.TransferFunction(self.system)


class DiscreteModel(Model):
    def __init__(self, *args, dt=1):
        self.system = signal.dlti(*args, dt=dt)

    def simulate(
            self,
            time: np.array = None,
            input_signal: np.array = None,
    ):
        if time is None:
            time = np.arange(0, 10, self.system.dt)

        if input_signal is None:
            input_signal = np.ones_like(time)

        time, yout = signal.dlsim(
            self.system,
            input_signal,
            time,
        )

        return time, yout.reshape(-1)
