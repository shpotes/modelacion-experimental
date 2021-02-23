#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def compare_models(models_dict, **kwargs):
    for name, model in models_dict.items():
        t, y = model.simulate(**kwargs)
        plt.plot(t, y, label=name)

        if 'real' in models_dict and name != 'real':
            _, y_real = models_dict['real'].simulate(**kwargs)
            print(f'Error {name} = ', np.mean((y_real - y) ** 2))

    plt.legend()
    plt.show()
