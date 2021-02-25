from typing import Optional
import math
import numpy as np

def PRBS31(code):
    for _ in range(32):
        next_bit = ~((code>>30) ^ (code>>27))&0x01
        code = ((code<<1) | next_bit) & 0xFFFFFFFF
    return code

def prbs(
        size: int,
        max_val: float = 1.,
        min_val: float = -1.,
        seed: Optional[int] = None
):
    if seed is None:
        seed = np.random.randint(0, 2 << 31)

    num_codes = math.ceil(size / 32) + 1
    codes = []
    for _ in range(num_codes):
        seed = PRBS31(seed)
        codes.extend(
            [(min_val - max_val) * int(i) + max_val for i in f'{seed:b}']
        )

    return np.array(codes[:size])
