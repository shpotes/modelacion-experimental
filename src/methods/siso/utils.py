from typing import List
from functools import partial
import sympy as sy
from sympy.abc import t

def function_maker(symbol: str, n: int, nk: int = 1):
    sym_string = ' '.join([
        f'{symbol}_{i}' if i != 0 else '1' for i in range(nk, n + 1)
    ])
    symbols = sy.symbols(sym_string)

    if isinstance(symbols, sy.Symbol):
        symbols = [symbols]

    return list(symbols)

B_maker = partial(function_maker, 'b')
A_maker = partial(function_maker, 'a', nk=0)
C_maker = partial(function_maker, 'c', nk=0)
D_maker = partial(function_maker, 'd', nk=0)

def diff_maker(sym_func: sy.Function, n: int, nk: int = 1) -> List[sy.Symbol]:
    symbols = [
        sym_func(t - i) for i in range(nk, n + nk)
    ]

    return symbols


def make_poly(q_sym, seed: List[sy.Symbol], nk: int = 0) -> sy.Expr:
    res = 0

    for i, sym in enumerate(seed, start=nk):
        res += sym * q_sym ** -i

    return res
