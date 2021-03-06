import logging
import sympy as sy
from src.methods.siso.utils import *

class ARMAX:
    def __init__(self, na, nb, nc, nk):
        self.A_seed = A_maker(n=na)
        self.B_seed = B_maker(n=nb)
        self.C_seed = C_maker(n=nc)

        self._na = na
        self._nb = nb
        self._nc = nc
        self._nk = nk

    def get_transfer_function(self, steps: bool = True):
        q = sy.symbols('q')

        A_poly = make_poly(q, self.A_seed)
        B_poly = make_poly(q, self.B_seed, self._nk)
        C_poly = make_poly(q, self.C_seed)

        G_tf = B_poly / A_poly
        H_tf = C_poly / A_poly

        tf_step = G_tf, H_tf
        logging.info(tf_step)

        z = sy.symbols('z')
        tf_z_step = tuple(map(lambda x: x.replace(q, z), tf_step))
        logging.info(tf_z_step)

        u_t = sy.symbols('u(t)')
        e_t = sy.symbols('e(t)')

        transfer_function = tf_z_step[0].simplify() * u_t + tf_z_step[1].simplify() * e_t

        return transfer_function

    def get_diff_equation(self):
        y = sy.symbols('y')
        u = sy.symbols('u')
        e = sy.symbols('e')

        ya_seed = diff_maker(y, self._na, self._nk)
        ub_seed = diff_maker(u, self._nb, self._nk)
        ec_seed = diff_maker(e, self._nc, self._nk)

        a = sum(y * a for y, a in zip(ya_seed, self.A_seed))
        b = sum(y * b for y, b in zip(ub_seed, self.B_seed))
        c = sum(y * c for y, c in zip(ec_seed, self.C_seed))
        
        return a, b + c

    def get_regresion_form(self):
        y = sy.symbols('y')
        u = sy.symbols('u')

        ya_seed = diff_maker(y, self._na, self._nk)
        ub_seed = diff_maker(u, self._nb, self._nk)
        ec_seed = diff_maker(e, self._nc, self._nk)

        phi = sy.Matrix([*ya_seed, *ub_seed, *ec_seed])
        theta = sy.Matrix([*self.A_seed[1:], *self.B_seed, self.C_seed[1:])

        return {'Φ': phi, 'θ': theta}
