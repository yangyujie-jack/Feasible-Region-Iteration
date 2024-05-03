import numpy as np
import sympy as sp
from relax.sos.utils import sym_poly, get_coefs
from SumOfSquares import SOSProblem


def compute_b_coefs():
    # state
    x = sp.symarray('x', 2)

    # dynamics
    f = np.array([x[1], 0])
    g = np.array([[0], [1]])

    # constraints
    # rho[i] >= 0
    rho = [
         x[0] + 5,
        -x[0] + 5,
         x[1] + 5,
        -x[1] + 5,
    ]

    # CBF
    b, b_coefs = sym_poly(x, 4, 'b')
    db = np.array([b.diff(xi) for xi in x])

    # controller
    beta = np.array([-10 * x[0] - 10 * x[1]])

    s = []
    for i in range(2 * len(rho) + 1):
        s.append(sym_poly(x, 2, f's_{i}')[0])

    prob = SOSProblem()

    # CBF necessary condition
    for si, r in zip(s[1:len(rho) + 1], rho):
        prob.add_sos_constraint(-b + r * si, x)
        prob.add_sos_constraint(si, x)

    # CBF invariant inequality
    alpha = 0.1
    prob.add_sos_constraint(
        db @ f + db @ g @ beta + alpha * b - 
        sum([si * r for si, r in zip(s[len(rho) + 1:], rho)]),
        x
    )
    for si in s[len(rho) + 1:]:
        prob.add_sos_constraint(si, x)

    # ellipsoid constraint
    r = 2.0
    Q = np.array([[1, 2], [2, 4.5]], dtype=np.float32)
    elp = x.T @ Q @ x
    prob.add_sos_constraint((elp - r) * s[0] + b, x)
    prob.add_sos_constraint(s[0], x)

    prob.solve()

    return get_coefs(b_coefs, prob)


def double_integrator_cbf(obs):
    b_coefs = compute_b_coefs()
    c = np.zeros((5, 5))

    coef_index_map = [
        (4, 0),
        (3, 1),
        (3, 0),
        (2, 2),
        (2, 1),
        (2, 0),
        (1, 3),
        (1, 2),
        (1, 1),
        (1, 0),
        (0, 4),
        (0, 3),
        (0, 2),
        (0, 1),
        (0, 0),
    ]

    for i, idx in enumerate(coef_index_map):
        c[idx] = b_coefs[i]

    x = obs[0, :, 0] / 1.1
    y = obs[:, 0, 1] / 0.85
    return -np.polynomial.polynomial.polygrid2d(x, y, c).T
