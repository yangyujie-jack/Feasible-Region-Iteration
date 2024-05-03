import numpy as np
import sympy as sp
from relax.sos.solve import solve_d, solve_mu_phi, solve_h
from relax.sos.utils import sym_poly_terms
from relax.sos.double_integrator2 import compute_b_coefs


def double_integrator_cbf(obs):
    c = [-6.46493662e-01, -6.25060383e-01, -3.08485811e-09,
         -1.95207947e+00, -1.68209774e-08,  1.49114304e+01]
    cbf = c[0] * obs[..., 0] ** 2 + c[1] * obs[..., 0] * obs[..., 1] + \
        c[2] * obs[..., 0] + c[3] * obs[..., 1] ** 2 + c[4] * obs[..., 1] + c[5]
    return -cbf


if __name__ == '__main__':
    x = sp.symarray('x', 2)

    # dynamics
    f = np.array([x[1], 0])
    g = np.array([[0], [1]])

    # action space convex hull
    us = [[-1], [1]]

    # constraints
    # rho[i] >= 0
    ps = [
        [ x[0] + 5],
        [-x[0] + 5],
        [ x[1] + 5],
        [-x[1] + 5],
    ]

    # CBF
    # [x_0**2, x_0*x_1, x_0, x_1**2, x_1, 1]
    h_terms = sym_poly_terms(x, 4)

    # ellipsoid
    Se = np.array([[1, 0], [0, 1]])
    elps = x.T @ Se @ x
    d = 2.0

    # initialize
    # h_coefs = np.array([-1, 0, 0, -1, 0, 1])
    h_coefs = compute_b_coefs()
    last_d = None

    while True:
        h = sum([c * t for c, t in zip(h_coefs, h_terms)])

        mu_coefs, phi_coefs = solve_mu_phi(
            x=x, us=us, ps=ps, h=h, f=f, g=g,
            mu_order=2, phi_order=2, beta=0.0, kappa=0.1)

        h_coefs = solve_h(
            x=x, us=us, ps=ps, h_terms=h_terms, f=f, g=g, elps=elps, d=d,
            mu_coefs=mu_coefs, phi_coefs=phi_coefs, mu_order=2, phi_order=2,
            beta=0.0, kappa=0.1, nu_degree=2)

        print(h_coefs)
