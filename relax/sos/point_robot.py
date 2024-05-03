import numpy as np
import sympy as sp
from relax.sos.solve import solve_d, solve_mu_phi, solve_h
from relax.sos.utils import sym_poly_terms, sym_poly


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def point_robot_cbf(obs):
    c = [-1.84337645e-03, 8.74590356e-04, 1.49205236e+00, 
         -7.44798947e-01, -8.75317506e-04, -7.75368840e-01]
    d = np.linalg.norm(obs[..., :2], axis=-1)
    theta = np.arctan2(obs[..., 4], obs[..., 3])
    heading = np.arctan2(-obs[..., 1], -obs[..., 0])
    alpha = angle_normalize(heading - theta)
    u = np.clip(obs[..., 2] * np.cos(alpha), 0, None)
    cbf = c[0] * d ** 2 + c[1] * d * u + \
        c[2] * d + c[3] * u ** 2 + c[4] * u + c[5]
    return -cbf


if __name__ == '__main__':
    # [d, u]
    xs = sp.symarray('x', 2)

    # dynamics
    f = np.array([-xs[1], 0])
    g = np.array([[0], [1]])

    # action space convex hull
    us = [[-1], [1]]

    # constraints
    # ps[i] is a convex set, ps[i][j]<=0 means constraint violation
    ps = [
        [xs[0] - 0.52],
    ]

    # CBF
    # [x_0**2, x_0*x_1, x_0, x_1**2, x_1, 1]
    h_terms = sym_poly_terms(xs, 2)

    # ellipsoid
    elps = (xs[0] - 2) ** 2 + xs[1] ** 2

    # initialize
    h_coefs = np.array([-1e-2, 0, 1, -1, 0, -1.0])
    mu_terms = sym_poly_terms(xs, 2)
    phi_terms = sym_poly_terms(xs, 2)
    last_d = None

    while True:
        h = sym_poly(h_terms, h_coefs)

        d = solve_d(x=xs, h=h, elps=elps, psi_order=2)
        if last_d is not None:
            d_incr = d - last_d
            print('d increment:', d_incr)
            if d_incr < 1e-5:
                print('h coefficients:', h_coefs)
                break
        last_d = d

        mu_coefs, phi_coefs = solve_mu_phi(
            x=xs, us=us, ps=ps, h=h, f=f, g=g, mu_terms=mu_terms, 
            phi_terms=phi_terms, beta=0.0, kappa=0.1)

        mus = []
        for i in range(len(us) + 1):
            mus.append(sym_poly(mu_terms, mu_coefs[i]))

        phis = []
        for i in range(len(ps)):
            phis_i = []
            for j in range(len(ps[i]) + 1):
                phis_i.append(sym_poly(phi_terms, phi_coefs[i][j]))
            phis.append(phis_i)

        h_coefs = solve_h(
            x=xs, us=us, ps=ps, h_terms=h_terms, f=f, g=g, elps=elps, d=d,
            mus=mus, phis=phis, beta=0.0, kappa=0.1, nu_degree=2)
