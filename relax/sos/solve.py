import numpy as np
import sympy as sp
from SumOfSquares import SOSProblem
from relax.sos.utils import sym_poly_terms, sym_poly, get_mu_coefs_var, \
    get_phi_coefs_var


def solve_d(x: np.ndarray, h: sp.Expr, elps: sp.Expr, psi_order: int):
    d = sp.symbols('d')
    psi, _ = sym_poly(x, psi_order, 'psi')
    prob = SOSProblem()
    prob.add_sos_constraint(elps - d + psi * h, x)
    prob.add_sos_constraint(psi, x)
    d_var = prob.sym_to_var(d)
    prob.set_objective('max', d_var)
    prob.solve()
    return d_var.value


def solve_mu_phi(x: np.ndarray, us: list, ps: list, h: sp.Expr, f: np.ndarray,
                 g: np.ndarray, mu_order: int, phi_order: list, beta: float,
                 kappa: float):
    dh = np.array([h.diff(x) for x in x])
    Lfh = dh @ f
    Lgh = dh @ g

    mu_coefs, mus = [], []
    for i in range(len(us) + 1):
        mu_term = sym_poly_terms(x, mu_order)
        mu_coef = sp.symarray(f'mu_{i}', len(mu_term))
        mu = sum([c * t for c, t in zip(mu_coef, mu_term)])
        mu_coefs.append(mu_coef)
        mus.append(mu)

    phi_coefs, phis = [], []
    for i in range(len(ps)):
        phi_coefs_i, phis_i = [], []
        for j in range(len(ps[i]) + 1):
            phi_term = sym_poly_terms(x, phi_order)
            phi_coef = sp.symarray(f'phi_{i}_{j}', len(phi_term))
            phi = sum([c * t for c, t in zip(phi_coef, phi_term)])
            phi_coefs_i.append(phi_coef)
            phis_i.append(phi)
        phi_coefs.append(phi_coefs_i)
        phis.append(phis_i)

    prob = SOSProblem()
    prob.add_sos_constraint(
        (1 + mus[0]) * (beta - h) + sum(
            [mu * (Lfh + Lgh @ u + kappa * h) for mu, u in zip(mus[1:], us)]
        ), x)
    for mu in mus:
        prob.add_sos_constraint(mu, x)
    for i in range(len(ps)):
        prob.add_sos_constraint(
            -(1 + phis[i][0]) * h + sum(
                [phi * p for phi, p in zip(phis[i][1:], ps[i])]
            ), x)
        for phi in phis[i]:
            prob.add_sos_constraint(phi, x)
    prob.solve()

    mu_coefs_var = get_mu_coefs_var(mu_coefs, prob)
    phi_coefs_var = get_phi_coefs_var(phi_coefs, prob)
    return mu_coefs_var, phi_coefs_var


def solve_h(x: np.ndarray, us: list, ps: list, h_terms: np.ndarray,
            f: np.ndarray, g: np.ndarray, elps: sp.Expr, d: float, 
            mu_coefs: list, phi_coefs: list, mu_order: int, phi_order: int,
            beta: float, kappa: float, nu_degree: int):
    t = sp.symbols('t')

    h_coefs = sp.symarray('h', len(h_terms))
    h = sum([c * t for c, t in zip(h_coefs, h_terms)])
    dh = np.array([h.diff(x) for x in x])
    Lfh = dh @ f
    Lgh = dh @ g

    mus = []
    for mu_coef in mu_coefs:
        mu_term = sym_poly_terms(x, mu_order)
        mu = sum([c * t for c, t in zip(mu_coef, mu_term)])
        mus.append(mu)

    phis = []
    for phi_coef in phi_coefs:
        phis_i = []
        for pc in phi_coef:
            phi_term = sym_poly_terms(x, phi_order)
            phi = sum([c * t for c, t in zip(pc, phi_term)])
            phis_i.append(phi)
        phis.append(phis_i)

    nu_terms = sym_poly_terms(x, nu_degree)
    nu_coefs = sp.symarray('nu', len(nu_terms))
    nu = sum([c * t for c, t in zip(nu_coefs, nu_terms)])

    prob = SOSProblem()
    prob.add_sos_constraint(h - t - nu * (d - elps), x)
    prob.add_sos_constraint(nu, x)
    prob.add_sos_constraint(
        (1 + mus[0]) * (beta - h) + sum(
            [mu * (Lfh + Lgh @ u + kappa * h) for mu, u in zip(mus[1:], us)]
        ), x)
    for i in range(len(ps)):
        prob.add_sos_constraint(
            -(1 + phis[i][0]) * h + sum(
                [phi * p for phi, p in zip(phis[i][1:], ps[i])]
            ), x)
    t_var = prob.sym_to_var(t)
    prob.set_objective('max', t_var)
    prob.solve()

    h_coefs_var = np.array([prob.sym_to_var(c).value for c in h_coefs])
    return h_coefs_var
