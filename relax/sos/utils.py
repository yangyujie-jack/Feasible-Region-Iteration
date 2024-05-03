import itertools
import math

import numpy as np
import sympy as sp
from SumOfSquares import SOSProblem


def sym_poly_terms(x: np.ndarray, order: int):
    x_aug = np.concatenate((x, [1]))
    terms = []
    for comb in itertools.combinations_with_replacement(x_aug, order): 
        terms.append(math.prod(comb))
    return terms


def sym_poly(x: np.ndarray, order: int, name: str = ''):
    terms = sym_poly_terms(x, order)
    coefs = sp.symarray(name, len(terms))
    return sum([c * t for c, t in zip(coefs, terms)]), coefs


def get_coefs(coefs: np.ndarray, prob: SOSProblem) -> np.ndarray:
    return np.array([prob.sym_to_var(c).value for c in coefs])


def get_mu_coefs_var(mu_coefs: list, prob: SOSProblem):
    mu_coefs_var = []
    for mc in mu_coefs:
        mu_coefs_var.append(get_coefs(mc, prob))
    return mu_coefs_var


def get_phi_coefs_var(phi_coefs: list, prob: SOSProblem):
    phi_coefs_var = []
    for pcs in phi_coefs:
        pcs_var = []
        for pc in pcs:
            pcs_var.append(get_coefs(pc, prob))
        phi_coefs_var.append(pcs_var)
    return phi_coefs_var
