from cvxpy import Variable, Parameter, Problem, Maximize, quad_form, sqrt
import numpy as np


def mpt_opt(data, gamma_vec):
    NUM_SAMPLES = len(gamma_vec)
    w_vec_results = [None] * NUM_SAMPLES
    ret_results = np.zeros(NUM_SAMPLES)
    risk_results = np.zeros(NUM_SAMPLES)

    N = len(data)
    w_vec = Variable(N)
    mu_vec = np.array([np.mean(data[i]) for i in range(N)])
    sigma_mat = np.cov(data)

    gamma = Parameter(nonneg=True)

    ret_val = mu_vec.T * w_vec
    risk_val = quad_form(w_vec, sigma_mat)  # w^T Sigma w
    problem = Problem(Maximize(ret_val - gamma * risk_val),
                      [sum(w_vec) == 1, w_vec >= 0])

    for i, new_gamma in enumerate(gamma_vec):
        gamma.value = new_gamma
        problem.solve()
        w_vec_results[i] = w_vec.value
        ret_results[i] = ret_val.value
        risk_results[i] = sqrt(risk_val).value

    return (w_vec_results, ret_results, risk_results)


def mktcap_opt(marketcaps, index=0):
    target_marketcaps = [marketcap[index] for marketcap in marketcaps]
    total = float(sum(target_marketcaps))
    return [marketcap / total for marketcap in target_marketcaps]

