import numpy as np


# Daily and weekly risk-free rate are computed from 1-year treasury yield curves
R_F_DAY = 1.0024580517515007
R_F_WEEK = 1.0173817974567245
R_F_MONTH = 1.93
R_F_QUARTER = 2.03
R_F_YEAR = 2.45


def get_sharpe_ratio(price_vec, r_f=R_F_DAY):
    return (1 + np.mean(price_vec) - r_f) / np.std(price_vec)


def get_portfolio_price_history(data, w_vec):
    # Transform histories by portfolio allocation weights
    allocated_price_histories = np.diag(w_vec) * np.matrix(data)
    # Collapse to a single price history
    return np.array(np.sum(allocated_price_histories, 0).T)


def compute_scores(data, w_vec):
    """
    Return [tuple]: (expectation, standard deviation, Sharpe ratio)
    """
    price_history = get_portfolio_price_history(data, w_vec)
    return (np.mean(price_history), np.std(price_history),
            get_sharpe_ratio(price_history))
