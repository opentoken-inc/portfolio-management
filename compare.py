import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_prices, load_marketcaps
from allocate import mpt_opt, mktcap_opt
from evaluate import compute_scores
from strategies import (MeanVarianceStrategy, MarketCapStrategy,
                        EquallyWeightedStrategy, BuyAndHoldStrategy)


def plot_efficient_frontier(ret_vec, risk_vec, gamma_vec, show=False):
    NUM_MARKERS = 3
    markers_on = [int(i * len(gamma_vec) / (NUM_MARKERS-1))
                  for i in range(NUM_MARKERS-1)] + [len(gamma_vec) - 1]

    plt.scatter(risk_vec, ret_vec)
    for marker in markers_on:
        plt.plot(risk_vec[marker], ret_vec[marker], 'bs')
        plt.annotate(r'$\gamma = {:.2f}$'.format(gamma_vec[marker]),
                     xy=(risk_vec[marker], ret_vec[marker] + 0.001))
    plt.xlabel('Standard deviation')
    plt.ylabel('Return')
    if show:
        plt.show()


def plot_marketcap_weighted(data, show=False):
    mktcap_w_vec = mktcap_opt(load_marketcaps())
    mean, std, sharpe = compute_scores(data, mktcap_w_vec)
    plt.plot(std, mean, 'mo')
    if show:
        plt.show()


def plot_equally_weighted(data, show=False):
    N = len(data)
    equal_w_vec = [1 / float(N)] * N
    mean, std, sharpe = compute_scores(data, equal_w_vec)
    plt.plot(std, mean, 'co')
    if show:
        plt.show()


def plot_individual_assets(data, show=False):
    for i in range(len(data)):
        new_w_vec = [0] * len(data)
        new_w_vec[i] = 1
        mean, std, sharpe = compute_scores(data, new_w_vec)
        plt.plot(std, mean, 'ro')
        plt.annotate('coin {}'.format(i), xy=(std, mean))
    if show:
        plt.show()


def plot_sharpe_ratios(data, w_vec, gammas, show=False):
    for i, gamma in enumerate(gammas):
        plt.plot(gamma, compute_scores(data, w_vec[i])[2], 'bo')
    plt.xlabel('Gamma')
    plt.ylabel('Sharpe Ratio')
    if show:
        plt.show()


def test_scores(prices):
    """DEPRECATED"""
    N = len(prices)
    results = []
    for i in range(N):
        new_w_vec = [0] * N
        new_w_vec[i] = 1
        results.append([i] + list(compute_scores(prices, new_w_vec)))
    # MPT
    gammas = np.logspace(-2, 2, num=100)
    w_vec_results, ret_results, risk_results = mpt_opt(prices, gammas)
    results.append(['MVO'] + list(compute_scores(prices, w_vec_results[0])))
    # Market Cap
    mcap_vec = mktcap_opt(load_marketcaps())
    results.append(['MCAP'] + list(compute_scores(prices, mcap_vec)))
    # Equal Weights
    eq_w_vec = [1 / float(N)] * N
    results.append(['EQ'] + list(compute_scores(prices, eq_w_vec)))

    sorted_results = sorted(results, key=lambda x: -x[-1])
    for result in sorted_results:
        print(result)


def test_efficient_frontier(prices):
    N = len(prices)
    gammas = np.logspace(-2, 2, num=100)
    w_vec_results, ret_results, risk_results = mpt_opt(prices, gammas)

    plot_efficient_frontier(ret_results, risk_results, gammas)
    plot_individual_assets(prices)
    plot_marketcap_weighted(prices)
    plot_equally_weighted(prices)
    plt.show()


def test_portfolio_sharpe_ratio(prices):
    N = len(prices)
    gammas = np.logspace(-2, 2, num=100)
    w_vec_results, ret_results, risk_results = mpt_opt(prices, gammas)

    plot_sharpe_ratios(prices, w_vec_results, gammas)
    plt.show()


def compare_strategies():
    windows = list(range(0, 240, 30))
    # tuple: (Strategy, results, color, label)
    mvo = (MeanVarianceStrategy(), [], 'blue', 'Mean-Variance Optimized')
    mcap = (MarketCapStrategy(), [], 'green', 'Market Cap Weighted')
    eq = (EquallyWeightedStrategy(), [], 'magenta', 'Equally Weighted')
    bah_btc = (BuyAndHoldStrategy('bitcoin'), [], 'black', 'BAH BTC')
    bah_eth = (BuyAndHoldStrategy('ethereum'), [], 'red', 'BAH ETH')
    bah_xrp = (BuyAndHoldStrategy('ripple'), [], 'cyan', 'BAH XRP')
    bah_zrx = (BuyAndHoldStrategy('0x'), [], 'orange', 'BAH ZRX')
    bah_tether = (BuyAndHoldStrategy('tether'), [], 'yellow', 'BAH TETHER')

    strategy_tuples = [mvo, mcap, eq, bah_btc, bah_eth, bah_tether]

    for i in windows:
        print('WINDOW START:', i)
        for strategy_tuple in strategy_tuples:
            strategy, results, color, label = strategy_tuple
            strategy.train(i, i+30)
            _, _, sharpe = strategy.test(i+30, i+60)
            results.append(sharpe)
            plt.plot(i, sharpe, color=color, marker='o')
            print(label, strategy.test(i+30, i+60))

    for strategy_tuple in strategy_tuples:
        strategy, results, color, label = strategy_tuple
        plt.plot(windows, results, color=color, label=label)

    plt.legend()
    plt.show()


####################
# RUN TEST METHODS #
####################

#prices = load_prices()
#test_scores(prices)
#test_efficient_frontier(prices)
#test_portfolio_sharpe_ratio(prices)

compare_strategies()
