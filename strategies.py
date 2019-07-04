import numpy as np

from preprocess import load_prices, load_marketcaps, DEFAULT_CURRENCIES
from allocate import mpt_opt, mktcap_opt
from evaluate import compute_scores


def slice_index_params(l, start_index, end_index):
    start_index = start_index or 0
    end_index = end_index or len(l)
    return [x[start_index:end_index] for x in l]


class PortfolioStrategy():
    def test(self, start_index=None, end_index=None):
        sliced_prices = slice_index_params(self.prices, start_index, end_index)
        return compute_scores(sliced_prices, self.weights)


class MeanVarianceStrategy(PortfolioStrategy):
    def __init__(self):
        self.prices = load_prices()
        self.N = len(self.prices)
        self.weights = [0] * self.N
        self.gamma = None
        self.trained_sharpe_ratio = None

    def train(self, start_index=None, end_index=None):
        sliced_prices = slice_index_params(self.prices, start_index, end_index)

        gammas = np.logspace(-2, 2, num=100)
        w_vec_results, ret_results, risk_results = mpt_opt(sliced_prices, gammas)
        # This is slow - bisect so it runs in log(N) time
        sharpe_ratios = [compute_scores(sliced_prices, w_vec_results[i])[-1]
                         for i, gamma in enumerate(gammas)]

        max_sharpe_ratio = max(sharpe_ratios)
        max_sharpe_ratio_index = sharpe_ratios.index(max_sharpe_ratio)

        self.weights = w_vec_results[max_sharpe_ratio_index]
        self.gamma = gammas[max_sharpe_ratio_index]
        self.trained_sharpe_ratio = max_sharpe_ratio
        return (self.weights,
                self.gamma,
                ret_results[max_sharpe_ratio_index],
                risk_results[max_sharpe_ratio_index],
                self.trained_sharpe_ratio)


class MarketCapStrategy(PortfolioStrategy):
    def __init__(self, num_assets=10):
        self.prices = load_prices()
        self.marketcaps = load_marketcaps()
        self.N = len(self.prices)
        self.weights = [0] * self.N
        self.gamma = None
        self.num_assets = num_assets

    def train(self, start_index=None, end_index=None):
        sliced_prices = slice_index_params(self.prices, start_index, end_index)
        sliced_market_caps = slice_index_params(self.marketcaps, start_index, end_index)

        # Limit to a portfolio of the top @self.num_assets market caps
        sorted_avg_market_caps = sorted(
            [(i, np.mean(l)) for i, l in enumerate(sliced_market_caps)],
            key=lambda x: -x[1]
        )[:self.num_assets]
        total_market_cap = sum([x[1] for x in sorted_avg_market_caps])

        self.weights = [0] * self.N  # Reset
        for i, avg_market_cap in sorted_avg_market_caps:
            self.weights[i] = avg_market_cap / total_market_cap


class EquallyWeightedStrategy(PortfolioStrategy):
    def __init__(self, num_assets=10):
        self.prices = load_prices()
        self.marketcaps = load_marketcaps()
        self.N = len(self.prices)
        self.weights = [0] * self.N
        self.gamma = None
        self.num_assets = num_assets

    def train(self, start_index=None, end_index=None):
        sliced_prices = slice_index_params(self.prices, start_index, end_index)
        sliced_market_caps = slice_index_params(self.marketcaps, start_index, end_index)

        # Limit to a portfolio of the top @self.num_assets market caps
        sorted_avg_market_caps = sorted(
            [(i, np.mean(l)) for i, l in enumerate(sliced_market_caps)],
            key=lambda x: -x[1]
        )[:self.num_assets]

        self.weights = [0] * self.N  # Reset
        for i, avg_market_cap in sorted_avg_market_caps:
            self.weights[i] = 1.0 / self.num_assets


class BuyAndHoldStrategy(PortfolioStrategy):
    def __init__(self, currency):
        self.prices = load_prices()
        self.N = len(self.prices)
        self.weights = [0] * self.N
        self.currency = currency
        self.currency_index = DEFAULT_CURRENCIES.index(currency)

    def train(self, start_index=None, end_index=None):
        self.weights[self.currency_index] = 1

