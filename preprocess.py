from csv import reader as csvreader
from scipy import stats

import matplotlib.pyplot as plt


DEFAULT_CURRENCIES = ['bitcoin', 'ethereum', 'ripple', 'bitcoincash', 'eos',
                      'litecoin', 'stellar', 'iota', 'cardano', 'tether',
                      'tron', 'monero', 'neo', 'dash', 'ethereumclassic',
                      'nem', 'binancecoin', 'zcash', 'omisego', 'qtum', '0x']
DATA_PRICE_O_IDX = 1
DATA_MARKETCAP_IDX = 6


def format_number(s):
    new_s = s.replace(',', '').replace('-', '')
    return float(new_s) if new_s != '' else None


def compute_deltas(price_list):
    """
    @price_list is in reverse chronological order, so compute deltas in reverse
    """
    return [(price_list[i] - price_list[i+1]) / price_list[i]
            for i, _ in enumerate(price_list[:-1])]


def compute_binned_deltas(price_list):
    """
    @price_list is in reverse chronological order, so compute deltas in reverse
    """
    return [1 if price_list[i] > price_list[i+1] else -1
            for i, _ in enumerate(price_list[1:])]


def generate_bin_sequence(min_val, max_val, nbins):
    interval = (max_val - min_val) / float(nbins)
    return [min_val + i * interval for i in range(0, nbins+1)]


def load_raw_data_from_csv(filename):
    prices = []
    filename = filename if filename[:5] == 'data/' else 'data/{}'.format(filename)
    with open(filename, 'r') as f:
        f = csvreader(f, delimiter='\t')
        for i, line in enumerate(f):
            if i == 0:  # Header
                continue
            # date, price_o, price_h, price_l, price_c, volume, mkt_cap
            prices.append(line)
    return prices


def preprocess_prices(price_dump, f_preprocess=compute_deltas):
    """Format and convert to deltas"""
    opening_prices = [format_number(x[DATA_PRICE_O_IDX]) for x in price_dump]
    return f_preprocess(opening_prices)


def load_prices(currencies=DEFAULT_CURRENCIES):
    prices = []
    for filename in currencies:
        prices.append(preprocess_prices(load_raw_data_from_csv(filename + '.tsv')))

    # Truncate so each price set is the same size
    # Prices are sorted in reverse: from most to least recent - reverse this
    min_size = min([len(x) for x in prices])
    return [p[:min_size][::-1] for p in prices]


def load_marketcaps(currencies=DEFAULT_CURRENCIES):
    marketcaps = []
    for filename in currencies:
        marketcaps.append([format_number(x[DATA_MARKETCAP_IDX])
                           for x in load_raw_data_from_csv(filename + '.tsv')])

    # Truncate so each price set is the same size
    # Prices are sorted in reverse: from most to least recent - reverse this
    min_size = min([len(x) for x in marketcaps])
    return [p[:min_size][::-1] for p in marketcaps]


def test_plot(filename, limit=None):
    price_changes = preprocess_prices(load_raw_data_from_csv(filename + '.tsv'))
    if limit:
        price_changes = price_changes[:limit]
    bins = generate_bin_sequence(-1, 1, 50)

    # Plot histograms
    plt.hist(price_changes, bins=bins)
    plt.axis([-1, 1, 0, len(price_changes)])
    plt.title('{} price historgram ({} days)'.format(filename, len(price_changes)))
    print(stats.describe(price_changes))
    plt.show()


def test_all_plots(limit=None):
    for currency in DEFAULT_CURRENCIES:
        test_plot(currency, limit)


####################
# RUN TEST METHODS #
####################

#test_all_plots(limit=300)
