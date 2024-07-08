import numpy as np
import scipy


def ScaledBetaDist(x, n_levels, a, b):
    dist = scipy.stats.beta(a, b)
    return 1 / n_levels * dist.cdf(x / n_levels)


def QuantiseScaledBetaDist(total_volume, n_levels, a, b):
    probs = []
    for i in range(n_levels):
        prob = ScaledBetaDist(i + 1, n_levels, a, b) - ScaledBetaDist(i, n_levels, a, b)
        probs.append(prob)

    probs = np.array(probs) / np.sum(probs)
    order_profile = np.round(probs * total_volume)

    return order_profile


def ScaledBetaDist_v2(x, n_levels, a, b):
    dist = scipy.stats.beta(a, b)
    return 1 / n_levels * dist.pdf(x / n_levels)


def QuantiseScaledBetaDist_v2(total_volume, n_levels, a, b):
    probs = []
    for i in range(1, n_levels + 1):
        x = i - 0.5
        prob = ScaledBetaDist_v2(x, n_levels, a, b)
        probs.append(prob)

    probs = np.array(probs) / np.sum(probs)
    order_profile = np.round(probs * total_volume)

    return order_profile


total_volume = 100
print("Output of using CDF:", QuantiseScaledBetaDist(total_volume, 11, 2, 5))
# print("Output of the method in the paper:", QuantiseScaledBetaDist(total_volume, 2, 5, 10))