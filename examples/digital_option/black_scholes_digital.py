from scipy.stats import norm
from math import exp, sqrt, log

def bs_digital_call(S, K, T, r, sigma):
    d2 = (log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return exp(-r * T) * norm.cdf(d2)

if __name__ == '__main__':
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.03
    sigma = 0.25
    price = bs_digital_call(S, K, T, r, sigma)
    print(f"Analytical digital call price: {price:.4f}")
