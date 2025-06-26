from scipy.stats import norm
from math import log, sqrt, exp

# European call option Black-Scholes formula

def black_scholes_call(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)

if __name__ == '__main__':
    # Match parameters in call_option.mlir and config.yaml
    S = 100.0
    K = 105.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    price = black_scholes_call(S, K, T, r, sigma)
    print(f"Analytical Black-Scholes price: {price:.4f}")
