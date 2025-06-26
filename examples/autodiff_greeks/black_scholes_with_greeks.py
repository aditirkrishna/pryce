from scipy.stats import norm
from math import log, sqrt, exp

def bs_call(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    return norm.cdf(d1)

def bs_gamma(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5*sigma**2)*T) / (sigma * sqrt(T))
    return norm.pdf(d1) / (S * sigma * sqrt(T))

if __name__ == '__main__':
    S = 100.0
    K = 105.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    print(f"Call:  {bs_call(S,K,T,r,sigma):.4f}")
    print(f"Delta: {bs_delta(S,K,T,r,sigma):.4f}")
    print(f"Gamma: {bs_gamma(S,K,T,r,sigma):.6f}")
