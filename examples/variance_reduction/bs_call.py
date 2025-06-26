# Analytical Black-Scholes price for a European call option
import math
from scipy.stats import norm

def bs_call_price(spot, strike, rate, vol, expiry):
    d1 = (math.log(spot/strike) + (rate + 0.5*vol**2)*expiry) / (vol*math.sqrt(expiry))
    d2 = d1 - vol*math.sqrt(expiry)
    return spot * norm.cdf(d1) - strike * math.exp(-rate*expiry) * norm.cdf(d2)

if __name__ == "__main__":
    spot = 100.0
    strike = 100.0
    rate = 0.05
    vol = 0.2
    expiry = 1.0
    price = bs_call_price(spot, strike, rate, vol, expiry)
    print(f"BS Call Price: {price:.4f}")
