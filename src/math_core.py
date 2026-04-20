# src/math_core.py
import math

def poisson_pmf(lmbda, k):
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)

def kelly_fraction(prob, odds, fraction=0.25):
    """
    Kelly fractional: f* = fraction * ((b*p - q) / b)
    where b = odds - 1, q = 1 - p
    """
    b = odds - 1.0
    q = 1.0 - prob
    if b <= 0:
        return 0.0
    f = (b * prob - q) / b
    return max(0.0, f * fraction)

def expected_value(prob, odds):
    """
    EV = p * (odds - 1) - (1 - p)
    """
    return prob * (odds - 1.0) - (1.0 - prob)
