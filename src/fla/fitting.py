import numpy as np


def linear(x, m: float, b: float):
    """Linear model: m*x + b."""
    return m * np.asarray(x) + b


def power_law(x, a: float, b: float):
    """Power-law model: a * x^b."""
    return a * np.asarray(x) ** b
