"""La Liga (Primera División) match prediction.

A time-decayed Dixon–Coles goals model produces a full-time scoreline
distribution. Half-time probabilities come from a separate first-half fit.
Both distributions yield 1X2 probabilities, and the full-time matrix yields
the three most likely exact scores.
"""

__version__ = "1.0.0"
