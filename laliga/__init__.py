"""La Liga (Primera División) match prediction.

A time-decayed Dixon–Coles model fits first-half and second-half rates.
Their convolution is one joint (half-time score, full-time score) distribution.
Full-time 1X2, half-time 1X2, exact scores, and the nine HT/FT outcomes are
margins of that distribution. A market track backs the same rates out of
pre-match prices, and a blend mixes the two.
"""

__version__ = "2.0.0"
