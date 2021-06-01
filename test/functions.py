"""
Functions to generate data for training.
"""
import numpy as np


def piecewise_linear(knot_xs, knot_ys, n_pts):
    """
    Given a set of knots, interpolate to fill the space between them.

    Parameters
    ----------
        knot_xs : array
        knot_ys : array
        n_pts : int
            Number of points to fill between each knot.

    Returns
    -------
    x, y : arrays
    """

    res_y = np.zeros((len(knot_xs) - 1) * n_pts + 1)
    res_x = np.zeros((len(knot_xs) - 1) * n_pts + 1)

    for idx, x in enumerate(knot_xs[:-1]):
        y = knot_ys[idx]
        for j in range(n_pts):
            res_x[idx * n_pts + j] = (j / n_pts) * (knot_xs[idx + 1] - x) + x
            res_y[idx * n_pts + j] = (j / n_pts) * (knot_ys[idx + 1] - y) + y

    res_x[-1] = knot_xs[-1]
    res_y[-1] = knot_ys[-1]

    return res_x, res_y


def fourier(a0, a, b, period, interval):
    """
    Fourier series function:
        a0 / 2 + sum_{i=1}^N(a_i * cos(2 * pi * i * x / period) + b_i * sin(2 * pi * i * x / period))

    Parameters
    ----------
    a0 : float
    a : array
    b : array
    period : float
    interval : array

    Returns
    -------
    y : array
        Fourier function evaluated at each point on the interval.
    """
    y = np.zeros_like(interval)
    for i, x in enumerate(interval):
        sum = a0 / 2
        for j in range(len(a)):
            sum += a[j] * np.cos((j + 1) * 2 * np.pi * x / period) + \
                   b[j] * np.sin((j + 1) * 2 * np.pi * x / period)

        y[i] = sum

    return y
