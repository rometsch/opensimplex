
# Based on: https://gist.github.com/KdotJPG/b1270127455a94ac5d19

import sys
import numpy as np
from array import array
from ctypes import c_int64
from numpy import floor


STRETCH_CONSTANT_2D = -0.211324865405187    # (1/Math.sqrt(2+1)-1)/2
SQUISH_CONSTANT_2D = 0.366025403784439      # (Math.sqrt(2+1)-1)/2
STRETCH_CONSTANT_3D = -1.0 / 6              # (1/Math.sqrt(3+1)-1)/3
SQUISH_CONSTANT_3D = 1.0 / 3                # (Math.sqrt(3+1)-1)/3
STRETCH_CONSTANT_4D = -0.138196601125011    # (1/Math.sqrt(4+1)-1)/4
SQUISH_CONSTANT_4D = 0.309016994374947      # (Math.sqrt(4+1)-1)/4

NORM_CONSTANT_2D = 47
NORM_CONSTANT_3D = 103
NORM_CONSTANT_4D = 30

DEFAULT_SEED = 0


# Gradients for 2D. They approximate the directions to the
# vertices of an octagon from the center.
GRADIENTS_2D = (
    5,  2,    2,  5,
    -5,  2,   -2,  5,
    5, -2,    2, -5,
    -5, -2,   -2, -5,
)
GRADIENTS_2D = np.array(GRADIENTS_2D)


def overflow(x):
    # Since normal python ints and longs can be quite humongous we have to use
    # this hack to make them be able to overflow
    return c_int64(x).value


class OpenSimplex(object):
    """
    OpenSimplex n-dimensional gradient noise functions.
    """

    def __init__(self, seed=DEFAULT_SEED):
        """
        Initiate the class using a permutation array generated from a 64-bit seed number.
        """
        # Generates a proper permutation (i.e. doesn't merely perform N
        # successive pair swaps on a base array)
        # Have to zero fill so we can properly loop over it later
        perm = self._perm = array("l", [0] * 256)
        perm_grad_index_3D = self._perm_grad_index_3D = array("l", [0] * 256)
        source = array("l", [i for i in range(0, 256)])
        seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
        seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
        seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
        for i in range(255, -1, -1):
            seed = overflow(seed * 6364136223846793005 + 1442695040888963407)
            r = int((seed + 31) % (i + 1))
            if r < 0:
                r += i + 1
            perm[i] = source[r]
            
            source[r] = source[i]
        self._perm = np.array(self._perm)

    def noise2d(self, x, y):
        return noise2d(x, y, self._perm)


def extrapolate2d(xsb, ysb, dx, dy, perm):
    inds = np.bitwise_and(xsb, 0xFF)
    inds = np.bitwise_and(perm[inds] + ysb, 0xFF)
    index = np.bitwise_and(perm[inds], 0x0E)

    g1 = GRADIENTS_2D[index]
    g2 = GRADIENTS_2D[index+1]
    return g1 * dx + g2 * dy


def noise2d(x, y, perm):
    """
    Generate 2D OpenSimplex noise from X,Y coordinates.
    """
    # Place input coordinates onto grid.
    stretch_offset = (x + y) * STRETCH_CONSTANT_2D
    xs = x + stretch_offset
    ys = y + stretch_offset

    # Floor to get grid coordinates of rhombus (stretched square) super-cell origin.
    xsb = floor(xs).astype(np.int32)
    ysb = floor(ys).astype(np.int32)

    # Skew out to get actual coordinates of rhombus origin. We'll need these later.
    squish_offset = (xsb + ysb) * SQUISH_CONSTANT_2D
    xb = xsb + squish_offset
    yb = ysb + squish_offset

    # Compute grid coordinates relative to rhombus origin.
    xins = xs - xsb
    yins = ys - ysb

    # Sum those together to get a value that determines which region we're in.
    in_sum = xins + yins

    # Positions relative to origin point.
    dx0 = x - xb
    dy0 = y - yb

    value = np.zeros(x.shape)

    # Contribution (1,0)
    dx1 = dx0 - 1 - SQUISH_CONSTANT_2D
    dy1 = dy0 - 0 - SQUISH_CONSTANT_2D
    attn1 = 2 - dx1 * dx1 - dy1 * dy1
    m = attn1 > 0
    attn1 = np.power(attn1[m], 4)
    value[m] += attn1 * \
        extrapolate2d(xsb[m] + 1, ysb[m] + 0, dx1[m], dy1[m], perm)

    # Contribution (0,1)
    dx2 = dx0 - 0 - SQUISH_CONSTANT_2D
    dy2 = dy0 - 1 - SQUISH_CONSTANT_2D
    attn2 = 2 - dx2 * dx2 - dy2 * dy2

    m = attn2 > 0
    attn2 = np.power(attn2[m], 4)
    value[m] += attn2 * \
        extrapolate2d(xsb[m] + 0, ysb[m] + 1, dx2[m], dy2[m], perm)

    dx_ext = np.zeros(x.shape, dtype=np.float64)
    dy_ext = np.zeros(x.shape, dtype=np.float64)
    xsv_ext = np.zeros(x.shape, dtype=np.int32)
    ysv_ext = np.zeros(x.shape, dtype=np.int32)

    m = in_sum <= 1  # We're inside the triangle (2-Simplex) at (0,0)
    noise2d_insum_le_1(xsb, ysb, xins, yins, dx0, dy0,
                       in_sum, dx_ext, dy_ext, xsv_ext, ysv_ext, m)

    # else:  # We're inside the triangle (2-Simplex) at (1,1)
    m = ~m
    noise2d_insum_geq_1(xsb, ysb, xins, yins, dx0, dy0,
                        in_sum, dx_ext, dy_ext, xsv_ext, ysv_ext, m)

    # Contribution (0,0) or (1,1)
    attn0 = 2 - dx0 * dx0 - dy0 * dy0

    m = attn0 > 0
    attn0[m] *= attn0[m]
    value[m] += attn0[m] * attn0[m] * \
        extrapolate2d(xsb[m], ysb[m], dx0[m], dy0[m], perm)

    # Extra Vertex
    attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
    m = attn_ext > 0
    attn_ext[m] *= attn_ext[m]
    value[m] += attn_ext[m] * attn_ext[m] * \
        extrapolate2d(xsv_ext[m], ysv_ext[m], dx_ext[m], dy_ext[m], perm)

    return value / NORM_CONSTANT_2D


def noise2d_insum_le_1(xsb, ysb, xins, yins, dx0, dy0, in_sum, dx_ext, dy_ext, xsv_ext, ysv_ext, gm):
    zins = 1 - in_sum
    # (0,0) is one of the closest two triangular vertices
    # if zins > xins or zins > yins:
    m = np.logical_and(gm, np.logical_or(zins > xins, zins > yins))
    # if xins > yins:
    m2 = np.logical_and(gm, np.logical_and(m, xins > yins))
    xsv_ext[m2] = xsb[m2] + 1
    ysv_ext[m2] = ysb[m2] - 1
    dx_ext[m2] = dx0[m2] - 1
    dy_ext[m2] = dy0[m2] + 1
    # else:
    m2 = np.logical_and(gm, np.logical_and(m, xins <= yins))
    xsv_ext[m2] = xsb[m2] - 1
    ysv_ext[m2] = ysb[m2] + 1
    dx_ext[m2] = dx0[m2] + 1
    dy_ext[m2] = dy0[m2] - 1
    # else:  # (1,0) and (0,1) are the closest two vertices.
    m = np.logical_and(gm, ~m)
    xsv_ext[m] = xsb[m] + 1
    ysv_ext[m] = ysb[m] + 1
    dx_ext[m] = dx0[m] - 1 - 2 * SQUISH_CONSTANT_2D
    dy_ext[m] = dy0[m] - 1 - 2 * SQUISH_CONSTANT_2D


def noise2d_insum_geq_1(xsb, ysb, xins, yins, dx0, dy0, in_sum, dx_ext, dy_ext, xsv_ext, ysv_ext, gm):
    zins = 2 - in_sum
    # (0,0) is one of the closest two triangular vertices
    # if zins < xins or zins < yins:
    m = np.logical_and(gm, np.logical_or(zins < xins, zins < yins))
    # if xins > yins:
    m2 = np.logical_and(gm, np.logical_and(m, xins > yins))
    xsv_ext[m2] = xsb[m2] + 2
    ysv_ext[m2] = ysb[m2] + 0
    dx_ext[m2] = dx0[m2] - 2 - 2 * SQUISH_CONSTANT_2D
    dy_ext[m2] = dy0[m2] + 0 - 2 * SQUISH_CONSTANT_2D
    # else:
    m2 = np.logical_and(gm, np.logical_and(m, xins <= yins))
    xsv_ext[m] = xsb[m] + 0
    ysv_ext[m] = ysb[m] + 2
    dx_ext[m] = dx0[m] + 0 - 2 * SQUISH_CONSTANT_2D
    dy_ext[m] = dy0[m] - 2 - 2 * SQUISH_CONSTANT_2D
    # else:  # (1,0) and (0,1) are the closest two vertices.
    m = np.logical_and(gm, ~m)
    dx_ext[m] = dx0[m]
    dy_ext[m] = dy0[m]
    xsv_ext[m] = xsb[m]
    ysv_ext[m] = ysb[m]

    xsb[gm] += 1
    ysb[gm] += 1
    dx0[gm] = dx0[gm] - 1 - 2 * SQUISH_CONSTANT_2D
    dy0[gm] = dy0[gm] - 1 - 2 * SQUISH_CONSTANT_2D
