#!/usr/bin/env python

from __future__ import division
from numpy import *
from scipy.linalg import norm
import scipy.sparse
from pylab import *
from itertools import count
from sys import exit

# homogeneous model problem:
# -u[j - 1] + 2u[j] - u[j + 1] = 0, 1 <= j <= n - 1
# u[0] = u[n] = 0

def model_problem(n):
    d0 = 2 * ones(n - 1)
    d1 = -ones(n - 1)
    return scipy.sparse.spdiags([d1, d0, d1], [-1, 0, 1], n - 1, n - 1)

def fourier_mode(k, n):
    j = arange(1, n)
    return sin(j * k * pi / n)

def weighted_jacobi(A, v, w=1, nr_iterations=1):
    D_inv = scipy.sparse.extract_diagonal(A) ** -1

def main(): 
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-n', help='Problem size', type=int, default=64)
    parser.add_option('-f', help='Fourier mode', type=int, default=1)
    parser.add_option('-p', help='Number of plots', type=int, default=50)
    parser.add_option('-i', help='Iterations per plot', type=int, default=1)
    parser.add_option('-w', help='Jacobi weight', type=float, default=2/3)
    options, args = parser.parse_args()

    A = model_problem(options.n)
    D_inv = scipy.sparse.spdiags([scipy.sparse.extract_diagonal(A) ** -1], [0], *A.shape)

    # initial guess: Fourier modes
    v = fourier_mode(options.f, options.n)

    M = int(sqrt(ceil(options.p/2)))
    N = 2 * M

    def add_plot(p, k):
        subplot(M, N, p)
        plot(v)
        title('%d' % k)
        ylim(-1, 1)

    for k in count():
        if k % options.i == 0:
            plot_nr = 1 + k // options.i
            add_plot(plot_nr, k)
            if plot_nr >= M * N:
                break

        # weighted Jacobi in stationary linear iteration form
        r = -A.matvec(v)
        v = v + options.w * D_inv.matvec(r)


    show()

if __name__ == '__main__':
    main()
