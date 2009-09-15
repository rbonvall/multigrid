#!/usr/bin/env python

from __future__ import division
from numpy import *
from pylab import *
import scipy.sparse
import scipy.linalg.iterative

import multigrid
import model_problem
import gs

def main():
    N = 256
    h = 1/N

    i_dom = linspace(0.0, 1.0, N + 1)
    j_dom = linspace(0.0, 1.0, N + 1)
    i, j = meshgrid(i_dom[1:-1], j_dom[1:-1])

    A, f = model_problem.create_problem(i, j)
    u_e = model_problem.exact_u(i, j)

    # operators for coarse grid correction
    R = multigrid.restriction_operator(N)
    P = 4 * R.transpose()
    A_2h = R.matmat(A).matmat(P)

    M, N = 2, 7
    def add_plot(p, fn, t):
        subplot(M, N, p)
        contourf(i, j, fn)
        colorbar()
        title(t)

    add_plot(1, u_e, 'Exact solution')

    u = zeros_like(u_e)
    #add_plot(2, u, 'Initial guess')

    # pre-smoothing
    for k in (3, 4):
        gs.red_black_gauss_seidel_step(u, f, h)
        add_plot(k, u, 'Pre-smoothing step')
        error = abs(u_e - u)
        add_plot(k + N, error, 'Error %d: %f' % (k, norm(error)))

    # coarse grid correction
    e_h = multigrid.coarse_grid_correction_step(A, f, u, R, P, A_2h)
    add_plot(5, u, 'Coarse grid correction')
    error = abs(u_e - u)
    add_plot(5 + N, error, 'Error %d: %f' % (5, norm(error)))

    # post-smoothing
    for k in (6, 7):
        gs.red_black_gauss_seidel_step(u, f, h)
        add_plot(k, u, 'Post-smoothing step')
        error = abs(u_e - u)
        add_plot(k + N, error, 'Error: %f' % norm(error))

    show()


if __name__ == '__main__':
    main()

