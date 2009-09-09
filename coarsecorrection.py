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
    N = 64
    h = 1/N

    i_dom = linspace(0.0, 1.0, N + 1)
    j_dom = linspace(0.0, 1.0, N + 1)
    i, j = meshgrid(i_dom[1:-1], j_dom[1:-1])

    A, f = model_problem.create_problem(i, j)
    u_e = model_problem.exact_u(i, j).flatten()

    # operators for coarse grid correction
    R = multigrid.restriction_operator(N)
    P = 4 * R.transpose()
    A_2h = R.matmat(A).matmat(P)

    print "Plotting exact solution"
    subplot(231)
    contourf(i, j, u_e.reshape(i.shape))
    colorbar()
    title('Exact solution')

    print "Solving Au = f with conjugate gradient method"
    u, u_info = scipy.linalg.iterative.cg(A, f, maxiter=10)
    error = abs(u - u_e)

    print "Plotting CG solution and error"
    subplot(232)
    contourf(i, j, u.reshape(i.shape))
    colorbar()
    title('CG solution')
    subplot(235)
    contourf(i, j, error.reshape(i.shape))
    colorbar()
    title('CG error (norm = %f)' % norm(error))

    # initial guess
    u = zeros_like(u)

    # pre-smoothing
    for _ in xrange(2):
        gs.red_black_gauss_seidel_step(u.reshape((N - 1, N - 1)), f, h)

    # coarse grid correction
    e_h = multigrid.coarse_grid_correction_step(A, f, u, R, P, A_2h)

    # post-smoothing
    for _ in xrange(2):
        gs.red_black_gauss_seidel_step(u.reshape((N - 1, N - 1)), f, h)
    
    subplot(233)
    contourf(i, j, u.reshape(i.shape))
    colorbar()
    title('Coarse grid correction')

    subplot(236)
    error = abs(u - u_e)
    contourf(i, j, error.reshape(i.shape))
    colorbar()
    title('CG error (norm = %f)' % norm(error))

    show()


if __name__ == '__main__':
    main()

