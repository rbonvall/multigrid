#!/usr/bin/env python

'''Red-Black Gauss-Seidel relaxation for the 2-D Poisson equation
using Weave'''

from __future__ import division
from numpy import *
from scipy import weave
from scipy.linalg import norm
import time

def red_black_gauss_seidel_step(u, f, h):
    m, n = u.shape
    code = '''
        double h2 = (double) h * (double) h;
        for (int sweep = 0; sweep <= 1; ++sweep)
            for (int i = 1; i < m - 1; ++i)
                for (int j = (sweep ? 2 - i % 2 : 1 + i % 2); j < n - 1; j += 2)
                    u(i, j) = (u(i + 1, j) +
                               u(i - 1, j) +
                               u(i, j + 1) +
                               u(i, j - 1) +
                               h2 * f(i, j)) * 0.25;
    '''
    weave.inline(code, 'm n u f h'.split(),
                 type_converters=weave.converters.blitz,
                 verbose=1)


def pure_python_red_black_gauss_seidel_step(u, f, h):
    m, n = u.shape
    h2 = h * h;
    for sweep in ('red', 'black'):
        for i in range(1, m - 1):
            start = 1 + i % 2 if sweep == 'red' else 2 - i % 2
            for j in range(start, n - 1, 2):
                u[i, j] = (u[i + 1, j] +
                           u[i - 1, j] +
                           u[i, j + 1] +
                           u[i, j - 1] +
                           h2 * f[i, j]) * 0.25


def main():
    from pylab import figure, subplot, contour, contourf, title, show, colorbar
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-n', '--grid-size', type=int, default=64)
    parser.add_option('-i', '--iterations', type=int, default=0)
    parser.add_option('-p', '--pure-python', action='store_true', default=False)
    parser.add_option('-v', '--verbose', action='store_true', default=False)
    (options, args) = parser.parse_args()

    if options.pure_python:
        rbgs_step = pure_python_red_black_gauss_seidel_step
    else:
        rbgs_step = red_black_gauss_seidel_step

    # create grid for test problem
    N = options.grid_size
    i_dom = linspace(0.0, 1.0, N + 1)
    j_dom = linspace(0.0, 1.0, N + 1)
    i, j = meshgrid(i_dom, j_dom)
    h = i_dom[1] - i_dom[0]

    # create model problem, from Multigrid Tutorial slides
    def source(i, j):
        return 2 * ((1 - 6 * i**2) * j**2 * (1 - j**2) +
                    (1 - 6 * j**2) * i**2 * (1 - i**2))
    f = source(i, j)
    u = zeros_like(f)

    # exact solution of the model problem
    def solution(i, j):
        return (i**2 - i**4) * (j**4 - j**2)
    exact_u = solution(i, j)

    # subplot grid size
    M, N = 4, 6

    def add_plot(k, fn, plot_title, *args):
        subplot(M, N, k)
        contour(i, j, fn, *args)
        colorbar()
        title(plot_title)

    add_plot(1, exact_u, 'Exact solution')
    add_plot(2, f, 'Source term')
    add_plot(3, u, 'Initial guess')

    if options.iterations:
        iterations_per_plot = int(ceil(options.iterations / (M * N - 3)))
    else:
        iterations_per_plot = 100
    iteration = 0
    new_error = norm(u - exact_u)

    if options.verbose:
        print 'Problem size:', options.grid_size
        print 'Total iterations:', options.iterations
        print 'Iterations per plot:', iterations_per_plot

    for p in range(4, M * N + 1):
        # relaxation steps
        start = time.clock()
        for k in range(iterations_per_plot):
            rbgs_step(u, f, h)
        avg_iteration_time = (time.clock() - start) / iterations_per_plot
        previous_error, new_error = new_error, norm(u - exact_u)
        if options.verbose:
            improval = abs((previous_error - new_error)/previous_error)
            print 'k=%d to %d' % (iteration, iteration + iterations_per_plot),
            print '[avg improval: %f]' % (improval/iterations_per_plot),
            print '[avg time: %f]' % (avg_iteration_time)
        iteration += iterations_per_plot
        add_plot(p, u, 'k = %d (error = %f)' % (iteration, new_error))

    show()



if __name__ == '__main__':
    main()
