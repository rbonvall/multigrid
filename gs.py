#!/usr/bin/env python

'''Red-Black Gauss-Seidel relaxation for the 2-D Poisson equation
using Weave'''

from __future__ import division
from numpy import *
from scipy import weave
from scipy.linalg import norm

def red_black_gauss_seidel_step(u, f, h):
    m, n = u.shape
    code = '''
        double h2 = (double) h * (double) h;
        for (int sweep = 0; sweep <= 1; ++sweep)
            for (int i = 1; i < m; ++i)
                for (int j = (sweep ? 2 - i % 2 : 1 + i % 2); j < n; j += 2)
                    u(i, j) = (u(i + 1, j) +
                               u(i - 1, j) +
                               u(i, j + 1) +
                               u(i, j - 1) +
                               h2 * f(i, j)) * 0.25;
    '''
    weave.inline(code, 'm n u f h'.split(),
                 type_converters=weave.converters.blitz,
                 verbose=1)
    

def pure_python_rbgs(u, f, h):
    m, n = u.shape
    h2 = h * h;
    for sweep in ('red', 'black'):
        for i in range(1, m - 1):
           for j in range(1 + i % 2 if sweep == 'red' else 2 - i % 2, n - 1, 2):
                u[i, j] = (u[i + 1, j] +
                           u[i - 1, j] +
                           u[i, j + 1] +
                           u[i, j - 1] +
                           h2 * f[i, j]) * 0.25


def main():
    # create grid for test problem
    N = 64
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

    from pylab import figure, subplot, contour, contourf, title, show, colorbar
    
    figure(1)
    # subplot grid size
    M, N = 4, 6

    subplot(M, N, 1)
    contour(i, j, exact_u)
    colorbar()
    title('Exact solution')
    
    subplot(M, N, 2)
    contour(i, j, f)
    colorbar()
    title('Source term')

    subplot(M, N, 3)
    contour(i, j, u)
    title('Initial guess')

    iteration = 0
    iterations_per_plot = 100
    for p in range(4, M * N + 1):
        # relaxation steps
        for k in range(iterations_per_plot):
            #pure_python_rbgs(u, f, h)
            red_black_gauss_seidel_step(u, f, h)

        iteration += iterations_per_plot
        print iteration

        error_norm = norm(u - exact_u)

        subplot(M, N, p)
        contour(i, j, u)
        colorbar()
        title('k = %d (error = %f)' % (iteration, error_norm))

    show()



if __name__ == '__main__':
    main()
