# vim: set fileencoding=utf-8 :

from __future__ import division
from numpy import ones
import scipy.sparse

# model problem:
# -u_{xx} - u_{yy} = 2[(1 - 6x^2) y^2 (1 - y^2) + (1 - 6y^2) x^2 (1 - x^2)]
# solution:
# u(x, y) = (x^4 - x^4)(y^4 - y^4)

def problem_f(x, y):
    return 2 * ((1 - 6 * x**2) * y**2 * (1 - y**2) +
                (1 - 6 * y**2) * x**2 * (1 - x**2))

def exact_u(x, y):
    return (x**2 - x**4) * (y**4 - y**2)

# need to solve only (N - 1) x (N - 1) interior points
# (u = 0 at the boundary)

def problem_A(N):
    A_size = (N - 1)**2
    d0 = 4 * ones(A_size)
    d1 = -ones(A_size); d1[N - 2::N - 1] = 0
    d2 = -ones(A_size)
    return scipy.sparse.spdiags([    d2, d1, d0, d1,    d2],
                                [-(N-1), -1,  0,  1, (N-1)],
                                A_size, A_size)

def create_problem(i, j):
    N, h = len(i) + 1, i[0, 1] - i[0, 0]
    A = problem_A(N)
    f = (h**2 * problem_f(i, j)).flatten()
    return A, f

