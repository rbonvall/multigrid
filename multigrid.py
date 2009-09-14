from __future__ import division
from numpy import array, tile, hstack, arange
import scipy.sparse
import scipy.linsolve

def restriction_operator(N, weighting='full'):
    '''Construct restriction operator for mapping a function
    evaluated on (n - 1) x (n - 1) mesh onto a (n/2 - 1) x (n/2 - 1) mesh.'''

    if weighting == 'full':
        c = array([1/16, 1/8, 1/16])
        d = array([1/8,  1/4, 1/8])
    elif weighting == 'half':
        c = array([  0, 1/8, 0])
        d = array([1/8, 1/2, 1/8])
    else:
        raise ValueError('weighting must be "half" or "full"')

    m_h = N - 1
    m_2h = N//2 - 1

    # build column index array
    a = array([0, 1, 2])
    base_row = hstack([a + m_h * i for i in xrange(3)])
    base_block = hstack([base_row + 2 * i for i in xrange(m_2h)])
    col = hstack([base_block + 2 * m_h * i for i in xrange(m_2h)])

    data = tile(hstack([c, d, c]), m_2h**2)
    ptr = arange(9 * m_2h**2 + 1, step=9)
    return scipy.sparse.csr_matrix((data, col, ptr))


def prolongation_operator(n):
    return 4 * restriction_operator(n, 'full').transpose()


def coarse_grid_correction_step(A_h, f_h, u_h, R, P, A_2h):
    u_h, f_h = u_h.flatten(), f_h.flatten()
    r_h = f_h - A_h.matvec(u_h)
    r_2h = R.matvec(r_h)
    e_2h = scipy.linsolve.spsolve(A_2h, r_2h)
    e_h = P.matvec(e_2h)
    u_h += e_h
    return e_h
