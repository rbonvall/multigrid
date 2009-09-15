from __future__ import division
from numpy import array, tile, hstack, arange, zeros
import scipy.sparse
import scipy.linsolve
from scipy import weave

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


def coarse_grid_correction_step_with_operators(A_h, f_h, u_h, R, P, A_2h):
    u_h, f_h = u_h.flatten(), f_h.flatten()
    r_h = f_h - A_h.matvec(u_h)
    r_2h = R.matvec(r_h)
    e_2h = scipy.linsolve.spsolve(A_2h, r_2h)
    e_h = P.matvec(e_2h)
    u_h += e_h
    return e_h


def coarse_grid_correction_step(A_h, f_h, u_h, A_2h):
    r_h = f_h.flatten() - A_h.matvec(u_h.flatten())
    r_h = r_h.reshape(u_h.shape)
    r_2h = restrict(r_h)
    e_2h = scipy.linsolve.spsolve(A_2h, r_2h.flatten())
    e_2h = e_2h.reshape(r_2h.shape)
    e_h = prolongate(e_2h)
    u_h += e_h
    return e_h


def restrict(u_h, weighting='full'):
    m_h, n_h = u_h.shape
    m_2h, n_2h = (m_h + 1)//2 - 1, (n_h + 1)//2 - 1,
    u_2h = zeros((m_2h, n_2h))
    center, side, corner = ({
        'full':   (4, 2, 1),
        'half':   (4, 1, 0),
        'inject': (1, 0, 0),
    })[weighting]
    divider = center + 4 * side + 4 * corner
    code = '''
        for (int i = 0; i < m_2h; ++i)
            for (int j = 0; j < n_2h; ++j)
                u_2h(i, j) = (center * (u_h(2 * i + 1, 2 * j + 1)) +
                              side   * (u_h(2 * i    , 2 * j + 1) +
                                        u_h(2 * i + 2, 2 * j + 1) +
                                        u_h(2 * i + 1, 2 * j    ) +
                                        u_h(2 * i + 1, 2 * j + 2)) +
                              corner * (u_h(2 * i    , 2 * j    ) +
                                        u_h(2 * i + 2, 2 * j    ) +
                                        u_h(2 * i    , 2 * j + 2) +
                                        u_h(2 * i + 2, 2 * j + 2))) / divider;
    '''
    weave.inline(code, 'm_2h n_2h u_2h u_h center side corner divider'.split(),
                 type_converters=weave.converters.blitz,
                 verbose=1)
    return u_2h

def prolongate(u_2h):
    m_2h, n_2h = u_2h.shape
    m_h, n_h = (m_2h + 1) * 2 - 1, (n_2h + 1) * 2 - 1,
    u_h = zeros((m_h, n_h))
    code = '''
        for (int i = 0; i < m_2h; ++i)
            for (int j = 0; j < n_2h; ++j) {
                double u(u_2h(i, j));
                double u_half(u/2), u_quarter(u/4);
                u_h(2 * i + 1, 2 * j + 1) =  u;
                u_h(2 * i    , 2 * j + 1) += u_half;
                u_h(2 * i + 2, 2 * j + 1) += u_half;
                u_h(2 * i + 1, 2 * j    ) += u_half;
                u_h(2 * i + 1, 2 * j + 2) += u_half;
                u_h(2 * i    , 2 * j    ) += u_quarter;
                u_h(2 * i + 2, 2 * j    ) += u_quarter;
                u_h(2 * i    , 2 * j + 2) += u_quarter;
                u_h(2 * i + 2, 2 * j + 2) += u_quarter;
            }
    '''
    weave.inline(code, 'm_2h n_2h u_2h u_h'.split(),
                 type_converters=weave.converters.blitz,
                 verbose=1)
    return u_h

