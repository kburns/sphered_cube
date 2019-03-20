
import numpy as np
from dedalus_sphere import ball128, ball_wrapper
from scipy import sparse


def BC_rows(N):
    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1
    N4 = N + N3 + 1
    return N0,N1,N2,N3,N4

def matrices(B, N, ell, Prandtl, eta, alpha_BC):

    def D(mu,i,deg):
        if mu == +1: return B.op('D+',N,i,ell+deg)
        if mu == -1: return B.op('D-',N,i,ell+deg)

    def E(i,deg):
        return B.op('E',N,i,ell+deg)

    def C(deg):
        return ball128.connection(N,ell+deg,alpha_BC,2)

    Z = B.op('0',N,0,ell)

    N0,N1,N2,N3,N4 = BC_rows(N)

    if ell == 0:
        I = B.op('I',N,0,ell).tocsr()
        M44 = E(1, 0).dot(E( 0, 0))
        M = sparse.bmat([[Z,Z,Z,Z,  Z],
                         [Z,Z,Z,Z,  Z],
                         [Z,Z,Z,Z,  Z],
                         [Z,Z,Z,Z,  Z],
                         [Z,Z,Z,Z,M44]]).tocsr()
        L44 = -D(-1,1,+1).dot(D(+1, 0, 0))
        L = sparse.bmat([[I,Z,Z,Z,  Z],
                         [Z,I,Z,Z,  Z],
                         [Z,Z,I,Z,  Z],
                         [Z,Z,Z,I,  Z],
                         [Z,Z,Z,Z,L44]]).tocsr()

        row0=np.concatenate(( np.zeros(N3+1), B.op('r=1',N,0,ell) ))

        tau0 = C(0)[:,-1]
        tau0 = tau0.reshape((len(tau0),1))

        col0 = np.concatenate((np.zeros((N3+1,1)),tau0))

        L = sparse.bmat([[   L, col0],
                         [row0,    0]])

        M = sparse.bmat([[     M, 0*col0],
                         [0*row0,      0]])

        L = L.tocsr()
        M = M.tocsr()

        return M, L

    xim, xip = B.xi([-1,+1],ell)

    M00 = 1/Prandtl*E(1,-1).dot(E( 0,-1))
    M11 = 1/Prandtl*E(1, 0).dot(E( 0, 0))
    M22 = 1/Prandtl*E(1,+1).dot(E( 0,+1))
    M44 = E(1, 0).dot(E( 0, 0))
    I = B.op('I',N,0,ell).tocsr()

    M=sparse.bmat([[M00, Z,   Z, Z,   Z],
                   [Z, M11,   Z, Z,   Z],
                   [Z,   Z, M22, Z,   Z],
                   [Z,   Z,   Z, Z,   Z],
                   [Z,   Z,   Z, Z, M44]])
    M = M.tocsr()

    L00 = -D(-1,1, 0).dot(D(+1, 0,-1))
    L11 = -D(-1,1,+1).dot(D(+1, 0, 0))
    L22 = -D(+1,1, 0).dot(D(-1, 0,+1))
    L44 = -D(-1,1,+1).dot(D(+1, 0, 0))

    L03 = xim*E(+1,-1).dot(D(-1,0,0))
    L23 = xip*E(+1,+1).dot(D(+1,0,0))

    L30 = xim*D(+1,0,-1)
    L32 = xip*D(-1,0,+1)

    L04 = Z
    L24 = Z

    L=sparse.bmat([[L00,  Z,   Z, L03, L04],
                   [Z,  L11,   Z,   Z,   Z],
                   [Z,    Z, L22, L23, L24],
                   [L30,  Z, L32,   Z,   Z],
                   [Z,    Z,   Z,   Z, L44]])
    L = L.tocsr()

    Q = B.Q[(ell,2)]
    if ell == 1: rDmm = 0.*B.op('r=1',N,1,ell)
    else: rDmm = B.xi(-1,ell-1)*B.op('r=1',N,1,ell-2)*D(-1,0,-1)
    rDpm = B.xi(+1,ell-1)*B.op('r=1',N,1,ell  )*D(+1,0,-1)
    rDm0 = B.xi(-1,ell  )*B.op('r=1',N,1,ell-1)*D(-1,0, 0)
    rDp0 = B.xi(+1,ell  )*B.op('r=1',N,1,ell+1)*D(+1,0, 0)
    rDmp = B.xi(-1,ell+1)*B.op('r=1',N,1,ell  )*D(-1,0,+1)
    rDpp = B.xi(+1,ell+1)*B.op('r=1',N,1,ell+2)*D(+1,0,+1)

    rD = np.array([rDmm, rDm0, rDmp, 0.*rDmm, 0.*rDm0, 0.*rDmp, rDpm, rDp0, rDpp])
    QSm = Q[:,::3].dot(rD[::3])
    QS0 = Q[:,1::3].dot(rD[1::3])
    QSp = Q[:,2::3].dot(rD[2::3])
    u0m = B.op('r=1',N,0,ell-1)*B.Q[(ell,1)][1,0]
    u0p = B.op('r=1',N,0,ell+1)*B.Q[(ell,1)][1,2]
    N0, N1, N2, N3, N4 = BC_rows(N)

    row0=np.concatenate(( B.op('r=1',N,0,ell-1), np.zeros(N4-N0)))
    row1=np.concatenate(( np.zeros(N0+1), B.op('r=1',N,0,ell), np.zeros(N4-N1)))
    row2=np.concatenate(( np.zeros(N1+1), B.op('r=1',N,0,ell+1), np.zeros(N4-N2)))
    row3=np.concatenate(( np.zeros(N3+1), B.op('r=1',N,0,ell) ))

    tau0 = C(-1)[:,-1]
    tau1 = C( 0)[:,-1]
    tau2 = C( 1)[:,-1]
    tau3 = C( 0)[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))
    tau3 = tau3.reshape((len(tau3),1))

    col0 = np.concatenate((                   tau0,np.zeros((N4-N0,1))))
    col1 = np.concatenate((np.zeros((N0+1,1)),tau1,np.zeros((N4-N1,1))))
    col2 = np.concatenate((np.zeros((N1+1,1)),tau2,np.zeros((N4-N2,1))))
    col3 = np.concatenate((np.zeros((N3+1,1)),tau3))

    L = sparse.bmat([[   L, col0, col1, col2, col3],
                     [row0,    0 ,   0,    0,    0],
                     [row1,    0 ,   0,    0,    0],
                     [row2,    0,    0,    0,    0],
                     [row3,    0,    0,    0,    0]])

    M = sparse.bmat([[     M, 0*col0, 0*col1, 0*col2, 0*col3],
                     [0*row0,      0,      0,      0,      0],
                     [0*row1,      0,      0,      0,      0],
                     [0*row2,      0,      0,      0,      0],
                     [0*row3,      0,      0,      0,      0]])

    L = L.tocsr()
    M = M.tocsr()
    return M, L

# calculate RHS terms from state vector
def nonlinear(state_vector, RHS, t, M, Prandtl, Rayleigh, eta, psi):
    sb = state_vector.simpleball
    B = sb.B

    DT = ball_wrapper.TensorField_3D(1, sb.B, sb.domain)
    om = ball_wrapper.TensorField_3D(1, sb.B, sb.domain)

    u, p, T = state_vector.tensors
    u_rhs, p_rhs, T_rhs = RHS.tensors

    # get U in coefficient space
    state_vector.unpack([u,p,T])

    DT.layout = 'c'
    om.layout = 'c'
    # take derivatives
    for ell in range(sb.ell_start,sb.ell_end+1):
        ell_local = ell - sb.ell_start
        B.curl(ell,1,u['c'][ell_local],om['c'][ell_local])
        DT['c'][ell_local] = B.grad(ell,0,T['c'][ell_local])

    # R = ez cross u
    theta = sb.theta
    r = sb.r
    ez = np.array([np.cos(theta), -np.sin(theta), 0*np.cos(theta)])
    z = r*np.cos(theta)
    T_ref = -z
    u_rhs.layout = 'g'
    T_rhs.layout = 'g'
    u_rhs['g'] = B.cross_grid(u['g'],om['g'])/Prandtl
    u_rhs['g'] -= 1/eta*psi['g']*u['g']
    u_rhs['g'][0] += Rayleigh*ez[0]*T['g'][0]
    u_rhs['g'][1] += Rayleigh*ez[1]*T['g'][0]
    T_rhs['g'][0] = - (u['g'][0]*DT['g'][0] + u['g'][1]*DT['g'][1] + u['g'][2]*DT['g'][2]) - 1/eta*psi['g']*T['g']
    T_rhs['g'][0] += u['g'][0]*ez[0] + u['g'][1]*ez[1]

    # transform (ell, r) -> (ell, N)
    for ell in range(sb.ell_start, sb.ell_end+1):
        ell_local = ell - sb.ell_start

        N = sb.N_max - B.N_min(ell-sb.R_max)

        # multiply by conversion matrices (may be very important)
        u_len = u_rhs['c'][ell_local].shape[0]
        u_rhs['c'][ell_local] = M[ell_local][:u_len,:u_len].dot(u_rhs['c'][ell_local])*Prandtl
        p_len = p_rhs['c'][ell_local].shape[0]
        T_rhs['c'][ell_local] = M[ell_local][u_len+p_len:u_len+2*p_len,u_len+p_len:u_len+2*p_len].dot(T_rhs['c'][ell_local])

    RHS.pack([u_rhs,p_rhs,T_rhs])
