
import numpy as np
from dedalus_sphere import ball128
from scipy import sparse


def BC_rows(N):
    N0 = N
    N1 = N + N0 + 1
    N2 = N + N1 + 1
    N3 = N + N2 + 1
    N4 = N + N3 + 1
    N5 = N + N4 + 1
    N6 = N + N5 + 1

    return N0,N1,N2,N3,N4,N5,N6

def matrices(B, N, ell, Prandtl, eta, alpha_BC):

    def D(mu,i,deg):
        if mu == +1: return B.op('D+',N,i,ell+deg)
        if mu == -1: return B.op('D-',N,i,ell+deg)

    def E(i,deg):
        return B.op('E',N,i,ell+deg)

    def C(deg):
        return ball128.connection(N,ell+deg,alpha_BC,2)

    Z = B.op('0',N,0,ell)

    N0,N1,N2,N3,N4,N5,N6 = BC_rows(N)

    if ell == 0:
        I = B.op('I',N,0,ell).tocsr()
        M44 = E(1, 0).dot(E( 0, 0))
        M = sparse.bmat([[Z,Z,Z,Z,  Z,  Z, Z],
                         [Z,Z,Z,Z,  Z,  Z, Z],
                         [Z,Z,Z,Z,  Z,  Z, Z],
                         [Z,Z,Z,Z,  Z,  Z, Z],
                         [Z,Z,Z,Z,M44,  Z, Z],
                         [Z,Z,Z,Z,  Z,M44, Z],
                         [Z,Z,Z,Z,  Z,  I, Z]]).tocsr()
        L44 = -D(-1,1,+1).dot(D(+1, 0, 0))
        L = sparse.bmat([[I,Z,Z,Z,  Z,  Z, Z],
                         [Z,I,Z,Z,  Z,  Z, Z],
                         [Z,Z,I,Z,  Z,  Z, Z],
                         [Z,Z,Z,I,  Z,  Z, Z],
                         [Z,Z,Z,Z,L44,  Z, Z],
                         [Z,Z,Z,Z,  Z,L44, Z],
                         [Z,Z,Z,Z,  Z,  Z,-I]]).tocsr()

        row0=np.concatenate(( np.zeros(N3+1), B.op('r=1',N,1,ell+1) @ D(+1,0,0) , np.zeros(N6-N4) ))
        row1=np.concatenate(( np.zeros(N4+1), B.op('r=1',N,1,ell+1) @ D(+1,0,0) , np.zeros(N6-N5) ))

        tau0 = C(0)[:,-1]
        tau0 = tau0.reshape((len(tau0),1))

        col0 = np.concatenate((np.zeros((N3+1,1)),tau0,np.zeros((N6-N4,1))))
        col1 = np.concatenate((np.zeros((N4+1,1)),tau0,np.zeros((N6-N5,1))))

        L = sparse.bmat([[   L, col0, col1],
                         [row0,    0,    0],
                         [row1,    0,    0]])

        M = sparse.bmat([[     M, 0*col0, 0*col1],
                         [0*row0,      0,      0],
                         [0*row1,      0,      0]])

        L = L.tocsr()
        M = M.tocsr()

        return M, L

    xim, xip = B.xi([-1,+1],ell)

    M00 = 1/Prandtl*E(1,-1).dot(E( 0,-1))
    M11 = 1/Prandtl*E(1, 0).dot(E( 0, 0))
    M22 = 1/Prandtl*E(1,+1).dot(E( 0,+1))
    M44 = E(1, 0).dot(E( 0, 0))
    M55 = E(1, 0).dot(E( 0, 0))
    I = B.op('I',N,0,ell).tocsr()

    M=sparse.bmat([[M00, Z,   Z, Z,   Z,   Z, Z],
                   [Z, M11,   Z, Z,   Z,   Z, Z],
                   [Z,   Z, M22, Z,   Z,   Z, Z],
                   [Z,   Z,   Z, Z,   Z,   Z, Z],
                   [Z,   Z,   Z, Z, M44,   Z, Z],
                   [Z,   Z,   Z, Z,   Z, M55, Z],
                   [Z,   Z,   Z, Z,   Z,   I, Z]])
    M = M.tocsr()

    L00 = -D(-1,1, 0).dot(D(+1, 0,-1)) + E(1,-1).dot(E( 0,-1))/eta
    L11 = -D(-1,1,+1).dot(D(+1, 0, 0)) + E(1, 0).dot(E( 0, 0))/eta
    L22 = -D(+1,1, 0).dot(D(-1, 0,+1)) + E(1,+1).dot(E( 0,+1))/eta
    L44 = -D(-1,1,+1).dot(D(+1, 0, 0))
    L55 = -D(-1,1,+1).dot(D(+1, 0, 0))

    L03 = xim*E(+1,-1).dot(D(-1,0,0))
    L23 = xip*E(+1,+1).dot(D(+1,0,0))

    L30 = xim*D(+1,0,-1)
    L32 = xip*D(-1,0,+1)

    L04 = Z
    L24 = Z

    L=sparse.bmat([[L00,  Z,   Z, L03, L04,   Z,  Z],
                   [Z,  L11,   Z,   Z,   Z,   Z,  Z],
                   [Z,    Z, L22, L23, L24,   Z,  Z],
                   [L30,  Z, L32,   Z,   Z,   Z,  Z],
                   [Z,    Z,   Z,   Z, L44,   Z,  Z],
                   [Z,    Z,   Z,   Z,   Z, L55,  Z],
                   [Z,    Z,   Z,   Z,   Z,   Z, -I]])
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
    N0, N1, N2, N3, N4, N5, N6 = BC_rows(N)

    Dr_scalar = ( B.Q[(ell,1)][1,0]*B.op('r=1',N,1,ell-1) @ D(-1,0,0)
                + B.Q[(ell,1)][1,2]*B.op('r=1',N,1,ell+1) @ D(+1,0,0) )

#    row0=np.concatenate(( QSm[1]+QSm[3], QS0[1]+QS0[3] , QSp[1]+QSp[3], np.zeros(N6-N2)))
#    row1=np.concatenate(( u0m          , np.zeros(N0+1), u0p          , np.zeros(N6-N2)))
#    row2=np.concatenate(( QSm[5]+QSm[7], QS0[5]+QS0[7] , QSp[5]+QSp[7], np.zeros(N6-N2)))
    row0=np.concatenate(( B.op('r=1',N,0,ell-1), np.zeros(N6-N0)))
    row1=np.concatenate(( np.zeros(N0+1), B.op('r=1',N,0,ell), np.zeros(N6-N1)))
    row2=np.concatenate(( np.zeros(N1+1), B.op('r=1',N,0,ell+1), np.zeros(N6-N2)))
    row3=np.concatenate(( np.zeros(N3+1), Dr_scalar, np.zeros(N6-N4) ))
    row4=np.concatenate(( np.zeros(N4+1), Dr_scalar , np.zeros(N6-N5) ))

    tau0 = C(-1)[:,-1]
    tau1 = C( 0)[:,-1]
    tau2 = C( 1)[:,-1]
    tau3 = C( 0)[:,-1]
    tau4 = C( 0)[:,-1]

    tau0 = tau0.reshape((len(tau0),1))
    tau1 = tau1.reshape((len(tau1),1))
    tau2 = tau2.reshape((len(tau2),1))
    tau3 = tau3.reshape((len(tau3),1))
    tau4 = tau4.reshape((len(tau4),1))

    col0 = np.concatenate((                   tau0,np.zeros((N6-N0,1))))
    col1 = np.concatenate((np.zeros((N0+1,1)),tau1,np.zeros((N6-N1,1))))
    col2 = np.concatenate((np.zeros((N1+1,1)),tau2,np.zeros((N6-N2,1))))
    col3 = np.concatenate((np.zeros((N3+1,1)),tau3,np.zeros((N6-N4,1))))
    col4 = np.concatenate((np.zeros((N4+1,1)),tau4,np.zeros((N6-N5,1))))

    L = sparse.bmat([[   L, col0, col1, col2, col3, col4],
                     [row0,    0 ,   0,    0,    0,    0],
                     [row1,    0 ,   0,    0,    0,    0],
                     [row2,    0,    0,    0,    0,    0],
                     [row3,    0,    0,    0,    0,    0],
                     [row4,    0,    0,    0,    0,    0]])

    M = sparse.bmat([[     M, 0*col0, 0*col1, 0*col2, 0*col3, 0*col4],
                     [0*row0,      0,      0,      0,      0,      0],
                     [0*row1,      0,      0,      0,      0,      0],
                     [0*row2,      0,      0,      0,      0,      0],
                     [0*row3,      0,      0,      0,      0,      0],
                     [0*row4,      0,      0,      0,      0,      0]])

    L = L.tocsr()
    M = M.tocsr()

    return M, L

