from dedalus_sphere import ball_wrapper as ball
from dedalus_sphere import ball128
import numpy as np
import dedalus.public as de
from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from dedalus_sphere import timesteppers
import os
from evaluator import FileHandler

import equations
from simpleball import SimpleBall, StateVector

import logging
logger = logging.getLogger(__name__)


# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Spatial discretization
N = 127
L_max = N
N_max = N
R_max = 3
alpha_BC = 0
L_dealias = 3/2
N_dealias = 3/2

# Physical parameters
R = np.sqrt(3) / 2
L = 1
dx = R / N
Prandtl = 1
Danger = 20
Rayleigh = Danger * Prandtl * (L / dx)**(8/3)
t_ff = (Rayleigh * Prandtl)**(-1/2)

# Volume penalization
epsilon = (L / dx)**2
delta = dx

# Temporal discretization
t_end = 100 * t_ff
dt = 1 / epsilon / 2
dt_max = 1 / epsilon / 2
safety = 0.4
threshold = 0.1
snapshots_cadence = t_ff/10
snapshots_time = -1e-20


# Domain
mesh = [12,32]
simpleball = SimpleBall(L_max, N_max, R_max, L_dealias, N_dealias, mesh=mesh)
domain = simpleball.domain
B = simpleball.B

u  = ball.TensorField_3D(1,B,domain)
p  = ball.TensorField_3D(0,B,domain)
T  = ball.TensorField_3D(0,B,domain)

u_rhs = ball.TensorField_3D(1,B,domain)
p_rhs = ball.TensorField_3D(0,B,domain)
T_rhs = ball.TensorField_3D(0,B,domain)

psi = ball.TensorField_3D(0,B,domain)
noise = ball.TensorField_3D(0,B,domain)

# initial condition
r = R * simpleball.r
phi = simpleball.phi
theta = simpleball.theta

gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
slices = domain.dist.grid_layout.slices(scales=domain.dealias)
rand = np.random.RandomState(seed=42)
noise['g'] = rand.standard_normal(gshape)[slices]
amp = 1e-3
T['g'] = amp*noise['g']

x = r * np.cos(phi) * np.sin(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(theta)
Xinf = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))
psi['g'] = 0.5 + 0.5*np.tanh((Xinf - L/2)/delta)

# build state vector
def SVWrap(*args):
    return StateVector(simpleball, args)

state_vector = StateVector(simpleball, [u,p,T])
NL = StateVector(simpleball, [u_rhs,p_rhs,T_rhs])
timestepper = timesteppers.SBDF2(SVWrap, u,p,T)

# build matrices
M,L,P,LU = [],[],[],[]
for ell in range(simpleball.ell_start, simpleball.ell_end+1):
    logger.info('Building pencil ell = %i' %ell)
    N = B.N_max - B.N_min(ell-B.R_max)
    M_ell,L_ell = equations.matrices(B, N, ell, alpha_BC, R, Prandtl)
    M.append(M_ell.astype(np.complex128))
    L.append(L_ell.astype(np.complex128))
    P.append(M_ell.astype(np.complex128))
    LU.append([None])

reducer = GlobalArrayReducer(domain.dist.comm_cart)

data_dir = 'rbc'

if rank == 0:
    if not os.path.exists('{:s}/'.format(data_dir)):
        os.mkdir('{:s}/'.format(data_dir))

theta_index = int((L_max+1)*3/4)
logger.info(theta_index)
snapshots = FileHandler(data_dir,domain,B,max_writes=1)
snapshots.add_task(T, name='T', index=0)
snapshots.add_task(u, name='u0', index=0)
snapshots.add_task(u, name='u1', index=1)
snapshots.add_task(u, name='u2', index=2)
t = 0.

reducer = GlobalArrayReducer(domain.dist.comm_cart)
dr = np.gradient(r[0,0])

def calculate_dt(dt_old):
    local_freq = u['g'][0]/dr + u['g'][1]*(L_max+1) + u['g'][2]*(L_max+1)
    global_freq = reducer.global_max(local_freq)
    if global_freq == 0.:
        dt = np.inf
    else:
        dt = 1 / global_freq
        dt *= safety
    if dt > dt_max:
        dt = dt_max
    if dt < dt_old*(1+threshold) and dt > dt_old*(1-threshold):
        dt = dt_old
    return dt


t_list = []
E_list = []

# timestepping loop
start_time = time.time()
iter = 0

while t<t_end:

    equations.nonlinear(state_vector, NL, t, M, R, Prandtl, Rayleigh, epsilon, psi)

    if iter % 10 == 0:
        E0 = R**3*np.sum(simpleball.weight_r*simpleball.weight_theta*0.5*u['g']**2)*(np.pi)/((L_max+1)*L_dealias)
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        Tmax = np.max(T['g'])
        Tmax = reducer.reduce_scalar(Tmax, MPI.MAX)
        logger.info("iter: {:d}, dt={:e}, t={:e}, E0={:e}, T_max={:e}".format(iter, dt, t,E0,Tmax))
        if rank == 0:
            t_list.append(t)
            E_list.append(E0)

    if t>snapshots_time:
        snapshots_time += snapshots_cadence
        snapshots.process(time.time(),t,dt,iter)

    if iter % 10 == 0:
        dt = calculate_dt(dt)

    timestepper.step(dt, state_vector, B, L, M, P, NL, LU)
    t += dt
    iter += 1

end_time = time.time()
if rank==0:
    print('simulation took: %f' %(end_time-start_time))
    t_list = np.array(t_list)
    E_list = np.array(E_list)

